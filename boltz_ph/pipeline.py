import copy
import os
import random
import sys
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from LigandMPNN.wrapper import LigandMPNNWrapper

from boltz_ph.constants import CHAIN_TO_NUMBER
from utils.metrics import get_CA_and_sequence # Used implicitly in design.py
from utils.convert import calculate_holo_apo_rmsd, convert_cif_files_to_pdb


from model_utils import (
    binder_binds_contacts,
    clean_memory,
    design_sequence,
    get_boltz_model,
    get_cif,
    load_canonicals,
    plot_from_pdb,
    plot_run_metrics,
    process_msa,
    run_prediction,
    sample_seq,
    save_pdb,
    shallow_copy_tensor_dict,
    smart_split,
)

# --- ipAE / ipSAE helpers

def extract_cb_coords(structure, coords):

    atom_coords = coords[0].detach().cpu().numpy()  # (N_atoms, 3)
    cb_list = []

    for chain in structure.chains:
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]

        for res in structure.residues[res_start:res_end]:
            a0 = res["atom_idx"]
            a1 = a0 + res["atom_num"]
            residue_atoms = structure.atoms[a0:a1]

            cb_coord = None
            ca_coord = None

            for local_i, atom in enumerate(residue_atoms):
                name = atom["name"]

                if name == "CB":
                    cb_coord = atom_coords[a0 + local_i]
                    break  # prefer CB immediately

                if name == "CA":
                    ca_coord = atom_coords[a0 + local_i]

            # Use CB if present
            if cb_coord is not None:
                cb_list.append(cb_coord)
                continue

            # Use CA if present (covers Gly automatically)
            if ca_coord is not None:
                cb_list.append(ca_coord)
                continue

            # Fallback: centroid (same as extract_ca_coords)
            meanpos = atom_coords[a0:a1].mean(axis=0)
            cb_list.append(meanpos)

    return np.array(cb_list)




def compute_best_target_ip_metrics(pae, coords, structure,
                                   target_lengths, binder_len):
    """
    Corrected version: assumes chain ordering is:
        [binder][target1][target2]...[targetN]

    Parameters
    ----------
    pae : torch.Tensor [1, L, L]
    coords : torch.Tensor [1, N_atoms, 3]
    structure : Boltz structure
    target_lengths : list[int]
    binder_len : int

    Returns
    -------
    best_ipae : float
    best_ipsae_min : float
    best_target_index : int
    """

    pae_np = pae.detach().cpu().numpy()[0]

    # Extract CA coordinates (same code as your script)
    cb = extract_cb_coords(structure, coords)
    assert cb.shape[0] == (binder_len + sum(target_lengths))

    # binder first in the chain order
    B = np.arange(0, binder_len)

    best_ipae = -1e9
    best_ipsae_min = 0.0
    best_t = -1

    # offset starts AFTER binder
    offset = binder_len

    # Precompute distance matrix
    diff = cb[:, None, :] - cb[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    for t_idx, L_T in enumerate(target_lengths):

        A = np.arange(offset, offset + L_T)
        offset += L_T

        # --- ipAE ---
        block_AB = pae_np[np.ix_(A, B)]
        block_BA = pae_np[np.ix_(B, A)]
        ipae = float(np.mean(np.concatenate([block_AB.ravel(), block_BA.ravel()])))

        # --- ipSAE_min ---
        def _asym_d0res(S, T):
            """
            Ground-truth ipSAE_d0res_asym used in ipsae.py:
            
            For each residue i in S:
                - find valid partner residues j in T with:
                    pae[i,j] < 15
                    dist[i,j] < 15 Å
                - n0res(i) = number of such residues j
                - d0(i) = calc_d0(n0res(i)) with minimum 1.0
                - ipSAE(i) = mean_j PTM(pae[i,j], d0(i))
            Score = max_i ipSAE(i)
            """
            best = 0.0

            for i in S:
                # indices j in T that satisfy both cutoffs
                mask = (pae_np[i, T] < 15.0) & (dist[i, T] < 15.0)
                if not np.any(mask):
                    continue

                vals = pae_np[i, T][mask]
                n0res = len(vals)

                # --- d0res formula ---
                L_eff = max(n0res, 27)
                d0 = max(1.0, 1.24 * (L_eff - 15)**(1/3) - 1.8)

                # PTM scores
                ptm = 1.0 / (1.0 + (vals / d0)**2)
                score_i = float(ptm.mean())

                if score_i > best:
                    best = score_i

            return best


        a_to_b = _asym_d0res(A, B)
        b_to_a = _asym_d0res(B, A)
        ipsae_min = float(min(a_to_b, b_to_a))

        # Track best
        if ipae > best_ipae:
            best_ipae = ipae
            best_ipsae_min = ipsae_min
            best_t = t_idx

    return best_ipae, best_ipsae_min


class InputDataBuilder:
    """Handles parsing command-line arguments and constructing the base Boltz input data dictionary."""

    def __init__(self, args):
        self.args = args
        self.save_dir = (
            args.save_dir if args.save_dir else f"./results_boltz/{args.name}"
        )
        self.protein_hunter_save_dir = f"{self.save_dir}/0_protein_hunter_design"
        os.makedirs(self.protein_hunter_save_dir, exist_ok=True)


    def _process_sequence_inputs(self):
        """
        Parses and groups protein sequences and MSAs from command line arguments.
        Ensures protein_seqs_list and protein_msas_list are aligned and padded.
        """
        a = self.args
        
        protein_seqs_list = smart_split(a.protein_seqs) if a.protein_seqs else []

        # Handle special "empty" case: --protein_msas "empty" applies to all seqs
        if a.msa_mode == "single":
            protein_msas_list = ["empty"] * len(protein_seqs_list)
        elif a.msa_mode == "mmseqs":
            protein_msas_list = ["mmseqs"] * len(protein_seqs_list)
        else:
            raise ValueError(f"Invalid msa_mode: {a.msa_mode}")

        return protein_seqs_list, protein_msas_list


    def build(self):
        """
        Constructs the base Boltz input data dictionary (sequences, templates, constraints).
        """
        a = self.args

        if a.mode == "unconditional":
            data = self._build_unconditional_data()
            pocket_conditioning = False
        else:
            data, pocket_conditioning = self._build_conditional_data()

        # Sort sequences by chain ID for consistent processing
        data["sequences"] = sorted(
            data["sequences"], key=lambda entry: list(entry.values())[0]["id"][0]
        )

        print("Data dictionary:\n", data)
        return data, pocket_conditioning


    def _build_unconditional_data(self):
        """Constructs data for unconditional binder design."""
        data = {
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": "X",
                        "msa": "empty",
                    }
                }
            ]
        }
        return data


    def _build_conditional_data(self):
        """Constructs data for conditioned design (binder + target + optional non-protein)."""
        a = self.args
        protein_seqs_list, protein_msas_list = (
            self._process_sequence_inputs()
        )
        sequences = []
        
        # Assign chain IDs to proteins first
        protein_chain_ids = [chr(ord('B') + i) for i in range(len(protein_seqs_list))]
        # Find next available chain letter
        next_chain_idx = len(protein_chain_ids)
        
        ligand_chain_id = None
        if a.ligand_smiles or a.ligand_ccd:
            ligand_chain_id = chr(ord('B') + next_chain_idx)
            next_chain_idx += 1
            
        nucleic_chain_id = None
        if a.nucleic_seq:
            nucleic_chain_id = chr(ord('B') + next_chain_idx)
            next_chain_idx += 1
        # --- END NEW ---

        # Step 1: Determine canonical MSA for each unique target sequence
        seq_to_indices = defaultdict(list)
        for idx, seq in enumerate(protein_seqs_list):
            if seq:
                seq_to_indices[seq].append(idx)
        
        seq_to_final_msa = {}
        for seq, idx_list in seq_to_indices.items():
            chosen_msa = next(
                (
                    protein_msas_list[i]
                    for i in idx_list
                ),
                None
            )
            chosen_msa = chosen_msa if chosen_msa is not None else ""

            if chosen_msa == "mmseqs":
                pid = protein_chain_ids[idx_list[0]]
                msa_value = process_msa(pid, seq, Path(self.protein_hunter_save_dir))
                seq_to_final_msa[seq] = str(msa_value)
            elif chosen_msa == "empty":
                seq_to_final_msa[seq] = "empty"
            else:
                raise ValueError(f"Invalid msa_mode: {a.msa_mode}")

        # Step 2: Build sequences list for target proteins
        for i, (seq, msa) in enumerate(zip(protein_seqs_list, protein_msas_list)):
            if not seq:
                continue
            pid = protein_chain_ids[i]
            final_msa = seq_to_final_msa.get(seq, "empty")
            sequences.append(
                {
                    "protein": {
                        "id": [pid],
                        "sequence": seq,
                        "msa": final_msa,
                    }
                }
            )

        # Step 3: Add binder chain
        sequences.append(
            {
                "protein": {
                    "id": ["A"], # Hardcoded 'A'
                    "sequence": "X",
                    "msa": "empty",
                    "cyclic": a.cyclic,
                }
            }
        )

        # Step 4: Add ligands/nucleic acids
        if a.ligand_smiles:
            sequences.append(
                {"ligand": {"id": [ligand_chain_id], "smiles": a.ligand_smiles}}
            )
        elif a.ligand_ccd:
            sequences.append({"ligand": {"id": [ligand_chain_id], "ccd": a.ligand_ccd}})
        if a.nucleic_seq:
            sequences.append(
                {a.nucleic_type: {"id": [nucleic_chain_id], "sequence": a.nucleic_seq}}
            )

        # Step 5: Add templates
        templates = self._build_templates(protein_chain_ids)

        data = {"sequences": sequences}
        if templates:
            data["templates"] = templates

        # Step 6: Add constraints (pocket conditioning)
        pocket_conditioning = bool(a.contact_residues and a.contact_residues.strip())
        if pocket_conditioning:
            contacts = []
            residues_chains = a.contact_residues.split("|")
            for i, residues_chain in enumerate(residues_chains):
                residues = residues_chain.split(",")
                contacts.extend([
                    [protein_chain_ids[i], int(res)]
                    for res in residues
                    if res.strip() != ""
                ])
            constraints = [{"pocket": {"binder": "A", "contacts": contacts}}]
            data["constraints"] = constraints

        return data, pocket_conditioning

    def _build_templates(self, protein_chain_ids):
        """
        Constructs the list of template dictionaries.
        """
        a = self.args
        templates = []
        if a.template_path:
            template_path_list = smart_split(a.template_path)
            # We use the internal protein_chain_ids list
            template_cif_chain_id_list = (
                smart_split(a.template_cif_chain_id)
                if a.template_cif_chain_id
                else []
            )
            
            # Use protein_chain_ids to determine the number of expected templates
            num_proteins = len(protein_chain_ids)
            
            # Pad template paths list to match number of proteins
            if len(template_path_list) != num_proteins:
                print(f"Warning: Mismatch between number of proteins ({num_proteins}) and template paths ({len(template_path_list)}). Padding with empty entries.")
                while len(template_path_list) < num_proteins:
                    template_path_list.append("")
            
            # Pad cif chains list to match number of proteins
            if len(template_cif_chain_id_list) != num_proteins:
                print(f"Warning: Mismatch between number of proteins ({num_proteins}) and template CIF chains ({len(template_cif_chain_id_list)}). Padding with empty entries.")
                while len(template_cif_chain_id_list) < num_proteins:
                    template_cif_chain_id_list.append("")
            
            # Now, iterate up to num_proteins, linking them
            for i in range(num_proteins):
                template_file_path = template_path_list[i]
                if not template_file_path:
                    continue # Skip if no template path for this protein
                    
                template_file = get_cif(template_file_path)
                
                t_block = (
                    {"cif": template_file}
                    if template_file.endswith(".cif")
                    else {"pdb": template_file}
                )
                
                t_block["chain_id"] = protein_chain_ids[i] # e.g., 'B'
                
                # Only add cif_chain_id if provided for this template
                cif_chain = template_cif_chain_id_list[i]
                if cif_chain:
                    t_block["cif_chain_id"] = cif_chain # e.g., 'P'
                
                templates.append(t_block)
                
        return templates

class ProteinHunter_Boltz:
    """
    Core class to manage the protein design pipeline, including Boltz structure
    prediction, LigandMPNN design, cycle optimization, and downstream validation.
    """

    def __init__(self, args):
        self.args = args
        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # 1. Initialize shared resources (persists across all runs)
        self.ccd_path = Path(args.ccd_path).expanduser()
        self.ccd_lib = load_canonicals(str(self.ccd_path))
        
        # Initialize the Input Data Builder
        self.data_builder = InputDataBuilder(args)
        
        # 2. Initialize Models
        self.boltz_model = self._load_boltz_model()
        self.designer = LigandMPNNWrapper("./LigandMPNN/run.py")

        # 3. Setup Directories
        self.save_dir = self.data_builder.save_dir
        self.protein_hunter_save_dir = self.data_builder.protein_hunter_save_dir
        self.binder_chain = "A"

        print("✅ ProteinHunter_Boltz initialized.")

    def _load_boltz_model(self):
        """Loads and configures the Boltz model."""
        predict_args = {
            "recycling_steps": self.args.recycling_steps,
            "sampling_steps": self.args.diffuse_steps,
            "diffusion_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": True,
            "write_full_pde": False,
            "max_parallel_samples": 1,
        }
        return get_boltz_model(
            checkpoint=self.args.boltz_model_path,
            predict_args=predict_args,
            device=self.device,
            model_version=self.args.boltz_model_version,
            no_potentials=False if self.args.contact_residues else True,
            grad_enabled=self.args.grad_enabled,
        )

    def _run_design_cycle(self, data_cp, run_id, pocket_conditioning):
        """
        Executes a single design run, including Cycle 0 contact filtering and
        the subsequent design/prediction cycles.
        
        Note: The cycle logic remains highly coupled due to the nature of the
        sequential design process, but relies on modular imports.
        """
        a = self.args
        run_save_dir = os.path.join(self.protein_hunter_save_dir, f"run_{run_id}")
        os.makedirs(run_save_dir, exist_ok=True)

        # Initialize run metrics and tracking variables
        best_iptm = float("-inf")
        best_seq = None
        best_structure = None
        best_output = None
        best_pdb_filename = None
        best_cycle_idx = -1
        best_alanine_percentage = None
        best_ipae = float("-inf")
        best_ipsae_min = float("-inf")
        run_metrics = {"run_id": run_id}


        if a.seq =="":
            binder_length = random.randint(
                a.min_protein_length, a.max_protein_length
            )
        else:
            binder_length = len(a.seq)

        if "constraints" in data_cp:
            protein_chain_ids = sorted({chain for chain, _ in data_cp["constraints"][0]["pocket"]["contacts"]})

        # Helper function to update sequence in the data dictionary
        def update_binder_sequence(new_seq):
            for seq_entry in data_cp["sequences"]:
                if (
                    "protein" in seq_entry
                    and self.binder_chain in seq_entry["protein"]["id"]
                ):
                    seq_entry["protein"]["sequence"] = new_seq
                    return
            # Should not happen if data_cp is built correctly
            raise ValueError("Binder chain not found in data dictionary.")

        # Set initial binder sequence
        if a.seq =="":
            initial_seq = sample_seq(
                binder_length, exclude_P=a.exclude_P, frac_X=a.percent_X/100
            )
        else:
            initial_seq = a.seq
        update_binder_sequence(initial_seq)
        print(f"Binder initial sequence length: {binder_length}")

        # --- Cycle 0 structure prediction, with contact filtering check ---
        contact_filter_attempt = 0
        pdb_filename = ""
        structure = None
        output = None

        while True:
            output, structure = run_prediction(
                data_cp,
                self.binder_chain,
                randomly_kill_helix_feature=a.randomly_kill_helix_feature,
                negative_helix_constant=a.negative_helix_constant,
                boltz_model=self.boltz_model,
                ccd_lib=self.ccd_lib,
                ccd_path=self.ccd_path,
                logmd=a.logmd,
                device=self.device,
                boltz_model_version=a.boltz_model_version,
                pocket_conditioning=pocket_conditioning,
            )
            pdb_filename = f"{run_save_dir}/{a.name}_run_{run_id}_predicted_cycle_0.pdb"
            plddts = output["plddt"].detach().cpu().numpy()[0]
            save_pdb(structure, output["coords"], plddts, pdb_filename)

            contact_check_okay = True
            if a.contact_residues.strip() and not a.no_contact_filter:
                try:
                    binds = all([binder_binds_contacts(
                        pdb_filename,
                        self.binder_chain,
                        protein_chain_ids[i],
                        contact_res,
                        cutoff=a.contact_cutoff,
                    )
                    for i, contact_res in enumerate(a.contact_residues.split("|"))
                ])
                    if not binds:
                        print(
                            "❌ Binder does NOT contact required residues after cycle 0. Retrying..."
                        )
                        contact_check_okay = False
                except Exception as e:
                    print(f"WARNING: Could not perform binder-contact check: {e}")
                    contact_check_okay = True  # Fail open

            if contact_check_okay:
                break
            contact_filter_attempt += 1
            if contact_filter_attempt >= a.max_contact_filter_retries:
                print("WARNING: Max retries for contact filtering reached. Proceeding.")
                break

            # Resample initial sequence
            new_seq = sample_seq(binder_length, exclude_P=a.exclude_P, frac_X=a.percent_X/100)
            update_binder_sequence(new_seq)
            clean_memory()

        clean_memory() # <-- ADD THIS CALL HERE
        # Capture Cycle 0 metrics

        # -------- Determine target_lengths and binder_len --------
        protein_seqs = []
        binder_seq = None

        for entry in data_cp["sequences"]:
            if "protein" in entry:
                cid = entry["protein"]["id"][0]
                seq = entry["protein"]["sequence"]
                if cid == self.binder_chain:
                    binder_len = len(seq)
                else:
                    protein_seqs.append(seq)

        target_lengths = [len(s) for s in protein_seqs]
        # ---------------------------------------------------------


        binder_chain_idx = CHAIN_TO_NUMBER[self.binder_chain]
        pair_chains = output["pair_chains_iptm"]

        # Calculate i-pTM
        if len(pair_chains) > 1:
            values = [
                (
                    pair_chains[binder_chain_idx][i].detach().cpu().numpy()
                    + pair_chains[i][binder_chain_idx].detach().cpu().numpy()
                )
                / 2.0
                for i in range(len(pair_chains))
                if i != binder_chain_idx
            ]
            cycle_0_iptm = float(np.mean(values) if values else 0.0)
        else:
            cycle_0_iptm = 0.0
        
        cycle_0_ipae, cycle_0_ipsae_min = compute_best_target_ip_metrics(
            output["pae"],
            output["coords"],
            structure,
            target_lengths,
            binder_len,
        )


        run_metrics["cycle_0_iptm"] = cycle_0_iptm
        run_metrics["cycle_0_plddt"] = float(
            output.get("complex_plddt", torch.tensor([0.0])).detach().cpu().numpy()[0]
        )
        run_metrics["cycle_0_iplddt"] = float(
            output.get("complex_iplddt", torch.tensor([0.0])).detach().cpu().numpy()[0]
        )
        run_metrics["cycle_0_alanine"] = 0

        run_metrics["cycle_0_ipae"] = cycle_0_ipae
        run_metrics["cycle_0_ipsae_min"] = cycle_0_ipsae_min

        # --- Optimization Cycles ---
        for cycle in range(a.num_cycles):
            print(f"\n--- Run {run_id}, Cycle {cycle + 1} ---")

            # Calculate temperature and bias for the cycle
            cycle_norm = (cycle / (a.num_cycles - 1)) if a.num_cycles > 1 else 0.0
            alpha = a.alanine_bias_start - cycle_norm * (
                a.alanine_bias_start - a.alanine_bias_end
            )

            # 1. Design Sequence
            model_type = (
                "ligand_mpnn"
                if (a.ligand_smiles or a.ligand_ccd or a.nucleic_seq)
                else "soluble_mpnn"
            )
            design_kwargs = {
                "pdb_file": pdb_filename,
                "temperature": a.temperature,
                "chains_to_design": self.binder_chain,
                "omit_AA": f"{a.omit_AA},P" if cycle == 0 else a.omit_AA,
                "bias_AA": f"A:{alpha}" if a.alanine_bias else "",
            }

            seq_str, logits = design_sequence(
                self.designer, model_type, **design_kwargs
            )
            # The output seq_str is a dictionary-like string, we extract the binder chain sequence
            seq = seq_str.split(":")[CHAIN_TO_NUMBER[self.binder_chain]] 

            # Update data_cp with new sequence
            alanine_count = seq.count("A")
            alanine_percentage = (
                alanine_count / binder_length if binder_length != 0 else 0.0
            )
            update_binder_sequence(seq) # Use the helper function

            # 2. Structure Prediction
            output, structure = run_prediction(
                data_cp,
                self.binder_chain,
                seq=seq,
                randomly_kill_helix_feature=False,
                negative_helix_constant=0.0,
                boltz_model=self.boltz_model,
                ccd_lib=self.ccd_lib,
                ccd_path=self.ccd_path,
                logmd=False,
                device=self.device,
            )

            # Calculate ipTM
            current_chain_idx = CHAIN_TO_NUMBER[self.binder_chain]
            pair_chains = output["pair_chains_iptm"]
            if len(pair_chains) > 1:
                values = [
                    (
                        pair_chains[current_chain_idx][i].detach().cpu().numpy()
                        + pair_chains[i][current_chain_idx].detach().cpu().numpy()
                    )
                    / 2.0
                    for i in range(len(pair_chains))
                    if i != current_chain_idx
                ]
                current_iptm = float(np.mean(values) if values else 0.0)
            else:
                current_iptm = 0.0
            curr_ipae, curr_ipsae_min = compute_best_target_ip_metrics(
                output["pae"],
                output["coords"],
                structure,
                target_lengths,
                binder_len,
            )

            # Update best structure (only if alanine content is acceptable)
            if alanine_percentage <= 0.20 and current_iptm > best_iptm:
                best_iptm = current_iptm
                best_seq = seq
                best_structure = copy.deepcopy(structure)
                best_output = shallow_copy_tensor_dict(output)
                best_pdb_filename = (
                    f"{run_save_dir}/{a.name}_run_{run_id}_best_structure.pdb"
                )
                best_plddts = best_output["plddt"].detach().cpu().numpy()[0]
                save_pdb(
                    best_structure,
                    best_output["coords"],
                    best_plddts,
                    best_pdb_filename,
                )
                best_cycle_idx = cycle + 1
                best_alanine_percentage = alanine_percentage

            # 3. Log Metrics and Save PDB
            curr_plddt = float(
                output.get("complex_plddt", torch.tensor([0.0]))
                .detach()
                .cpu()
                .numpy()[0]
            )
            curr_iplddt = float(
                output.get("complex_iplddt", torch.tensor([0.0]))
                .detach()
                .cpu()
                .numpy()[0]
            )

            run_metrics[f"cycle_{cycle + 1}_iptm"] = current_iptm
            run_metrics[f"cycle_{cycle + 1}_plddt"] = curr_plddt
            run_metrics[f"cycle_{cycle + 1}_iplddt"] = curr_iplddt
            run_metrics[f"cycle_{cycle + 1}_alanine"] = alanine_count
            run_metrics[f"cycle_{cycle + 1}_seq"] = seq
            run_metrics[f"cycle_{cycle + 1}_ipae"] = curr_ipae
            run_metrics[f"cycle_{cycle + 1}_ipsae_min"] = curr_ipsae_min

            if curr_ipae > best_ipae:
                best_ipae = curr_ipae
            if curr_ipsae_min > best_ipsae_min:
                best_ipsae_min = curr_ipsae_min

            pdb_filename = (
                f"{run_save_dir}/{a.name}_run_{run_id}_predicted_cycle_{cycle + 1}.pdb"
            )
            plddts = output["plddt"].detach().cpu().numpy()[0]
            save_pdb(structure, output["coords"], plddts, pdb_filename)
            clean_memory()

            print(
                f"ipTM: {current_iptm:.2f} ipAE: {curr_ipae:.2f} ipSAE_min: {curr_ipsae_min:.2f} pLDDT: {curr_plddt:.2f} iPLDDT: {curr_iplddt:.2f} Alanine count: {alanine_count}"
            )

            # 4. Save YAML for High ipTM
            save_yaml_this_design = (alanine_percentage <= 0.20) and (
                current_iptm > a.high_iptm_threshold
                and curr_plddt > a.high_plddt_threshold
            )

            if save_yaml_this_design and a.contact_residues.strip():
                this_cycle_pdb_filename = f"{run_save_dir}/{a.name}_run_{run_id}_predicted_cycle_{cycle + 1}.pdb"
                try:
                    contact_binds = all([binder_binds_contacts(
                        this_cycle_pdb_filename,
                        self.binder_chain,
                        protein_chain_ids[i],
                        contact_res,
                        cutoff=a.contact_cutoff,
                    )
                    for i, contact_res in enumerate(a.contact_residues.split("|"))
                ])
                    if not contact_binds:
                        save_yaml_this_design = False
                        print(
                            "⛔️ Not saving YAML: binder failed contact check for high ipTM save."
                        )
                except Exception as e:
                    print(
                        f"WARNING: Exception during contact check: {e}. Saving YAML anyway."
                    )
            if save_yaml_this_design:
                # --- Define base names and directories ---
                base_filename = f"{a.name}_run_{run_id}_cycle_{cycle + 1}"
                
                # 1. Save YAML
                high_iptm_yaml_dir = os.path.join(self.save_dir, "high_iptm_yaml")
                os.makedirs(high_iptm_yaml_dir, exist_ok=True)
                yaml_base_name = f"{base_filename}_output.yaml"
                yaml_filename = os.path.join(high_iptm_yaml_dir, yaml_base_name)
                
                with open(yaml_filename, "w") as f:
                    yaml.dump(data_cp, f, default_flow_style=False)
                print(f"✅ Saved run {run_id} cycle {cycle + 1} YAML.")

                # 2. Copy PDB
                high_iptm_pdb_dir = os.path.join(self.save_dir, "high_iptm_pdb")
                os.makedirs(high_iptm_pdb_dir, exist_ok=True)
                
                # Use the 'base_filename' for a consistent destination name
                pdb_base_name = f"{base_filename}_structure.pdb" 
                dest_pdb_path = os.path.join(high_iptm_pdb_dir, pdb_base_name)
                
                # Use 'pdb_filename', which is *always* defined just above this block
                # This was the source of the UnboundLocalError
                shutil.copy(pdb_filename, dest_pdb_path)
                print(f"✅ Copied PDB to {dest_pdb_path}")

                # 3. Append to high-ipTM summary CSV
                summary_csv_path = os.path.join(self.save_dir, "summary_high_iptm.csv")
                
                metrics_dict = {
                    "run_id": run_id,
                    "cycle": cycle + 1,
                    "iptm": current_iptm,
                    "plddt": curr_plddt,
                    "iplddt": curr_iplddt,
                    "alanine_count": alanine_count,
                    "sequence": seq,
                    "pdb_filename": pdb_base_name, # Log the *new* filename
                    "yaml_filename": yaml_base_name,
                    "ipae": curr_ipae,
                    "ipsae_min": curr_ipsae_min,
                }
                
                # Check if file exists to write header only once
                file_exists = os.path.isfile(summary_csv_path)
                
                df_row = pd.DataFrame([metrics_dict])
                df_row.to_csv(summary_csv_path, mode='a', header=not file_exists, index=False)
                print(f"✅ Appended high-ipTM metrics to {summary_csv_path}")
        # End of cycle visualization
        if best_structure is not None and a.plot:
            plot_from_pdb(best_pdb_filename)
        elif best_structure is None:
            print(
                f"\nNo structure was generated for run {run_id} (no eligible best design with <= 20% alanine)."
            )

        # Finalize best metrics for CSV
        if best_alanine_percentage is not None and best_alanine_percentage <= 0.20:
            run_metrics["best_iptm"] = float(best_iptm)
            run_metrics["best_cycle"] = best_cycle_idx
            run_metrics["best_plddt"] = float(
                best_output.get("complex_plddt", torch.tensor([0.0]))
                .detach()
                .cpu()
                .numpy()[0]
            )
            run_metrics["best_seq"] = best_seq
            run_metrics["best_ipae"] = float(best_ipae) 
            run_metrics["best_ipsae_min"] = float(best_ipsae_min)
        else:
            run_metrics["best_iptm"] = float("nan")
            run_metrics["best_cycle"] = None
            run_metrics["best_plddt"] = float("nan")
            run_metrics["best_seq"] = None
            run_metrics["best_ipae"] = float("nan")
            run_metrics["best_ipsae_min"] = float("nan")

        if a.plot:
            plot_run_metrics(run_save_dir, a.name, run_id, a.num_cycles, run_metrics)

        return run_metrics

    def _save_summary_metrics(self, all_run_metrics):
        """Saves all run metrics to a single CSV file."""
        a = self.args
        columns = ["run_id"]
        # Columns for cycle 0 through num_cycles
        for i in range(a.num_cycles + 1):
            columns.extend(
                [
                    f"cycle_{i}_iptm",
                    f"cycle_{i}_plddt",
                    f"cycle_{i}_iplddt",
                    f"cycle_{i}_alanine",
                    f"cycle_{i}_seq",
                    f"cycle_{i}_ipae",
                    f"cycle_{i}_ipsae_min",
                ]
            )
        # Best metric columns
        columns.extend(["best_iptm", "best_cycle", "best_plddt", "best_seq", "best_ipae", "best_ipsae_min",])

        summary_csv = os.path.join(self.save_dir, "summary_all_runs.csv")
        df = pd.DataFrame(all_run_metrics)
        
        # Ensure all expected columns are present (filling missing with NaN)
        for col in columns:
            if col not in df.columns:
                df[col] = float("nan")
        
        # Filter to columns in the correct order (and existing ones)
        df = df[[c for c in columns if c in df.columns]]
        df.to_csv(summary_csv, index=False)
        print(f"\n✅ All run/cycle metrics saved to {summary_csv}")

    def _run_downstream_validation(self):
        # This ensures they are in scope when called later in this function
        sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
        from utils.alphafold_utils import run_alphafold_step
        from utils.pyrosetta_utils import run_rosetta_step
        
        """Executes AlphaFold and Rosetta validation steps."""
        a = self.args
        
        # Determine target type for Rosetta validation
        any_ligand_or_nucleic = a.ligand_smiles or a.ligand_ccd or a.nucleic_seq
        if a.nucleic_type.strip() and a.nucleic_seq.strip():
            target_type = "nucleic"
        elif any_ligand_or_nucleic:
            target_type = "small_molecule"
        else:
            target_type = "protein" # Default for unconditional mode

        success_dir = f"{self.save_dir}/1_af3_rosetta_validation"
        high_iptm_yaml_dir = os.path.join(self.save_dir, "high_iptm_yaml")

        if os.path.exists(high_iptm_yaml_dir):
            print("Starting downstream validation (AlphaFold3 and Rosetta)...")

            # --- AlphaFold Step ---
            af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo = (
                run_alphafold_step(
                    high_iptm_yaml_dir,
                    os.path.expanduser(a.alphafold_dir),
                    a.af3_docker_name,
                    os.path.expanduser(a.af3_database_settings),
                    os.path.expanduser(a.hmmer_path),
                    success_dir,
                    os.path.expanduser(a.work_dir) or os.getcwd(),
                    binder_id=self.binder_chain,
                    gpu_id=a.gpu_id,
                    high_iptm=True,
                    use_msa_for_af3=a.use_msa_for_af3,
                )
            )
            if target_type == "protein":
                # --- Rosetta Step ---
                run_rosetta_step(
                    success_dir,
                    af_pdb_dir,
                    af_pdb_dir_apo,
                    binder_id=self.binder_chain,
                    target_type=target_type,
                )


    def run_pipeline(self):
        """Orchestrates the entire protein design and validation pipeline."""
        # 1. Prepare Base Data (using the new InputDataBuilder)
        base_data, pocket_conditioning = self.data_builder.build()

        # 2. Run Design Cycles
        all_run_metrics = []
        for design_id in range(self.args.num_designs):
            run_id = str(design_id)
            print("\n=======================================================")
            print(f"=== Starting Design Run {run_id}/{self.args.num_designs - 1} ===")
            print("=======================================================")

        
            data_cp = copy.deepcopy(base_data)

            run_metrics = self._run_design_cycle(data_cp, run_id, pocket_conditioning)
            all_run_metrics.append(run_metrics)

        # 3. Save Summary
        self._save_summary_metrics(all_run_metrics)

        # 4. Run Downstream Validation
        if self.args.use_alphafold3_validation:
            self._run_downstream_validation()