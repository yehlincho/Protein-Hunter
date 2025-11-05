import copy
import os
import random
import sys
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
from utils.alphafold_utils import run_alphafold_step
from utils.pyrosetta_utils import run_rosetta_step

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
        """Parses and groups protein/ligand/nucleic acid inputs from command line arguments."""
        a = self.args
        protein_ids = a.protein_ids
        protein_seqs = a.protein_seqs
        protein_msas = a.protein_msas
        cyclics = a.cyclics

        # Handle various separator formats (comma, colon, space) or single string
        if ":" not in protein_ids and "," not in protein_ids and " " not in protein_ids.strip():
            protein_ids_list = [protein_ids.strip()] if protein_ids.strip() else []
            protein_seqs_list = [protein_seqs.strip()] if protein_seqs.strip() else []
            protein_msas_list = [protein_msas.strip()] if protein_msas.strip() else []
            cyclics_list = [cyclics.strip()] if cyclics.strip() else []
        else:
            protein_ids_list = smart_split(protein_ids)
            protein_seqs_list = smart_split(protein_seqs)
            protein_msas_list = (
                smart_split(protein_msas)
                if protein_msas
                else [""] * len(protein_ids_list)
            )
            cyclics_list = (
                smart_split(cyclics) if cyclics else ["False"] * len(protein_ids_list)
            )

        # Pad lists to the max length found
        max_len = max(
            len(protein_ids_list),
            len(protein_seqs_list),
            len(protein_msas_list),
            len(cyclics_list),
        )
        while len(protein_ids_list) < max_len:
            protein_ids_list.append("")
        while len(protein_seqs_list) < max_len:
            protein_seqs_list.append("")
        while len(protein_msas_list) < max_len:
            protein_msas_list.append("")
        while len(cyclics_list) < max_len:
            cyclics_list.append("False")

        return protein_ids_list, protein_seqs_list, protein_msas_list, cyclics_list


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
                        "id": [self.args.binder_chain],
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
        protein_ids_list, protein_seqs_list, protein_msas_list, cyclics_list = (
            self._process_sequence_inputs()
        )
        print("protein_msas_list", protein_msas_list)
        sequences = []

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
                    if protein_msas_list[i] not in ["", None]
                ),
                None
            )
            chosen_msa = chosen_msa if chosen_msa is not None else ""

            if chosen_msa == "": # If MSA path is empty, run MMseqs2
                idx0 = idx_list[0]
                pid0 = (
                    protein_ids_list[idx0]
                    if protein_ids_list[idx0]
                    else f"CHAIN_{idx0}"
                )
                print(f"Processing MSA for target sequence in {pid0}...")
                msa_value = process_msa(
                    pid0, seq, Path(self.protein_hunter_save_dir)
                )
                seq_to_final_msa[seq] = str(msa_value)
            elif chosen_msa == "empty":
                seq_to_final_msa[seq] = "empty"
            else:
                seq_to_final_msa[seq] = chosen_msa

        # Step 2: Build sequences list for target proteins
        for pid, seq, cyc in zip(protein_ids_list, protein_seqs_list, cyclics_list):
            if not pid or not seq:
                continue
            final_msa = seq_to_final_msa.get(seq, "empty")
            cyc_val = cyc.lower() in ["true", "1", "yes"]
            sequences.append(
                {
                    "protein": {
                        "id": [pid],
                        "sequence": seq,
                        "msa": final_msa,
                        "cyclic": cyc_val,
                    }
                }
            )

        # Step 3: Add binder chain
        sequences.append(
            {
                "protein": {
                    "id": [a.binder_chain],
                    "sequence": "X",
                    "msa": "empty",
                    "cyclic": False,
                }
            }
        )

        # Step 4: Add ligands/nucleic acids
        if a.ligand_smiles:
            sequences.append(
                {"ligand": {"id": [a.ligand_id], "smiles": a.ligand_smiles}}
            )
        elif a.ligand_ccd:
            sequences.append({"ligand": {"id": [a.ligand_id], "ccd": a.ligand_ccd}})
        if a.nucleic_seq:
            sequences.append(
                {a.nucleic_type: {"id": [a.nucleic_id], "sequence": a.nucleic_seq}}
            )

        # Step 5: Add templates
        templates = self._build_templates()

        data = {"sequences": sequences}
        if templates:
            data["templates"] = templates

        # Step 6: Add constraints (pocket conditioning)
        pocket_conditioning = a.add_constraints
        if a.add_constraints:
            residues = a.contact_residues.split(",")
            contacts = [
                [a.constraint_target_chain, int(res)]
                for res in residues
                if res.strip() != ""
            ]
            constraints = {
                "pocket": {"binder": a.binder_chain, "contacts": contacts}
            }
            data["constraints"] = [constraints]

        return data, pocket_conditioning

    def _build_templates(self):
        """Constructs the list of template dictionaries."""
        a = self.args
        templates = []
        if a.template_path:
            template_path_list = smart_split(a.template_path)
            template_chain_id_list = (
                smart_split(a.template_chain_id) if a.template_chain_id else []
            )
            template_cif_chain_id_list = (
                smart_split(a.template_cif_chain_id)
                if a.template_cif_chain_id
                else []
            )
            template_files = [get_cif(tp) for tp in template_path_list]
            
            # Ensure all lists are of the same length, padding with empty strings if necessary
            max_len = len(template_files)
            if len(template_chain_id_list) < max_len:
                template_chain_id_list += [""] * (max_len - len(template_chain_id_list))
            if len(template_cif_chain_id_list) < max_len:
                template_cif_chain_id_list += [""] * (max_len - len(template_cif_chain_id_list))

            for i, template_file in enumerate(template_files):
                t_block = (
                    {"cif": template_file}
                    if template_file.endswith(".cif")
                    else {"pdb": template_file}
                )
                
                # Only add chain IDs if they exist for this template entry
                if template_chain_id_list and template_chain_id_list[i]:
                    t_block["chain_id"] = template_chain_id_list[i]
                    if template_cif_chain_id_list and template_cif_chain_id_list[i]:
                        t_block["cif_chain_id"] = template_cif_chain_id_list[i]
                
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

        print("✅ ProteinHunter_Boltz initialized.")

    def _load_boltz_model(self):
        """Loads and configures the Boltz model."""
        predict_args = {
            "recycling_steps": self.args.recycling_steps,
            "sampling_steps": self.args.diffuse_steps,
            "diffusion_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
            "max_parallel_samples": 1,
        }
        return get_boltz_model(
            checkpoint=self.args.boltz_model_path,
            predict_args=predict_args,
            device=self.device,
            model_version=self.args.boltz_model_version,
            no_potentials=self.args.no_potentials,
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
        run_metrics = {"run_id": run_id}

        binder_length = random.randint(
            a.min_design_protein_length, a.max_design_protein_length
        )

        # Helper function to update sequence in the data dictionary
        def update_binder_sequence(new_seq):
            for seq_entry in data_cp["sequences"]:
                if (
                    "protein" in seq_entry
                    and a.binder_chain in seq_entry["protein"]["id"]
                ):
                    seq_entry["protein"]["sequence"] = new_seq
                    return
            # Should not happen if data_cp is built correctly
            raise ValueError("Binder chain not found in data dictionary.")

        # Set initial binder sequence
        initial_seq = sample_seq(
            binder_length, exclude_P=a.exclude_P, frac_X=a.frac_X
        )
        update_binder_sequence(initial_seq)
        print(f"Binder initial sequence length: {binder_length}")

        # --- Cycle 0 structure prediction, with contact filtering check ---
        contact_filter_attempt = 0
        target_chain_for_contacts = a.constraint_target_chain
        pdb_filename = ""
        structure = None
        output = None

        while True:
            output, structure = run_prediction(
                data_cp,
                a.binder_chain,
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
                    binds = binder_binds_contacts(
                        pdb_filename,
                        a.binder_chain,
                        target_chain_for_contacts,
                        a.contact_residues,
                        cutoff=a.contact_cutoff,
                    )
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
            new_seq = sample_seq(binder_length, exclude_P=a.exclude_P, frac_X=a.frac_X)
            update_binder_sequence(new_seq)
            clean_memory()

        # Capture Cycle 0 metrics
        binder_chain_idx = CHAIN_TO_NUMBER[a.binder_chain]
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

        run_metrics["cycle_0_iptm"] = cycle_0_iptm
        run_metrics["cycle_0_plddt"] = float(
            output.get("complex_plddt", torch.tensor([0.0])).detach().cpu().numpy()[0]
        )
        run_metrics["cycle_0_iplddt"] = float(
            output.get("complex_iplddt", torch.tensor([0.0])).detach().cpu().numpy()[0]
        )
        run_metrics["cycle_0_alanine"] = 0

        # --- Optimization Cycles ---
        for cycle in range(a.num_cycles):
            print(f"\n--- Run {run_id}, Cycle {cycle + 1} ---")

            # Calculate temperature and bias for the cycle
            cycle_norm = (cycle / (a.num_cycles - 1)) if a.num_cycles > 1 else 0.0
            alpha = a.alanine_bias_start - cycle_norm * (
                a.alanine_bias_start - a.alanine_bias_end
            )
            temperature = a.temperature_start - cycle_norm * (
                a.temperature_start - a.temperature_end
            )

            # 1. Design Sequence
            model_type = (
                "ligand_mpnn"
                if (a.ligand_smiles or a.ligand_ccd or a.nucleic_seq)
                else "soluble_mpnn"
            )
            design_kwargs = {
                "pdb_file": pdb_filename,
                "temperature": temperature,
                "chains_to_design": a.binder_chain,
                "omit_AA": f"{a.omit_AA},P" if cycle == 0 else a.omit_AA,
                "bias_AA": f"A:{alpha}" if a.alanine_bias else "",
            }

            seq_str, logits = design_sequence(
                self.designer, model_type, **design_kwargs
            )
            # The output seq_str is a dictionary-like string, we extract the binder chain sequence
            seq = seq_str.split(":")[CHAIN_TO_NUMBER[a.binder_chain]] 

            # Update data_cp with new sequence
            alanine_count = seq.count("A")
            alanine_percentage = (
                alanine_count / binder_length if binder_length != 0 else 0.0
            )
            update_binder_sequence(seq) # Use the helper function

            # 2. Structure Prediction
            output, structure = run_prediction(
                data_cp,
                a.binder_chain,
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
            current_chain_idx = CHAIN_TO_NUMBER[a.binder_chain]
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

            pdb_filename = (
                f"{run_save_dir}/{a.name}_run_{run_id}_predicted_cycle_{cycle + 1}.pdb"
            )
            plddts = output["plddt"].detach().cpu().numpy()[0]
            save_pdb(structure, output["coords"], plddts, pdb_filename)
            clean_memory()

            print(
                f"ipTM: {current_iptm:.2f} pLDDT: {curr_plddt:.2f} iPLDDT: {curr_iplddt:.2f} Alanine count: {alanine_count}"
            )

            # 4. Save YAML for High ipTM
            save_yaml_this_design = (alanine_percentage <= 0.20) and (
                current_iptm > a.high_iptm_threshold
            )

            if save_yaml_this_design and a.contact_residues.strip():
                this_cycle_pdb_filename = f"{run_save_dir}/{a.name}_run_{run_id}_predicted_cycle_{cycle + 1}.pdb"
                try:
                    contact_binds = binder_binds_contacts(
                        this_cycle_pdb_filename,
                        a.binder_chain,
                        target_chain_for_contacts,
                        a.contact_residues,
                        cutoff=a.contact_cutoff,
                    )
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
                high_iptm_yaml_dir = os.path.join(self.save_dir, "high_iptm_yaml")
                os.makedirs(high_iptm_yaml_dir, exist_ok=True)
                yaml_filename = os.path.join(
                    high_iptm_yaml_dir,
                    f"{a.name}_run_{run_id}_cycle_{cycle + 1}_output.yaml",
                )
                with open(yaml_filename, "w") as f:
                    yaml.dump(data_cp, f, default_flow_style=False)
                print(f"✅ Saved run {run_id} cycle {cycle + 1} YAML.")

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
        else:
            run_metrics["best_iptm"] = float("nan")
            run_metrics["best_cycle"] = None
            run_metrics["best_plddt"] = float("nan")
            run_metrics["best_seq"] = None

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
                ]
            )
        # Best metric columns
        columns.extend(["best_iptm", "best_cycle", "best_plddt", "best_seq"])

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
        """Executes AlphaFold and Rosetta validation steps."""
        a = self.args
        
        # Determine target type for Rosetta validation
        any_ligand_or_nucleic = a.ligand_smiles or a.ligand_ccd or a.nucleic_seq
        if a.nucleic_type.strip() and a.nucleic_seq.strip():
            target_type = "nucleic"
        elif any_ligand_or_nucleic:
            target_type = "small_molecule"
        elif a.protein_ids.strip() and a.protein_seqs.strip():
            target_type = "protein"
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
                    binder_id=a.binder_chain,
                    gpu_id=a.gpu_id,
                    high_iptm=True,
                    use_msa_for_af3=a.use_msa_for_af3,
                )
            )

            # --- Rosetta Step ---
            run_rosetta_step(
                success_dir,
                af_pdb_dir,
                af_pdb_dir_apo,
                binder_id=a.binder_chain,
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

            # Deep copy to allow modification of 'sequence' per run
            data_cp = copy.deepcopy(base_data)

            run_metrics = self._run_design_cycle(data_cp, run_id, pocket_conditioning)
            all_run_metrics.append(run_metrics)

        # 3. Save Summary
        self._save_summary_metrics(all_run_metrics)

        # 4. Run Downstream Validation
        self._run_downstream_validation()