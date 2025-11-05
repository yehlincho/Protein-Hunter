import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyrosetta as pr
from Bio.PDB import PDBParser, Selection
from pyrosetta.rosetta.core.io import pose_from_pose
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from scipy.spatial import cKDTree

from boltz_ph.constants import RESTYPE_3TO1, HYDROPHOBIC_AA
from utils.metrics import get_CA_and_sequence, np_rmsd, radius_of_gyration

# Initialize PyRosetta with all needed options
dalphaball_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "DAlphaBall.gcc"
)
pr.init(
    f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {dalphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1"
)

def get_sequence(cif_file, chain_id="B"):
    """
    Extracts 1-letter protein sequence from a CIF file.
    Note: CIF parsing now relies on Bio.PDB.MMCIFParser which is used inside get_CA_and_sequence.
    """
    try:
        _, sequence = get_CA_and_sequence(cif_file, chain_id)
        return sequence
    except Exception as e:
        print(f"Error getting sequence from {cif_file} chain {chain_id}: {e}")
        return ""


def clean_pdb(pdb_file):
    """
    Removes non-standard lines from a PDB file to ensure PyRosetta compatibility.
    """
    with open(pdb_file) as f_in:
        relevant_lines = [
            line
            for line in f_in
            if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "END"))
        ]

    with open(pdb_file, "w") as f_out:
        f_out.writelines(relevant_lines)


def pr_relax(pdb_file, relaxed_pdb_path):
    """
    Runs PyRosetta FastRelax protocol on a PDB file.
    The implementation is robust to handle existence check, PDB loading, and alignment.
    """
    if os.path.exists(relaxed_pdb_path):
        return

    # Generate pose
    try:
        pose = pr.pose_from_pdb(pdb_file)
    except Exception as e:
        print(f"Error loading PDB {pdb_file} for relaxation: {e}")
        return
        
    start_pose = pose.clone()

    ### Generate movemaps
    mmf = MoveMap()
    mmf.set_chi(True)
    mmf.set_bb(True)
    mmf.set_jump(False)

    # Run FastRelax
    fastrelax = FastRelax()
    scorefxn = pr.get_fa_scorefxn()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mmf)
    fastrelax.max_iter(200)
    fastrelax.min_type("lbfgs_armijo_nonmonotone")
    fastrelax.constrain_relax_to_start_coords(True)
    fastrelax.apply(pose)

    # Align relaxed structure to original trajectory
    # Uses chain 0 (the whole complex) for alignment
    align = AlignChainMover()
    align.source_chain(0)
    align.target_chain(0)
    align.pose(start_pose)
    align.apply(pose)
    
    # Copy B factors from start_pose to pose for visualization consistency
    for resid in range(1, pose.total_residue() + 1):
        if pose.residue(resid).is_protein():
            # Get the B factor of the first heavy atom in the residue
            bfactor = start_pose.pdb_info().bfactor(resid, 1)
            for atom_id in range(1, pose.residue(resid).natoms() + 1):
                pose.pdb_info().bfactor(resid, atom_id, bfactor)


    # output relaxed and aligned PDB
    pose.dump_pdb(relaxed_pdb_path)
    clean_pdb(relaxed_pdb_path)


def hotspot_residues(trajectory_pdb, binder_chain="B", target_chain="A", atom_distance_cutoff=4.0):
    """
    Identifies interface residues on the binder chain by checking for any binder atom
    within cutoff of any target atom.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", trajectory_pdb)
    except Exception as e:
        print(f"Error parsing {trajectory_pdb}: {e}")
        return {}
        
    model = structure[0]
    
    if binder_chain not in model or target_chain not in model:
        print(f"Warning: One or both chains ({binder_chain}, {target_chain}) not found for hotspot analysis.")
        return {}

    binder_atoms = Selection.unfold_entities(model[binder_chain], "A")
    target_atoms = Selection.unfold_entities(model[target_chain], "A")

    if not binder_atoms or not target_atoms:
        return {}

    binder_coords = np.array([atom.coord for atom in binder_atoms])
    target_coords = np.array([atom.coord for atom in target_atoms])

    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Query the tree for pairs of atoms within the distance cutoff
    # 'pairs' is a list where pairs[i] is a list of indices in target_coords that are near binder_coords[i]
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Prepare to collect interacting residues (pdb res number: 1-letter AA)
    interacting_residues = {}

    for binder_idx, close_indices in enumerate(pairs):
        if close_indices: # If there is at least one interaction
            binder_residue = binder_atoms[binder_idx].get_parent()
            pdb_res_num = binder_residue.id[1]
            resname = binder_residue.get_resname().strip().upper()
            
            # Use the global mapping
            aa_single_letter = RESTYPE_3TO1.get(resname, "X")

            # Store the residue number and its 1-letter code
            interacting_residues[pdb_res_num] = aa_single_letter

    return interacting_residues


def score_interface(pdb_file, binder_chain="B", target_chain="A"):
    """
    Calculates various PyRosetta interface and complex metrics.
    """
    # Load pose
    try:
        pose = pr.pose_from_pdb(pdb_file)
    except Exception as e:
        print(f"Error loading PDB {pdb_file} for interface scoring: {e}")
        return {}, {}, ""

    # Define interface string for InterfaceAnalyzerMover
    interface_string = f"{binder_chain}_{target_chain}"

    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    iam.set_interface(interface_string)
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    # Enable all calculations
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # 1. Hotspot Analysis
    interface_residues_set = hotspot_residues(pdb_file, binder_chain, target_chain)
    interface_residues_pdb_ids = []
    interface_AA = dict.fromkeys("ACDEFGHIKLMNPQRSTVWY", 0)

    for pdb_res_num, aa_type in interface_residues_set.items():
        interface_AA[aa_type] = interface_AA.get(aa_type, 0) + 1
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    interface_nres = len(interface_residues_pdb_ids)
    interface_residues_pdb_ids_str = ",".join(interface_residues_pdb_ids)

    # 2. Interface Hydrophobicity
    hydrophobic_count = sum(interface_AA[aa] for aa in HYDROPHOBIC_AA)
    interface_hydrophobicity = (
        (hydrophobic_count / interface_nres) * 100 if interface_nres != 0 else 0
    )

    # 3. Retrieve IAM scores
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value
    interface_interface_hbonds = interfacescore.interface_hbonds
    interface_dG = iam.get_interface_dG()
    interface_dSASA = iam.get_interface_delta_sasa()
    interface_packstat = iam.get_interface_packstat()
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100
    
    # 4. Buried Unsaturated Hbonds (BUNS)
    buns_filter = XmlObjects.static_get_filter(
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
    )
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    # 5. Percentage scores
    interface_hbond_percentage = (
        (interface_interface_hbonds / interface_nres) * 100 if interface_nres != 0 else None
    )
    interface_bunsch_percentage = (
        (interface_delta_unsat_hbonds / interface_nres) * 100 if interface_nres != 0 else None
    )

    # 6. Binder Energy and SASA
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    interface_binder_fraction = (
        (interface_dSASA / binder_sasa) * 100 if binder_sasa > 0 else 0
    )

    # 7. Surface Hydrophobicity (on the binder chain)
    try:
        binder_pose = {
            pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
            for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
        }[binder_chain]

        layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_res = layer_sel.apply(binder_pose)

        exp_apol_count = 0
        total_count = 0

        # count apolar and aromatic residues at the surface
        for i in range(1, len(surface_res) + 1):
            if surface_res[i]:
                res = binder_pose.residue(i)
                # Count apolar and aromatic residues as hydrophobic
                if res.is_apolar() or res.name in ["PHE", "TRP", "TYR"]:
                    exp_apol_count += 1
                total_count += 1

        surface_hydrophobicity = exp_apol_count / total_count if total_count > 0 else 0
    except Exception as e:
        print(f"Warning: Failed surface hydrophobicity calculation: {e}")
        surface_hydrophobicity = 0.0

    # 8. Compile and Round Results
    interface_scores = {
        "binder_score": binder_score,
        "surface_hydrophobicity": surface_hydrophobicity,
        "interface_sc": interface_sc,
        "interface_packstat": interface_packstat,
        "interface_dG": interface_dG,
        "interface_dSASA": interface_dSASA,
        "interface_dG_SASA_ratio": interface_dG_SASA_ratio,
        "interface_fraction": interface_binder_fraction,
        "interface_hydrophobicity": interface_hydrophobicity,
        "interface_nres": interface_nres,
        "interface_interface_hbonds": interface_interface_hbonds,
        "interface_hbond_percentage": interface_hbond_percentage,
        "interface_delta_unsat_hbonds": interface_delta_unsat_hbonds,
        "interface_delta_unsat_hbonds_percentage": interface_bunsch_percentage,
    }

    interface_scores = {
        k: round(v, 2) if isinstance(v, float) else v
        for k, v in interface_scores.items()
    }

    return interface_scores, interface_AA, interface_residues_pdb_ids_str


# Removed align_pdbs (was unused or for debugging external files)

def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    Calculates RMSD between two chains without performing a Kabsch alignment.
    (This function is less commonly used than np_rmsd but kept for completeness).
    """
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    reference_chain_selector = ChainSelector(reference_chain_id)
    align_chain_selector = ChainSelector(align_chain_id)

    reference_chain_subset = reference_chain_selector.apply(reference_pose)
    align_chain_subset = align_chain_selector.apply(align_pose)

    reference_residue_indices = get_residues_from_subset(reference_chain_subset)
    align_residue_indices = get_residues_from_subset(align_chain_subset)
    
    if len(reference_residue_indices) != len(align_residue_indices):
        print("Warning: Chains have different lengths for RMSD calculation. Aligning will fail or be meaningless.")
        return None

    reference_chain_pose = pr.Pose()
    align_chain_pose = pr.Pose()

    pose_from_pose(reference_chain_pose, reference_pose, reference_residue_indices)
    pose_from_pose(align_chain_pose, align_pose, align_residue_indices)

    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(reference_chain_pose)
    rmsd = rmsd_metric.calculate(align_chain_pose)

    return round(rmsd, 2)


def get_binder_chain(pdb_file):
    """Simple utility to get the last chain ID in a PDB file."""
    parser = PDBParser(QUIET=True)
    binder_structure = parser.get_structure("protein", pdb_file)
    chain_ids = [chain.id for model in binder_structure for chain in model]
    return chain_ids[-1]


def measure_rosetta_energy(
    pdbs_path,
    pdbs_apo_path,
    save_dir,
    binder_holo_chain="B",
    binder_apo_chain="A",
    target="peptide",
):
    """
    Measures Rosetta energy metrics for a set of designs and filters them based on thresholds.
    """
    # Create relaxed output directory
    relaxed_dir = pdbs_path + "_relaxed"
    os.makedirs(relaxed_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(pdbs_path, "rosetta_energy.csv")

    df = pd.DataFrame()
    processed_files = set()
    
    # Load existing CSV if it exists
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            processed_files = set(existing_df["Model"].values)
            df = existing_df.copy() # Start with existing data
        except Exception as e:
            print(f"Warning: Could not load existing CSV at {output_path}: {e}")

    new_rows = []
    
    for pdb_file in os.listdir(pdbs_path):
        if pdb_file.endswith(".pdb") and not pdb_file.startswith("relax_"):
            if pdb_file in processed_files:
                continue

            try:
                design_pathway = os.path.join(pdbs_path, pdb_file)
                relax_pathway = os.path.join(relaxed_dir, f"relax_{pdb_file}")
                
                # Check for chain ID from holo PDB (might be A or B depending on AF3 processing)
                binder_chain = get_binder_chain(design_pathway)
                
                pr_relax(design_pathway, relax_pathway)
                
                (
                    trajectory_interface_scores,
                    trajectory_interface_AA,
                    trajectory_interface_residues,
                ) = score_interface(relax_pathway, binder_chain, target_chain="A")
                
                print(f"Rosetta scores for {pdb_file}: {trajectory_interface_scores}")

                row_data = {"PDB": relaxed_dir, "Model": f"relax_{pdb_file}"}
                row_data.update(trajectory_interface_scores)
                new_rows.append(row_data)
                processed_files.add(pdb_file)
                
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")

    # Append new rows and save
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(output_path, index=False)
        print(f"✅ New Rosetta results appended to {output_path}")
    else:
        print("No new PDB files to process for Rosetta scoring.")

    # --- Filtering Logic ---
    if df.empty:
        print("No data available for filtering.")
        return

    # Define filtering mask based on target type
    if target == "peptide":
        mask = (
            (df["binder_score"] < 0)
            & (df["surface_hydrophobicity"] < 0.35)
            & (df["interface_sc"] > 0.55)
            & (df["interface_packstat"] > 0)
            & (df["interface_dG"] < 0)
            & (df["interface_dSASA"] > 1)
            & (df["interface_dG_SASA_ratio"] < 0)
            & (df["interface_nres"] > 4)
            & (df["interface_interface_hbonds"] > 3)
            & (df["interface_hbond_percentage"] > 0)
            & (df["interface_delta_unsat_hbonds"] < 2)
        )
    else: # protein, small_molecule, nucleic (uses general protein interface criteria)
        mask = (
            (df["binder_score"] < 0)
            & (df["surface_hydrophobicity"] < 0.35)
            & (df["interface_sc"] > 0.55)
            & (df["interface_packstat"] > 0)
            & (df["interface_dG"] < 0)
            & (df["interface_dSASA"] > 1)
            & (df["interface_dG_SASA_ratio"] < 0)
            & (df["interface_nres"] > 7)
            & (df["interface_interface_hbonds"] > 3)
            & (df["interface_hbond_percentage"] > 0)
            & (df["interface_delta_unsat_hbonds"] < 4)
        )
        
    # Apply mask and filter
    filtered_df = df[mask].copy()
    failed_df = df[~mask].copy()

    print(f"Number of designs passing all Rosetta filters: {len(filtered_df)}")
    print(f"Number of designs failing Rosetta filters: {len(failed_df)}")

    all_filtered_rows = []
    
    # --- Final AlphaFold Metric Cross-Validation (Only for filtered designs) ---
    if len(filtered_df) > 0:
        confidence_csv_path = os.path.join(pdbs_path, "high_iptm_confidence_scores.csv")
        if os.path.exists(confidence_csv_path):
            af_confidence_df = pd.read_csv(confidence_csv_path)
            af_confidence_df['file_key'] = af_confidence_df['file'].str.replace('.cif', '.pdb')
        else:
            print("Warning: AlphaFold confidence CSV not found. Skipping final cross-validation checks.")
            af_confidence_df = None

        for i in range(len(filtered_df)):
            try:
                row = filtered_df.iloc[i].copy()
                relaxed_pdb_name = row["Model"]
                relaxed_pdb_path = os.path.join(row["PDB"], relaxed_pdb_name)
                
                # Extract original AF3 model name
                model_base = relaxed_pdb_name.replace("relax_", "").replace("_model.pdb", "")
                
                # Fetch AF3 metrics (iptm, plddt, i_pae, rmsd) from CSV if available
                af_row = af_confidence_df[af_confidence_df['file_key'] == model_base + ".pdb"].iloc[0] if af_confidence_df is not None and (model_base + ".pdb") in af_confidence_df['file_key'].values else {}
                
                # 1. AF3 Metrics
                row["iptm"] = af_row.get("iptm", float('nan'))
                row["rmsd"] = af_row.get("rmsd", float('nan')) # apo_holo_rmsd
                
                # 2. Sequence and Rg (Requires fetching original AF3 PDB)
                af_holo_pdb = os.path.join(pdbs_path, model_base + ".pdb")
                af_apo_pdb = os.path.join(pdbs_apo_path, model_base + ".pdb")

                xyz_holo, aa_seq = get_CA_and_sequence(af_holo_pdb, chain_id=binder_holo_chain)
                row["aa_seq"] = aa_seq
                rg, _ = radius_of_gyration(relaxed_pdb_path, chain_id=binder_holo_chain)
                row["rg"] = rg
                
                # 3. pLDDT and i-PAE (Requires reading JSONs) - simplified read (not implemented here, relies on prior CSV)
                # For this refactor, we rely on the RMSD calculation step to have populated plddt and i_pae into the AF confidence CSV.
                row["plddt"] = af_row.get("plddt", float('nan'))
                
                # We need i_pae, which is usually a manual calculation post-AF3 prediction. 
                # Assuming the logic in the original file would calculate/fetch i_pae and put it in the row. 
                # Since the original file fetches it from JSON *inside* this loop, we re-create the fetch logic for robustness.
                
                base_af3_path = os.path.join(pdbs_path.replace("03_af_pdb_success", "02_design_final_af3"), model_base)
                confidenece_json_2 = f"{base_af3_path}/{model_base}_confidences.json"
                
                if os.path.exists(confidenece_json_2):
                     with open(confidenece_json_2) as f:
                        confidence_data = json.load(f)
                        pae_matrix = np.array(confidence_data["pae"])
                        protein_len = len(aa_seq)
                        interface_pae1 = np.mean(pae_matrix[:protein_len, protein_len:])
                        interface_pae2 = np.mean(pae_matrix[protein_len:, :protein_len])
                        i_pae = (interface_pae1 + interface_pae2) / 2
                        row["i_pae"] = i_pae
                else:
                    row["i_pae"] = float('nan')


                # 4. Final Threshold Filter
                print(f"Final check for {model_base}: iptm={row['iptm']:.2f}, plddt={row['plddt']:.1f}, rg={row['rg']:.1f}, i_pae={row['i_pae']:.1f}, rmsd={row['rmsd']:.1f}")
                
                final_pass = (
                    row.get("iptm", 0.0) > 0.5
                    and row.get("plddt", 0.0) > 80
                    and row.get("rg", 20.0) < 17
                    and row.get("i_pae", 20.0) < 15
                    and row.get("rmsd", 5.0) < 3.5
                )

                if final_pass:
                    shutil.copy(relaxed_pdb_path, os.path.join(save_dir, relaxed_pdb_name))
                    all_filtered_rows.append(row)
                else:
                    failed_df = pd.concat([failed_df, pd.DataFrame([row])], ignore_index=True)
                    
            except Exception as e:
                print(f"Error during final cross-validation for {relaxed_pdb_name}: {e}")
                failed_df = pd.concat([failed_df, pd.DataFrame([filtered_df.iloc[i]])], ignore_index=True) # Add to failed if exception occurs
                continue

    # --- Save Final Results ---
    success_csv = os.path.join(save_dir, "success_designs.csv")
    failed_csv = os.path.join(save_dir, "failed_designs.csv")
    zip_path = save_dir + ".zip"

    save_df = pd.DataFrame(all_filtered_rows)
    save_df.to_csv(success_csv, index=False)

    print(f"🎉 Number of final successful designs: {len(save_df)}")

    failed_df.to_csv(failed_csv, index=False)

    # Zip the successful PDBs and CSVs
    if save_df.shape[0] > 0:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(save_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, save_dir)
                    zipf.write(file_path, arcname)
            print(f"📦 Successfully zipped results to {zip_path}")


def run_rosetta_step(
    ligandmpnn_dir, af_pdb_dir, af_pdb_dir_apo, binder_id="A", target_type="protein"
):
    """Run Rosetta energy calculation (protein targets only)"""

    if target_type not in ["protein", "peptide"]:
        print("Skipping Rosetta step (not a protein/peptide target)")
        return

    print("Starting Rosetta energy calculation...")
    af_pdb_rosetta_success_dir = f"{ligandmpnn_dir}/af_pdb_rosetta_success"

    measure_rosetta_energy(
        af_pdb_dir,
        af_pdb_dir_apo,
        af_pdb_rosetta_success_dir,
        binder_holo_chain=binder_id,
        binder_apo_chain="A",
        target=target_type,
    )

    print("Rosetta energy calculation completed!")