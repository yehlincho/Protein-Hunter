import os
import shutil
import json
import logging
import sys
import numpy as np
import pandas as pd
import requests, io
import urllib
from tqdm import tqdm

from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Selection
from utils.metrics import get_CA_and_sequence, np_rmsd

# AlphaFold3 Database Info (to be used by downstream modules)
RNA_DATABASE_INFO = {
    "rnacentral_active_seq_id_90_cov_80_linclust.fasta": "RNAcentral",
    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta": "NT_RNA",
    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta": "Rfam",
}
AF3_SOURCE = "https://storage.googleapis.com/alphafold-databases/v3.0"

# --- Chain Conversion Helpers ---

class OutOfChainsError(Exception):
    pass

def int_to_chain(i: int) -> str:
    """Convert an integer to a one-letter chain ID (A-Z, a-z, 0-9)."""
    if i < 26:
        return chr(ord("A") + i)
    elif i < 52:
        return chr(ord("a") + i - 26)
    elif i < 62:
        return chr(ord("0") + i - 52)
    else:
        raise OutOfChainsError


def rename_chains(structure):
    """
    Renames chains to be one-letter valid PDB chains (A-Z, a-z, 0-9).
    Existing one-letter chains are kept. Others are renamed uniquely.
    Returns a map between new and old chain IDs.
    """
    next_chain = 0
    chainmap = {c.id: c.id for c in structure.get_chains() if len(c.id) == 1}

    # Helper function to find the next available one-letter chain
    def get_next_chain():
        nonlocal next_chain
        while True:
            try:
                c = int_to_chain(next_chain)
                if c not in chainmap:
                    return c
            except OutOfChainsError:
                raise
            next_chain += 1

    for o in structure.get_chains():
        if len(o.id) != 1:
            try:
                c = get_next_chain()
                chainmap[c] = o.id
                o.id = c
            except OutOfChainsError as e:
                logging.error("Too many chains to represent in PDB format")
                raise e
    return chainmap


def sanitize_residue_names(structure):
    """
    Truncates all residue names to 3 characters (PDB format limit).
    Logs a warning if truncation occurs.
    """
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.resname
                if len(resname) > 3:
                    truncated = resname[:3]
                    logging.warning(
                        f"Truncating residue name '{resname}' to '{truncated}'"
                    )
                    residue.resname = truncated

# --- Core Conversion Logic ---

def convert_cif_to_pdb(ciffile, pdbfile):
    """
    Convert a CIF file to PDB format, handling chain renaming and residue name truncation.
    """
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

    # Use a dummy structure ID if path is not an ID
    strucid = os.path.basename(ciffile).split('.')[0] if len(os.path.basename(ciffile)) > 4 else "1xxx"

    # Parse CIF file
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(strucid, ciffile)
    except Exception as e:
        logging.error(f"Failed to parse CIF file {ciffile}: {e}")
        return False

    # Rename chains
    try:
        rename_chains(structure)
    except OutOfChainsError:
        return False

    # Truncate long ligand or residue names
    sanitize_residue_names(structure)

    # Write to PDB
    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(pdbfile)
    except Exception as e:
        logging.error(f"Failed to write PDB file {pdbfile}: {e}")
        return False
        
    return True


def convert_cif_files_to_pdb(
    results_dir: str, save_dir: str, af_dir: bool = False, high_iptm: bool = False, i_ptm_cutoff: float = 0.5
):
    """
    Convert all .cif files in results_dir to .pdb format and save in save_dir.
    Filters by i-pTM score if high_iptm is True.
    """
    confidence_scores = []
    os.makedirs(save_dir, exist_ok=True)
    
    # Find all result files that match the pattern
    cif_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            is_af_file = af_dir and file.endswith("_model.cif")
            is_boltz_file = (not af_dir) and file.endswith(".cif")
            if is_af_file or is_boltz_file:
                cif_files.append((root, file))

    if not cif_files:
        print(f"No CIF files found in {results_dir} matching pattern (af_dir={af_dir}).")
        return

    for root, file in cif_files:
        cif_path = os.path.join(root, file)
        pdb_path = os.path.join(save_dir, file.replace(".cif", ".pdb"))

        # Skip if conversion already done
        if os.path.exists(pdb_path):
             # Try to check if confidence data exists for existing files if in high_iptm mode
            if high_iptm:
                score_name = file.replace(".cif", ".cif")
                if any(score.get('file') == score_name for score in confidence_scores):
                     continue # Already processed and score recorded
            else:
                continue

        iptm = float("-inf")
        plddt = float("-inf")
        should_convert = True

        if high_iptm:
            try:
                # AlphaFold3 output format
                if af_dir:
                    base_name = file.replace("_model.cif", "")
                    confidence_file_summary = os.path.join(root, f"{base_name}_summary_confidences.json")
                    confidence_file = os.path.join(root, f"{base_name}_confidences.json")
                # Boltz output format
                else:
                    confidence_file_summary = os.path.join(root, "confidence_summary.json") # Example name
                    confidence_file = os.path.join(root, "confidence_full.json") # Example name

                
                # Try loading summary confidence data for ipTM
                if os.path.exists(confidence_file_summary):
                    with open(confidence_file_summary) as f:
                        confidence_data = json.load(f)
                        iptm = confidence_data.get("iptm", float("-inf"))
                
                # Try loading full confidence data for pLDDT
                if os.path.exists(confidence_file):
                    with open(confidence_file) as f:
                        confidence_data = json.load(f)
                        plddt = np.mean(confidence_data.get("atom_plddts", [0.0]))

                if iptm < i_ptm_cutoff:
                    should_convert = False
                    print(f"Skipping {file}: i-pTM ({iptm:.2f}) below threshold ({i_ptm_cutoff:.2f}).")

            except Exception as e:
                print(f"WARNING: Could not read confidence data for {file}: {e}")
                should_convert = True # Fail open

        if should_convert:
            print(f"Converting {cif_path} (i-pTM: {iptm:.2f})...")
            if convert_cif_to_pdb(cif_path, pdb_path):
                if high_iptm:
                    confidence_scores.append({"file": file, "iptm": iptm, "plddt": plddt})
            else:
                print(f"❌ Failed to convert {cif_path}.")


    if confidence_scores:
        confidence_scores_path = os.path.join(save_dir, "high_iptm_confidence_scores.csv")
        pd.DataFrame(confidence_scores).to_csv(confidence_scores_path, index=False)
        print(f"✅ Saved confidence scores to {confidence_scores_path}")


def download_with_progress(url, dest_path):
    """Download a file with a progress bar"""
    try:
        with urllib.request.urlopen(url) as response:
            file_size = int(response.info().get("Content-Length", 0))
            desc = f"Downloading {os.path.basename(dest_path)}"
            with tqdm(total=file_size, unit="B", unit_scale=True, desc=desc) as pbar:
                with open(dest_path, "wb") as out_file:
                    while True:
                        buffer = response.read(8192)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        pbar.update(len(buffer))
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False


def calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, binder_chain):
    """
    Calculate RMSD between holo and apo structures and update confidence CSV.
    Uses the unified numpy RMSD function.
    """
    confidence_csv_path = af_pdb_dir + "/high_iptm_confidence_scores.csv"
    if not os.path.exists(confidence_csv_path):
        print(f"Warning: Confidence CSV not found at {confidence_csv_path}")
        return

    df_confidence_csv = pd.read_csv(confidence_csv_path)
    rmsd_updates = {}

    for pdb_name in os.listdir(af_pdb_dir):
        if pdb_name.endswith(".pdb"):
            try:
                pdb_path = os.path.join(af_pdb_dir, pdb_name)
                pdb_path_apo = os.path.join(af_pdb_dir_apo, pdb_name)
                
                # Apo structure is expected to only contain the binder chain (Chain A after AlphaFold3's logic)
                xyz_holo, _ = get_CA_and_sequence(pdb_path, chain_id=binder_chain)
                xyz_apo, _ = get_CA_and_sequence(pdb_path_apo, chain_id="A")

                # RMSD calculation using the unified utility function
                rmsd = np_rmsd(xyz_holo, xyz_apo)
                
                file_key = pdb_name.split(".pdb")[0] + ".cif"
                rmsd_updates[file_key] = rmsd
                print(f"{pdb_name} rmsd: {rmsd:.2f}")

            except ValueError as e:
                print(f"WARNING: Skipping RMSD for {pdb_name}: {e}")
            except Exception as e:
                print(f"ERROR: Failed to calculate RMSD for {pdb_name}: {e}")
                
    # Update DataFrame efficiently
    for index, row in df_confidence_csv.iterrows():
        if row['file'] in rmsd_updates:
            df_confidence_csv.loc[index, 'rmsd'] = rmsd_updates[row['file']]

    df_confidence_csv.to_csv(confidence_csv_path, index=False)
    print("✅ Holo-apo RMSD calculated and updated in CSV.")