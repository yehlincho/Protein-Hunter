import gc
import json
import os
import random
import subprocess
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
import torch
from prody import parsePDB
import gemmi

from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_molecules, load_canonicals
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import (
    MSA,
    Coords,
    Ensemble,
    Input,
    Structure,
    StructureV2,
)
from boltz.data.write.pdb import to_pdb
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBIO import PDBIO
import logging


# Existing filter
warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
    category=UserWarning,
)

from utils.alphafold_utils import *

alphabet = list[str]("XXARNDCQEGHILKMFPSTWYV-")

chain_to_number = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
}


def get_cif(cif_code=""):
    if cif_code is None or cif_code == "":
        print("Error: No cif code specified and uploads not supported in CLI mode.")
        sys.exit(1)
    elif os.path.isfile(cif_code):
        return cif_code
    elif len(cif_code) == 4:
        os.system(f"wget -qnc https://files.rcsb.org/download/{cif_code}.cif")
        return f"{cif_code}.cif"
    else:
        os.system(
            f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{cif_code}-F1-model_v3.cif"
        )
        return f"AF-{cif_code}-F1-model_v3.cif"


def get_CA(x):
    xyz = []
    with open(x) as file:
        for line in file:
            line = line.rstrip()
            if line[:4] == "ATOM":
                atom = line[12 : 12 + 4].strip()
                if atom == "CA":
                    resi = line[17 : 17 + 3]
                    resn = int(line[22 : 22 + 5]) - 1
                    x = float(line[30 : 30 + 8])
                    y = float(line[38 : 38 + 8])
                    z = float(line[46 : 46 + 8])
                    xyz.append([x, y, z])
    return np.array(xyz)



def get_CA_and_sequence(structure_file, chain_id="A"):
    # Determine file type and use appropriate parser
    if structure_file.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    elif structure_file.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("File must be either .cif or .pdb format")

    structure = parser.get_structure("structure", structure_file)
    xyz = []
    sequence = []
    aa_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLU": "E",
        "GLN": "Q",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }

    model = structure[0]  # Get first model (default for most structures)

    if chain_id in model:
        chain = model[chain_id]
        for residue in chain:
            if "CA" in residue:
                xyz.append(residue["CA"].coord)
                sequence.append(aa_map.get(residue.resname, "X"))
    else:
        raise ValueError(f"Chain {chain_id} not found in {structure_file}")

    return xyz, sequence




def np_kabsch(a, b, return_v=False):
    """Get alignment matrix for two sets of coordinates using numpy

    Args:
        a: First set of coordinates
        b: Second set of coordinates
        return_v: If True, return U matrix from SVD. If False, return rotation matrix

    Returns
    -------
        Rotation matrix (or U matrix if return_v=True) to align coordinates
    """
    # Calculate covariance matrix
    ab = np.swapaxes(a, -1, -2) @ b

    # Singular value decomposition
    u, s, vh = np.linalg.svd(ab, full_matrices=False)

    # Handle reflection case
    flip = np.linalg.det(u @ vh) < 0
    if flip:
        u[..., -1] = -u[..., -1]

    return u if return_v else (u @ vh)


def np_rmsd(true, pred):
    """Compute RMSD of coordinates after alignment using numpy

    Args:
        true: Reference coordinates
        pred: Predicted coordinates to align

    Returns
    -------
        Root mean square deviation after optimal alignment
    """
    # Center coordinates
    p = true - np.mean(true, axis=-2, keepdims=True)
    q = pred - np.mean(pred, axis=-2, keepdims=True)

    # Get optimal rotation matrix and apply it
    p = p @ np_kabsch(p, q)

    # Calculate RMSD
    return np.sqrt(np.mean(np.sum(np.square(p - q), axis=-1)) + 1e-8)


class OutOfChainsError(Exception):
    pass


def rename_chains(structure):
    """
    Renames chains to be one-letter valid PDB chains.

    Existing one-letter chains are kept. Others are renamed uniquely.
    Raises OutOfChainsError if more than 62 chains are present.
    Returns a map between new and old chain IDs.
    """
    next_chain = 0
    chainmap = {c.id: c.id for c in structure.get_chains() if len(c.id) == 1}

    for o in structure.get_chains():
        if len(o.id) != 1:
            if o.id[0] not in chainmap:
                chainmap[o.id[0]] = o.id
                o.id = o.id[0]
            else:
                c = int_to_chain(next_chain)
                while c in chainmap:
                    next_chain += 1
                    if next_chain >= 62:
                        raise OutOfChainsError
                    c = int_to_chain(next_chain)
                chainmap[c] = o.id
                o.id = c
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


def convert_cif_to_pdb(ciffile, pdbfile):
    """
    Convert a CIF file to PDB format, handling chain renaming and residue name truncation.

    Args:
        ciffile (str): Path to input CIF file
        pdbfile (str): Path to output PDB file

    Returns
    -------
        bool: True if conversion succeeds, False otherwise
    """
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

    strucid = ciffile[:4] if len(ciffile) > 4 else "1xxx"

    # Parse CIF file
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(strucid, ciffile)

    # Rename chains
    try:
        rename_chains(structure)
    except OutOfChainsError:
        logging.error("Too many chains to represent in PDB format")
        return False

    # Truncate long ligand or residue names
    sanitize_residue_names(structure)

    # Write to PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdbfile)
    return True


def convert_cif_files_to_pdb(
    results_dir, save_dir, af_dir=False, high_iptm=False, i_ptm_cutoff=0.5
):
    """
    Convert all .cif files in results_dir to .pdb format and save in save_dir
    Args:
        results_dir (str): Directory containing .cif files
        save_dir (str): Directory to save converted .pdb files
        af_dir (bool): If True, look for _model.cif_model.cif files instead of .cif
    """
    confidence_scores = []
    os.makedirs(save_dir, exist_ok=True)
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if af_dir:
                if file.endswith("_model.cif"):
                    cif_path = os.path.join(root, file)
                    pdb_path = os.path.join(save_dir, file.replace(".cif", ".pdb"))
                    print(f"Converting {cif_path}")

                    if high_iptm:
                        confidence_file_summary = [
                            os.path.join(root, f)
                            for f in os.listdir(root)
                            if f.endswith("summary_confidences.json")
                        ][0]
                        confidence_file = [
                            os.path.join(root, f)
                            for f in os.listdir(root)
                            if f.endswith("_confidences.json")
                            and not f.endswith("summary_confidences.json")
                        ][0]
                        with open(confidence_file_summary) as f:
                            confidence_data = json.load(f)
                            iptm = confidence_data["iptm"]

                        with open(confidence_file) as f:
                            confidence_data = json.load(f)
                            plddt = np.mean(confidence_data["atom_plddts"])

                        if iptm > i_ptm_cutoff:
                            print(f"Converting {cif_path}")
                            print(f"iptm score: {iptm}")
                            print(f"pdb_path: {pdb_path}")
                            convert_cif_to_pdb(cif_path, pdb_path)
                            confidence_scores.append(
                                {"file": file, "iptm": iptm, "plddt": plddt}
                            )

                    else:
                        print(f"Converting {cif_path}")
                        convert_cif_to_pdb(cif_path, pdb_path)
            elif file.endswith(".cif"):
                cif_path = os.path.join(root, file)
                pdb_path = os.path.join(save_dir, file.replace(".cif", ".pdb"))
                print(f"Converting {cif_path}")

                if high_iptm:
                    confidence_files = [
                        f
                        for f in os.listdir(root)
                        if f.startswith("confidence_") and f.endswith(".json")
                    ]
                    if confidence_files:
                        confidence_file = os.path.join(root, confidence_files[0])
                        with open(confidence_file) as f:
                            confidence_data = json.load(f)
                            iptm = confidence_data["iptm"]
                        if iptm > i_ptm_cutoff:
                            print(f"Converting {cif_path}")
                            print(f"Confidence file: {confidence_file}")
                            print(f"iptm score: {iptm}")
                            convert_cif_to_pdb(cif_path, pdb_path)

                else:
                    print(f"Converting {cif_path}")
                    convert_cif_to_pdb(cif_path, pdb_path)

            if confidence_scores:
                confidence_scores_path = os.path.join(
                    save_dir, "high_iptm_confidence_scores.csv"
                )
                pd.DataFrame(confidence_scores).to_csv(
                    confidence_scores_path, index=False
                )
                print(f"Saved confidence scores to {confidence_scores_path}")


def binder_binds_contacts(
    pdb_path, binder_chain, target_chain, contact_residues, cutoff=10.0
):
    """
    Returns True if at least 20% of contact_residues on target_chain
    are contacted by any CA atom of binder_chain within cutoff angstroms.
    Works even if resname is UNK (or non-canonical).
    """
    if isinstance(contact_residues, str):
        if contact_residues.strip() == "":
            return True
        contact_residues = [int(x) for x in contact_residues.split(",") if x.strip()]

    structure = parsePDB(pdb_path)
    if structure is None:
        return False

    def get_chain_ca_atoms_and_resnums(chain_id):
        ca_atoms = []
        ca_resnums = []
        for atom in structure.iterAtoms():
            if atom.getChid() == chain_id and atom.getName() == "CA":
                ca_atoms.append(atom)
                ca_resnums.append(atom.getResnum())
        return ca_atoms, ca_resnums

    binder_ca_atoms, _ = get_chain_ca_atoms_and_resnums(binder_chain)
    target_ca_atoms, target_resnums = get_chain_ca_atoms_and_resnums(target_chain)

    if len(binder_ca_atoms) == 0 or len(target_ca_atoms) == 0:
        return False

    binder_coords = np.array([atom.getCoords() for atom in binder_ca_atoms])
    target_coords = np.array([atom.getCoords() for atom in target_ca_atoms])

    contact_indices = [
        i for i, resnum in enumerate(target_resnums) if resnum in contact_residues
    ]
    if not contact_indices:
        return False

    filtered_target_coords = target_coords[contact_indices]
    filtered_contact_resnums = [target_resnums[i] for i in contact_indices]

    # For each contact residue, check if any binder CA is within cutoff.
    contacted = 0
    for idx, c_resnum in enumerate(filtered_contact_resnums):
        c_coord = filtered_target_coords[idx]
        distances = np.sqrt(np.sum((binder_coords - c_coord) ** 2, axis=-1))
        if np.any(distances < cutoff):
            print(f"Contacted residue {c_resnum} by binder CA")
            contacted += 1

    if len(filtered_contact_resnums) == 0:
        return False
    return contacted >= 2


def sample_seq(length: int, exclude_P: bool = True, frac_X: float = 0.0) -> str:
    aas = "ACDEFGHIKLMNQRSTVWY" + ("" if exclude_P else "P")
    num_x = round(length * frac_X)
    pool = aas if aas else "X"
    seq_list = ["X"] * num_x + random.choices(pool, k=length - num_x)
    random.shuffle(seq_list)
    return "".join(seq_list)



# Amino acid conversion dict
restype_3to1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M',
}


def extract_sequence_from_structure(pdb_path, chain_id):
    """Extract sequence from PDB file"""
    structure = gemmi.read_structure(str(pdb_path))

    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                seq = []
                for residue in chain:
                    res_name = residue.name.strip().upper()
                    if res_name in restype_3to1:
                        seq.append(restype_3to1[res_name])
                return ''.join(seq)

    raise ValueError(f"Chain {chain_id} not found in {pdb_path}")


def shallow_copy_tensor_dict(d):
    if isinstance(d, dict):
        return {k: shallow_copy_tensor_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [shallow_copy_tensor_dict(x) for x in d]
    if isinstance(d, torch.Tensor):
        return d.detach().clone()
    return d


def smart_split(s):
    if "," in s:
        return [x.strip() for x in s.split(",")]
    if ":" in s:
        return [x.strip() for x in s.split(":")]
    return [x.strip() for x in s.split()] if s else []


def get_boltz_model(
    checkpoint: Optional[str] = None,
    predict_args=None,
    device: Optional[str] = None,
    model_version: str = "boltz2",
    grad_enabled=True,
    no_potentials=True,
) -> Boltz2:
    torch.set_grad_enabled(grad_enabled)
    torch.set_float32_matmul_precision("highest")
    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = 1.638  # Default value

    steering_args = BoltzSteeringParams()
    if no_potentials:
        steering_args.fk_steering = False
        steering_args.guidance_update = False

    pairformer_args = (
        PairformerArgsV2() if model_version == "boltz2" else PairformerArgs()
    )
    pairformer_args.v2 = True if model_version == "boltz2" else False
    pairformer_args.activation_checkpointing = True

    msa_args = MSAModuleArgs(
        subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True
    )
    msa_args.activation_checkpointing = True

    model_class = Boltz2 if model_version == "boltz2" else Boltz1
    if model_version == "boltz2":
        model_module = model_class.load_from_checkpoint(
            checkpoint,
            strict=False,
            predict_args=predict_args,
            map_location=device,
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            structure_prediction_training=True,
            no_msa=False,
            no_atom_encoder=False,
            use_templates=True,
            use_templates_v2=True,
            use_trifast=False,
            max_parallel_samples=1,
            steering_args=asdict(steering_args),
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
        )
    elif model_version == "boltz1":
        model_module = Boltz1.load_from_checkpoint(
            checkpoint,
            strict=False,
            predict_args=predict_args,
            map_location=device,
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            structure_prediction_training=True,
            no_msa=False,
            no_atom_encoder=False,
        )
    return model_module


def get_batch(
    target,
    ccd_path,
    ccd_lib,
    max_seqs=0,
    pocket_conditioning=False,
    keep_record=False,
    boltz_model_version=None,
):
    max_seqs = 4096
    structure = target.structure

    coords = np.array([(atom["coords"],) for atom in structure.atoms], dtype=Coords)
    ensemble = np.array([(0, len(coords))], dtype=Ensemble)

    if boltz_model_version == "boltz2":
        structure = StructureV2(
            atoms=structure.atoms,
            bonds=structure.bonds,
            residues=structure.residues,
            chains=structure.chains,
            interfaces=structure.interfaces,
            mask=structure.mask,
            coords=coords,
            ensemble=ensemble,
        )

    elif boltz_model_version == "boltz1":
        structure = Structure(
            atoms=structure.atoms,
            bonds=structure.bonds,
            residues=structure.residues,
            chains=structure.chains,
            interfaces=structure.interfaces,
            mask=structure.mask,
            connections=structure.connections,
        )

    msas = {}
    for chain in target.record.chains:
        msa_id = chain.msa_id
        if msa_id != -1:
            msa = np.load(msa_id)
            msas[chain.chain_id] = MSA(**msa)

    input = Input(
        structure,
        msas,
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )

    if boltz_model_version == "boltz2":
        tokenizer = Boltz2Tokenizer()
        featurizer = Boltz2Featurizer()
    elif boltz_model_version == "boltz1":
        tokenizer = BoltzTokenizer()
        featurizer = BoltzFeaturizer()

    tokenized = tokenizer.tokenize(input)

    # seed = 42
    # random = np.random.default_rng(seed)
    random = np.random.default_rng()
    if boltz_model_version == "boltz2":
        molecules = {}
        molecules.update(ccd_lib)
        molecules.update(input.extra_mols)
        mol_names = set(tokenized.tokens["res_name"].tolist())
        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(ccd_path, mol_names))
    options = target.record.inference_options
    if pocket_conditioning:
        pocket_constraints, contact_constraints = (
            options.pocket_constraints,
            options.contact_constraints,
        )
        print("pocket_constraints", pocket_constraints)
        print("contact_constraints", contact_constraints)
        if boltz_model_version == "boltz2":
            batch = featurizer.process(
                tokenized,
                random=random,
                molecules=molecules,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                compute_symmetries=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                inference_contact_constraints=contact_constraints,
                compute_constraint_features=True,
                compute_affinity=False,
            )
        elif boltz_model_version == "boltz1":
            binders, pocket = options.binders, options.pocket
            batch = featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binders,
                inference_pocket=pocket,
            )

    else:
        pocket_constraints = None

        if boltz_model_version == "boltz2":
            batch = featurizer.process(
                tokenized,
                random=random,
                molecules=molecules,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                compute_symmetries=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                compute_constraint_features=True,
                compute_affinity=False,
            )
        else:
            batch = featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=None,
                inference_pocket=None,
            )

    if keep_record:
        batch["record"] = target.record

    return batch, structure


def process_msa(chain_id: str, sequence: str, msa_dir: Path) -> bool:
    """Process MSA for a single chain."""
    msa_chain_dir = msa_dir / f"{chain_id}"
    env_dir = msa_chain_dir.with_name(f"{msa_chain_dir.name}_env")
    env_dir.mkdir(exist_ok=True)

    # Run MSA
    unpaired_msa = run_mmseqs2(
        [sequence],
        str(msa_chain_dir),
        use_env=True,
        use_pairing=False,
        host_url="https://api.colabfold.com",
        pairing_strategy="greedy",
    )

    # Save MSA results
    msa_a3m_path = env_dir / "msa.a3m"
    msa_a3m_path.write_text(unpaired_msa[0])

    # Process MSA if not already processed
    msa_npz_path = env_dir / "msa.npz"
    if not msa_npz_path.exists():
        msa = parse_a3m(
            msa_a3m_path,
            taxonomy=None,
            max_seqs=4096,
        )
        msa.dump(msa_npz_path)

    return msa_npz_path


def aggressive_memory_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()

    for _ in range(3):
        gc.collect()

    torch._dynamo.reset()
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    aggressive_memory_cleanup()


def run_prediction(
    data,
    binder_chain,
    seq=None,
    logmd=False,
    name=None,
    ccd_lib=None,
    ccd_path=None,
    boltz_model=None,
    randomly_kill_helix_feature=False,
    negative_helix_constant=0.1,
    device="cpu",
    boltz_model_version="boltz2",
    pocket_conditioning=False,
):
    if seq is not None:
        data["sequences"][chain_to_number[binder_chain]]["protein"]["sequence"] = seq
    target = parse_boltz_schema(
        name,
        data,
        ccd_lib,
        ccd_path,
        boltz_2=True if boltz_model_version == "boltz2" else False,
    )
    batch, structure = get_batch(
        target,
        ccd_path,
        ccd_lib,
        boltz_model_version=boltz_model_version,
        pocket_conditioning=pocket_conditioning,
    )
    batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
    output = boltz_model.predict_step(
        batch,
        batch_idx=0,
        dataloader_idx=0,
        randomly_kill_helix_feature=randomly_kill_helix_feature,
        negative_helix_constant=negative_helix_constant,
        binder_chain=binder_chain,
        logmd=logmd,
        structure=structure,
    )
    return output, structure


def save_pdb(structure, coords, plddts, filename):
    structure.atoms["coords"] = (
        coords[0].detach().cpu().numpy()[: structure.atoms["coords"].shape[0]]
    )
    with open(filename, "w") as f:
        f.write(to_pdb(structure, plddts, boltz2=True))


def design_sequence(
    designer,
    model_type,
    pdb_file,
    chains_to_design="A",
    omit_AA="C",
    bias_AA="",
    temperature=0.02,
    return_logits=False,
):
    seq, logits = designer.run(
        model_type=model_type,
        pdb_path=pdb_file,
        seed=111,
        chains_to_design=chains_to_design,
        bias_AA=bias_AA,
        omit_AA=omit_AA,
        return_logits=return_logits,
        extra_args={
            "--temperature": temperature,
            "--batch_size": 1,
        },
    )
    if return_logits:
        return seq, logits
    return seq[0], logits


def plot_from_pdb(
    pdb_file: str,
    width: int = 400,
    height: int = 400,
    style: str = "cartoon",
    color_by: str = "plddt",
):
    """
    Visualize a structure directly from a PDB file using py3Dmol.

    Args:
        pdb_file (str): Path to the PDB file.
        width (int): Width of the visualization window.
        height (int): Height of the visualization window.
        style (str): Visualization style (default: cartoon).
        color_by (str): Coloring scheme ("none", "plddt", "chain").
    """
    pdb_text = Path(pdb_file).read_text()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")

    # Normalize color_by
    color_by = (color_by or "none").lower()

    if color_by == "plddt":
        view.setStyle(
            {"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
            {
                style: {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "roygb",
                        "min": 50,
                        "max": 90,
                    }
                }
            },
        )
    elif color_by == "chain":
        view.setStyle(
            {"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
            {style: {"colorscheme": "chain"}},
        )
    else:
        view.setStyle(
            {"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
            {style: {}},
        )

    # Show ligands/heteroatoms as sticks
    view.setStyle(
        {"or": [{"resn": ["UNL", "LIG"]}, {"hetflag": True}]},
        {"stick": {"radius": 0.2}},
    )

    view.zoomTo()
    return view.show()


def plot_run_metrics(
    run_save_dir: str, name: str, run_id: int, num_cycles: int, run_metrics: dict
):
    """Plot per-run figure to run subfolder, given run_metrics as input."""
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    colors = ["#9B59B6", "#E94560", "#FF7F11", "#2ECC71"]
    metrics = [
        (
            "iPTM",
            [
                run_metrics.get(f"cycle_{i}_iptm", float("nan"))
                for i in range(num_cycles + 1)
            ],
            0,
            1,
            "{:.3f}",
        ),
        (
            "pLDDT",
            [
                run_metrics.get(f"cycle_{i}_plddt", float("nan"))
                for i in range(num_cycles + 1)
            ],
            0,
            1,
            "{:.1f}",
        ),
        (
            "iPLDDT",
            [
                run_metrics.get(f"cycle_{i}_iplddt", float("nan"))
                for i in range(num_cycles + 1)
            ],
            0,
            1,
            "{:.1f}",
        ),
        (
            "Alanine Count",
            [
                run_metrics.get(f"cycle_{i}_alanine", float("nan"))
                for i in range(num_cycles + 1)
            ],
            0,
            max(
                [
                    run_metrics.get(f"cycle_{i}_alanine", 0)
                    for i in range(num_cycles + 1)
                ]
            )
            + 2,
            "{}",
        ),
    ]
    design_cycles = list(range(num_cycles + 1))
    for ax, (label, values, ymin, ymax, fmt), color in zip(axs, metrics, colors):
        ax.plot(
            design_cycles,
            values,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=2,
        )
        ax.set(
            xlabel="Design Iteration",
            ylabel=label,
            title=f"{label} (Run {run_id})",
            xticks=design_cycles,
            ylim=(ymin, ymax),
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        for x, y in zip(design_cycles, values):
            ax.annotate(
                fmt.format(y) if not pd.isnull(y) else "",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(f"{run_save_dir}/{name}_run_{run_id}_design_cycle_results.png", dpi=300)
    plt.show()
    # plt.close()


def calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, binder_chain):
    """Calculate RMSD between holo and apo structures and update confidence CSV.

    Args:
        af_pdb_dir (str): Directory containing holo PDB files
        af_pdb_dir_apo (str): Directory containing apo PDB files
    """
    confidence_csv_path = af_pdb_dir + "/high_iptm_confidence_scores.csv"
    if os.path.exists(confidence_csv_path):
        df_confidence_csv = pd.read_csv(confidence_csv_path)
        for pdb_name in os.listdir(af_pdb_dir):
            if pdb_name.endswith(".pdb"):
                pdb_path = os.path.join(af_pdb_dir, pdb_name)
                pdb_path_apo = os.path.join(af_pdb_dir_apo, pdb_name)
                xyz_holo, _ = get_CA_and_sequence(pdb_path, chain_id=binder_chain)
                xyz_apo, _ = get_CA_and_sequence(pdb_path_apo, chain_id="A")
                rmsd = np_rmsd(np.array(xyz_holo), np.array(xyz_apo))
                df_confidence_csv.loc[
                    df_confidence_csv["file"] == pdb_name.split(".pdb")[0] + ".cif",
                    "rmsd",
                ] = rmsd
                print(f"{pdb_path} rmsd: {rmsd}")
        df_confidence_csv.to_csv(confidence_csv_path, index=False)


def run_alphafold_step(
    yaml_dir,
    alphafold_dir,
    af3_docker_name,
    af3_database_settings,
    hmmer_path,
    ligandmpnn_dir,
    work_dir,
    binder_id="A",
    gpu_id=0,
    high_iptm=False,
    use_msa_for_af3=False,
):
    """Run AlphaFold validation step"""
    print("Starting AlphaFold validation step...")

    alphafold_dir = os.path.expanduser(alphafold_dir)
    afdb_dir = os.path.expanduser(af3_database_settings)
    hmmer_path = os.path.expanduser(hmmer_path)
    print("alphafold_dir", alphafold_dir)
    print("afdb_dir", afdb_dir)
    print("hmmer_path", hmmer_path)

    # Create AlphaFold directories
    af_input_dir = f"{ligandmpnn_dir}/02_design_json_af3"
    af_output_dir = f"{ligandmpnn_dir}/02_design_final_af3"
    af_input_apo_dir = f"{ligandmpnn_dir}/02_design_json_af3_apo"
    af_output_apo_dir = f"{ligandmpnn_dir}/02_design_final_af3_apo"

    for dir_path in [af_input_dir, af_output_dir, af_input_apo_dir, af_output_apo_dir]:
        os.makedirs(dir_path, exist_ok=True)

    process_yaml_files(
        yaml_dir,
        af_input_dir,
        af_input_apo_dir,
        binder_chain=binder_id,
        use_msa_for_af3=use_msa_for_af3,
    )
    # Run AlphaFold on holo state
    subprocess.run(
        [
            f"{work_dir}/boltz/utils/alphafold.sh",
            af_input_dir,
            af_output_dir,
            str(gpu_id),
            alphafold_dir,
            af3_docker_name,
        ],
        check=True,
    )

    # Run AlphaFold on apo state
    subprocess.run(
        [
            f"{work_dir}/boltz/utils/alphafold.sh",
            af_input_apo_dir,
            af_output_apo_dir,
            str(gpu_id),
            alphafold_dir,
            af3_docker_name,
        ],
        check=True,
    )
    print("AlphaFold validation step completed!")
    af_pdb_dir = f"{ligandmpnn_dir}/03_af_pdb_success"
    af_pdb_dir_apo = f"{ligandmpnn_dir}/03_af_pdb_apo"

    convert_cif_files_to_pdb(
        af_output_dir, af_pdb_dir, af_dir=True, high_iptm=high_iptm
    )
    if not any(f.endswith(".pdb") for f in os.listdir(af_pdb_dir)):
        print("No successful designs from AlphaFold")
        sys.exit(0)
    convert_cif_files_to_pdb(af_output_apo_dir, af_pdb_dir_apo, af_dir=True)
    print("convert_cif_files_to_pdb completed!")
    calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, binder_id)
    print("calculate_holo_apo_rmsd completed!")

    return af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo


def run_rosetta_step(
    ligandmpnn_dir, af_pdb_dir, af_pdb_dir_apo, binder_id="A", target_type="protein"
):
    from utils.pyrosetta_utils import measure_rosetta_energy

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
