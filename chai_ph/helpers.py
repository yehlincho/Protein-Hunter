import torch
import random
import gemmi
import numpy as np
from torch import Tensor
from typing import Optional
from chai_lab.data.parsing.structure.entity_type import EntityType

# Amino acid conversion dict
restype_3to1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}

# --- Sequence Utilities ---

def sample_seq(length: int, exclude_P: bool = True, frac_X: float = 0.0) -> str:
    """Generates a random amino acid sequence."""
    aas = "ACDEFGHIKLMNQRSTVWY" + ("" if exclude_P else "P")
    num_x = round(length * frac_X)
    pool = aas if aas else "X"
    seq_list = ["X"] * num_x + random.choices(pool, k=length - num_x)
    random.shuffle(seq_list)
    return "".join(seq_list)

def is_smiles(seq: str) -> bool:
    """Detect if sequence is SMILES string vs protein sequence"""
    smiles_chars = set("()[]=#@+-0123456789")
    return bool(set(seq.upper()) & smiles_chars)

def clean_protein_sequence(input_string: str) -> str:
    """
    Cleans a string to represent a valid protein sequence, replacing
    non-standard AA codes with 'X'.
    """
    amino_acids = {
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    }
    clean_sequence = []
    for char in input_string:
        if char.isalpha():
            upper_char = char.upper()
            if upper_char in amino_acids:
                clean_sequence.append(upper_char)
            else:
                clean_sequence.append("X")
    return "".join(clean_sequence)

def extract_sequence_from_pdb(pdb_path, chain_id):
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
                return "".join(seq)
    raise ValueError(f"Chain {chain_id} not found in {pdb_path}")

# --- Coordinate & Geometry Utilities ---

def extend(a, b, c, L, A, D):
    """
    Place 4th atom given 3 atoms and ideal geometry. Works for single or batched inputs.

    Args:
        a, b, c: Atom positions [3] or [batch, 3]
        L: Bond length from c to new atom
        A: Bond angle at c (radians)
        D: Dihedral angle (radians)

    Returns:
        Position of 4th atom [3] or [batch, 3]
    """
    ba = b - a
    bc = b - c

    x = bc * torch.rsqrt(torch.sum(bc * bc, dim=-1, keepdim=True) + 1e-8)
    z = torch.linalg.cross(ba, x)
    z = z * torch.rsqrt(torch.sum(z * z, dim=-1, keepdim=True) + 1e-8)
    y = torch.linalg.cross(z, x)

    cos_A, sin_A, cos_D, sin_D = np.cos(A), np.sin(A), np.cos(D), np.sin(D)
    L_sin_A = L * sin_A

    return c + L * cos_A * x + L_sin_A * cos_D * y - L_sin_A * sin_D * z

def get_backbone_coords_from_result(state) -> Tensor:
    """Extracts (N, CA, C) coordinates from a ChaiFolder state object."""
    result = state.result
    inputs = state.batch_inputs
    if inputs is None:
        return torch.empty(0, 3, 3)

    token_mask = inputs["token_exists_mask"].squeeze(0).cpu()
    all_bb_indices = inputs["token_backbone_frame_index"].squeeze(0).cpu()
    bb_indices = all_bb_indices[token_mask]
    
    padded_coords = result["coords"].cpu()
    if padded_coords.dim() == 3:
        padded_coords = padded_coords.squeeze(0)
        
    return padded_coords[bb_indices]

def prepare_refinement_coords(folder, parent_result, parent_batch_inputs) -> Tensor:
    """
    Prepare coordinates for refinement (partial diffusion) from a parent structure.
    - Copies protein backbone and CB atoms.
    - Places new sidechain atoms at the parent CB position.
    - Copies ligand atoms directly if atom counts match.
    """
    parent_padded = parent_result["coords"].cpu()

    if parent_batch_inputs is None:
        raise ValueError("parent_batch_inputs is missing in prepare_refinement_coords")

    parent_atom_mask = parent_batch_inputs["atom_exists_mask"][0].cpu()
    parent_token_idx = parent_batch_inputs["atom_token_index"][0].cpu()
    parent_token_asym_id = parent_batch_inputs["token_asym_id"][0].cpu()
    parent_token_entity_type = parent_batch_inputs["token_entity_type"][0].cpu()
    parent_atom_within_token = parent_batch_inputs["atom_within_token_index"][0].cpu()
    parent_bb_indices = parent_batch_inputs["token_backbone_frame_index"][0].cpu()

    if folder._current_batch is None or "inputs" not in folder._current_batch:
        raise ValueError("_current_batch is missing in prepare_refinement_coords")

    new_inputs = folder._current_batch["inputs"]
    new_atom_mask = new_inputs["atom_exists_mask"][0].cpu()
    new_token_idx = new_inputs["atom_token_index"][0].cpu()
    new_token_asym_id = new_inputs["token_asym_id"][0].cpu()
    new_token_entity_type = new_inputs["token_entity_type"][0].cpu()
    new_atom_within_token = new_inputs["atom_within_token_index"][0].cpu()

    n_atoms_new_padded = new_atom_mask.shape[0]
    new_padded = torch.zeros(n_atoms_new_padded, 3)

    parent_atom_asym_id = parent_token_asym_id[parent_token_idx]
    new_atom_asym_id = new_token_asym_id[new_token_idx]
    parent_atom_entity_type = parent_token_entity_type[parent_token_idx]
    new_atom_entity_type = new_token_entity_type[new_token_idx]

    parent_asym_ids = torch.unique(parent_atom_asym_id[parent_atom_mask])
    new_asym_ids = torch.unique(new_atom_asym_id[new_atom_mask])
    common_asym_ids = set(parent_asym_ids.tolist()) & set(new_asym_ids.tolist())

    for asym_id in sorted(common_asym_ids):
        parent_chain_mask = (parent_atom_asym_id == asym_id) & parent_atom_mask
        new_chain_mask = (new_atom_asym_id == asym_id) & new_atom_mask

        if not parent_chain_mask.any() or not new_chain_mask.any():
            continue

        parent_entity = parent_atom_entity_type[parent_chain_mask][0].item()
        new_entity = new_atom_entity_type[new_chain_mask][0].item()

        assert parent_entity == new_entity, f"Chain {asym_id} changed entity type!"

        if parent_entity == EntityType.LIGAND.value:
            parent_indices = torch.where(parent_chain_mask)[0]
            new_indices = torch.where(new_chain_mask)[0]
            assert len(parent_indices) == len(new_indices), \
                   f"Ligand atom count mismatch! Parent: {len(parent_indices)}, New: {len(new_indices)}"
            new_padded[new_indices] = parent_padded[parent_indices]

        elif parent_entity == EntityType.PROTEIN.value:
            parent_tokens_in_chain = parent_token_idx[parent_chain_mask]
            new_tokens_in_chain = new_token_idx[new_chain_mask]
            common_tokens = set(torch.unique(parent_tokens_in_chain).tolist()) & set(torch.unique(new_tokens_in_chain).tolist())

            for token_id in common_tokens:
                parent_token_atoms_mask = (parent_token_idx == token_id) & parent_atom_mask
                new_token_atoms_mask = (new_token_idx == token_id) & new_atom_mask

                # 1. Find/calculate parent CB position
                parent_CB_atoms = parent_token_atoms_mask & (parent_atom_within_token == 3)  # 3 = CB
                if parent_CB_atoms.any():
                    CB_pos = parent_padded[parent_CB_atoms][0]
                else: # Glycine case
                    N_idx, CA_idx, C_idx = parent_bb_indices[token_id]
                    N, CA, C = parent_padded[N_idx], parent_padded[CA_idx], parent_padded[C_idx]
                    CB_pos = extend(C, N, CA, 1.522, 1.927, -2.143) # Ideal geometry

                # 2. Set all new token atoms to this CB position
                new_padded[new_token_atoms_mask] = CB_pos

                # 3. Overwrite backbone atoms (N, CA, C, O, CB)
                for atom_idx in [0, 1, 2, 3, 4]:
                    parent_atoms = parent_token_atoms_mask & (parent_atom_within_token == atom_idx)
                    new_atoms = new_token_atoms_mask & (new_atom_within_token == atom_idx)
                    if parent_atoms.any() and new_atoms.any():
                        new_padded[new_atoms] = parent_padded[parent_atoms][0]

    return new_padded[new_atom_mask]


# --- Kabsch & RMSD Utilities ---

def kabsch_rotation_matrix(
    mobile_centered: Tensor,
    target_centered: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Compute optimal rotation matrix using Kabsch algorithm."""
    if weights is not None:
        H = mobile_centered.T @ (weights * target_centered)
    else:
        H = mobile_centered.T @ target_centered
    U, S, Vt = torch.linalg.svd(H)
    R = U @ Vt
    if torch.det(R) < 0: # Handle reflection
        Vt[-1, :] *= -1
        R = U @ Vt
    return R

def weighted_kabsch_align(
    mobile: Tensor,
    target: Tensor,
    weights: Optional[Tensor] = None,
    mobile_mask: Optional[Tensor] = None,
    target_mask: Optional[Tensor] = None,
) -> Tensor:
    """Aligns 'mobile' coordinates to 'target' coordinates using weighted Kabsch."""
    if mobile_mask is not None and target_mask is not None:
        mobile_sub = mobile[mobile_mask]
        target_sub = target[target_mask]
        weights_sub = weights[mobile_mask] if weights is not None else None
    else:
        mobile_sub, target_sub, weights_sub = mobile, target, weights

    if weights_sub is not None and weights_sub.ndim == 1:
        weights_sub = weights_sub.unsqueeze(-1)

    if weights_sub is not None:
        weights_sum = weights_sub.sum(dim=0, keepdim=True).clamp(min=1e-8)
        centroid_mobile = (mobile_sub * weights_sub).sum(dim=0, keepdim=True) / weights_sum
        centroid_target = (target_sub * weights_sub).sum(dim=0, keepdim=True) / weights_sum
    else:
        centroid_mobile = mobile_sub.mean(dim=0, keepdim=True)
        centroid_target = target_sub.mean(dim=0, keepdim=True)

    mobile_sub_centered = mobile_sub - centroid_mobile
    target_sub_centered = target_sub - centroid_target

    R = kabsch_rotation_matrix(mobile_sub_centered, target_sub_centered, weights_sub)

    mobile_centered = mobile - centroid_mobile
    mobile_aligned = mobile_centered @ R + centroid_target
    return mobile_aligned

def compute_rmsd(
    coords1: Tensor, coords2: Tensor, mask: Optional[Tensor] = None
) -> float:
    """Compute simple RMSD between two sets of coordinates."""
    if mask is not None:
        coords1 = coords1[mask]
        coords2 = coords2[mask]
    return torch.sqrt(torch.mean((coords1 - coords2) ** 2)).item()

def compute_ca_rmsd(
    coords1: Tensor,
    coords2: Tensor,
    mode: str = "all",
    n_target: Optional[int] = None,
) -> float:
    """
    Compute C-alpha RMSD with different alignment modes.
    (Expects backbone coords [N, 3, 3]).
    """
    ca1 = coords1[:, 1, :].clone().float()
    ca2 = coords2[:, 1, :].clone().float()

    if mode == "target_align_binder_rmsd":
        n_total = ca1.shape[0]
        target_mask = torch.zeros(n_total, dtype=bool, device=ca1.device)
        target_mask[:n_target] = True
        
        ca2_aligned = weighted_kabsch_align(
            ca2, ca1, mobile_mask=target_mask, target_mask=target_mask
        )
        return compute_rmsd(ca1, ca2_aligned, mask=~target_mask)

    elif mode == "binder_align_ligand_com_rmsd":
        ligand1, binder1 = ca1[:n_target], ca1[n_target:]
        ligand2, binder2 = ca2[:n_target], ca2[n_target:]

        ligand_com1 = ligand1.mean(dim=0)
        ligand_com2 = ligand2.mean(dim=0)

        centroid_binder1 = binder1.mean(dim=0)
        centroid_binder2 = binder2.mean(dim=0)
        R = kabsch_rotation_matrix(
            binder2 - centroid_binder2, binder1 - centroid_binder1
        )
        
        ligand_com2_aligned = (ligand_com2 - centroid_binder2) @ R + centroid_binder1
        return torch.sqrt(torch.sum((ligand_com1 - ligand_com2_aligned) ** 2)).item()

    else: # mode == "all"
        ca2_aligned = weighted_kabsch_align(ca2, ca1)
        return compute_rmsd(ca1, ca2_aligned)