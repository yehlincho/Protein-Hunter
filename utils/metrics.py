import os
import numpy as np
import torch
from Bio.PDB import PDBParser, MMCIFParser
from boltz_ph.constants import RESTYPE_3TO1

def np_kabsch(a, b, return_v=False):
    """
    Computes the optimal rotation matrix to align coordinates 'a' onto 'b' 
    using the Kabsch algorithm (numpy implementation).

    Args:
        a (np.ndarray): First set of coordinates (N, 3).
        b (np.ndarray): Second set of coordinates (N, 3).
        return_v (bool): If True, returns the U matrix from SVD.
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
    """
    Compute RMSD of coordinates after optimal alignment using numpy.

    Args:
        true (np.ndarray): Reference coordinates (N, 3).
        pred (np.ndarray): Predicted coordinates (N, 3) to align.

    Returns:
        float: Root mean square deviation after optimal alignment.
    """
    # Center coordinates
    p = true - np.mean(true, axis=-2, keepdims=True)
    q = pred - np.mean(pred, axis=-2, keepdims=True)

    # Get optimal rotation matrix and apply it
    p = p @ np_kabsch(p, q)

    # Calculate RMSD
    return np.sqrt(np.mean(np.sum(np.square(p - q), axis=-1)) + 1e-8)


def get_CA_and_sequence(structure_file, chain_id="A"):
    """
    Extracts C-alpha coordinates and 1-letter amino acid sequence for a specific chain
    from a PDB or CIF file.
    """
    # Determine file type and use appropriate parser
    if structure_file.lower().endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    elif structure_file.lower().endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("File must be either .cif or .pdb format")

    structure = parser.get_structure("structure", structure_file)
    xyz = []
    sequence = []
    
    model = structure[0]

    if chain_id in model:
        chain = model[chain_id]
        for residue in chain:
            res_name = residue.resname.strip().upper()
            if "CA" in residue and res_name in RESTYPE_3TO1:
                xyz.append(residue["CA"].coord)
                sequence.append(RESTYPE_3TO1[res_name])
            elif res_name not in RESTYPE_3TO1:
                 # Handle non-canonical residues by skipping or using 'X' if appropriate
                 if "CA" in residue:
                    xyz.append(residue["CA"].coord)
                    sequence.append("X")

    if not xyz:
         raise ValueError(f"Chain {chain_id} not found or contains no C-alpha atoms in {structure_file}")
         
    return np.array(xyz), "".join(sequence)


def radius_of_gyration(path, chain_id="B"):
    """Calculate the Radius of Gyration (Rg) for the CA atoms of a specified chain."""
    if path.lower().endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
        
    structure = parser.get_structure("structure", path)
    model = structure[0]
    
    if chain_id not in model:
         raise ValueError(f"Chain {chain_id} not found in {path}")
         
    chain = model[chain_id]
    
    ca_coords = [
        atom.get_coord()
        for residue in chain
        for atom in residue
        if atom.get_name() == "CA"
    ]
    
    if not ca_coords:
        raise ValueError(f"No C-alpha atoms found in chain {chain_id} of {path}")
        
    ca_coords_tensor = torch.tensor(ca_coords)
    
    # Rg calculation
    rg = torch.sqrt(torch.square(ca_coords_tensor - ca_coords_tensor.mean(0)).sum(-1).mean() + 1e-8)
    return rg.item(), len(ca_coords_tensor)