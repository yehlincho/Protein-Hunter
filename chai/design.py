import os
import numpy as np
import torch
import random
from torch import Tensor
import gemmi
from chai_lab.chai1 import _bin_centers
from chai_lab.data.parsing.structure.entity_type import EntityType
from typing import Optional
import gc

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

def get_backbone_coords_from_result(state):
    result = state.result
    # PATCH 9 (GLOBAL): self.state.batch -> self._current_batch
    # (state is passed in, so we use state.batch_inputs)
    inputs = state.batch_inputs
    if inputs is None:
        # Fallback for old state objects, though ideally restore_state handles this
        return torch.empty(0, 3, 3)

    # Get the mask for existing tokens (e.g., [T,T,F, T,T,F])
    token_mask = inputs["token_exists_mask"].squeeze(0).cpu()

    # Get ALL (padded) backbone indices
    all_bb_indices = inputs["token_backbone_frame_index"].squeeze(0).cpu()

    # Select ONLY the indices for existing tokens
    # This correctly gets all indices for chain 1, then all for chain 2
    bb_indices = all_bb_indices[token_mask]

    padded_coords = result['coords'].cpu()
    if padded_coords.dim() == 3:
        padded_coords = padded_coords.squeeze(0)

    # This indexing is now correct
    return padded_coords[bb_indices]

def prepare_refinement_coords(folder, parent_result, parent_batch_inputs):
    """
    Prepare coordinates for refinement from parent structure.
    - Combines robust chain-based matching (from Func 1) with
      protein refinement logic (from Func 2).
    - Protein tokens: Copies N, CA, C, O, CB from parent; places other sidechains at parent CB.
    - Ligand tokens: Copied exactly if atom count matches.
    """
    # --- Setup from Function 1 (Robust Data Loading) ---

    # FIX: parent_result["coords"] is already padded, so we just use it directly.
    parent_padded = parent_result["coords"].cpu()

    # Ensure parent_batch_inputs is not None and contains 'inputs'
    if parent_batch_inputs is None : # Removed "inputs" check as parent_batch_inputs *is* the inputs dict
        raise ValueError("parent_batch_inputs is missing or invalid in prepare_refinement_coords")


    parent_atom_mask = parent_batch_inputs["atom_exists_mask"][0].cpu()
    parent_token_idx = parent_batch_inputs["atom_token_index"][0].cpu()
    parent_token_asym_id = parent_batch_inputs["token_asym_id"][0].cpu()
    parent_token_entity_type = parent_batch_inputs["token_entity_type"][0].cpu()


    # Use folder.state.batch or folder.batch depending on your class structure
    # PATCH 9 (GLOBAL): self.state.batch -> self._current_batch
    # Ensure folder._current_batch is not None and contains 'inputs'
    if folder._current_batch is None or "inputs" not in folder._current_batch:
        raise ValueError("_current_batch is missing or invalid in prepare_refinement_coords")


    new_atom_mask = folder._current_batch["inputs"]["atom_exists_mask"][0].cpu()
    new_token_idx = folder._current_batch["inputs"]["atom_token_index"][0].cpu()
    new_token_asym_id = folder._current_batch["inputs"]["token_asym_id"][0].cpu()
    new_token_entity_type = folder._current_batch["inputs"]["token_entity_type"][0].cpu()

    # --- Added: Setup from Function 2 (Needed for Protein Logic) ---
    parent_atom_within_token = parent_batch_inputs["atom_within_token_index"][0].cpu() # Directly access keys
    # PATCH 9 (GLOBAL): self.state.batch -> self._current_batch
    new_atom_within_token = folder._current_batch["inputs"]["atom_within_token_index"][0].cpu()
    parent_bb_indices = parent_batch_inputs["token_backbone_frame_index"][0].cpu() # Directly access keys

    # --- Initialization ---
    n_atoms_new_padded = new_atom_mask.shape[0]
    new_padded = torch.zeros(n_atoms_new_padded, 3)

    # FIX: These lines are no longer needed as parent_padded is assigned above.
    # parent_padded = torch.zeros(parent_atom_mask.shape[0], 3)
    # parent_padded[parent_atom_mask] = parent_coords

    # --- Chain Matching Logic (from Function 1) ---
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

        # Get entity type (assuming all atoms in chain have same type)
        # Add check for empty mask to prevent indexing error
        if not parent_chain_mask.any() or not new_chain_mask.any():
            continue

        parent_entity = parent_atom_entity_type[parent_chain_mask][0].item()
        new_entity = new_atom_entity_type[new_chain_mask][0].item()

        assert parent_entity == new_entity, f"Chain {asym_id} changed entity type!"

        # --- Ligand Logic (from Function 1) ---
        if parent_entity == EntityType.LIGAND.value:
            parent_indices = torch.where(parent_chain_mask)[0]
            new_indices = torch.where(new_chain_mask)[0]

            assert len(parent_indices) == len(new_indices), \
                   f"Ligand atom count mismatch! Parent: {len(parent_indices)}, New: {len(new_indices)}"

            new_padded[new_indices] = parent_padded[parent_indices]

        elif parent_entity == EntityType.PROTEIN.value:
            # Find common tokens *within this chain*
            parent_tokens_in_chain = parent_token_idx[parent_chain_mask]
            new_tokens_in_chain = new_token_idx[new_chain_mask]

            common_tokens = set(torch.unique(parent_tokens_in_chain).tolist()) & set(torch.unique(new_tokens_in_chain).tolist())

            for token_id in common_tokens:
                # Get masks for this specific token
                parent_token_atoms_mask = (parent_token_idx == token_id) & parent_atom_mask
                new_token_atoms_mask = (new_token_idx == token_id) & new_atom_mask

                # 1. Find/calculate parent CB position
                parent_CB_atoms = parent_token_atoms_mask & (parent_atom_within_token == 3) # 3 = CB
                if parent_CB_atoms.any():
                    CB_pos = parent_padded[parent_CB_atoms][0]
                else:
                    # Glycine case: calculate CB (assuming 'extend' is defined)
                    N_idx, CA_idx, C_idx = parent_bb_indices[token_id]
                    N, CA, C = parent_padded[N_idx], parent_padded[CA_idx], parent_padded[C_idx]
                    CB_pos = extend(C, N, CA, 1.522, 1.927, -2.143) # Magic numbers from F2

                # 2. Set all new token atoms to this CB position
                new_padded[new_token_atoms_mask] = CB_pos

                # 3. Overwrite backbone atoms (N, CA, C, O, CB)
                for atom_idx in [0, 1, 2, 3, 4]:
                    parent_atoms = parent_token_atoms_mask & (parent_atom_within_token == atom_idx)
                    new_atoms = new_token_atoms_mask & (new_atom_within_token == atom_idx)

                    if parent_atoms.any() and new_atoms.any():
                        # Ensure we're only taking the first (and only) atom
                        new_padded[new_atoms] = parent_padded[parent_atoms][0]

    # Return the compact, unpadded tensor for the new structure
    return new_padded[new_atom_mask]

def is_smiles(seq):
    """Detect if sequence is SMILES string vs protein sequence"""
    smiles_chars = set('()[]=#@+-0123456789')
    return bool(set(seq.upper()) & smiles_chars)

def kabsch_rotation_matrix(
    mobile_centered: Tensor,
    target_centered: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute optimal rotation matrix using Kabsch algorithm.

    Args:
        mobile_centered: [N, 3] centered coordinates to rotate
        target_centered: [N, 3] centered reference coordinates
        weights: Optional [N, 1] weights for each point

    Returns:
        [3, 3] rotation matrix
    """
    # Compute covariance matrix
    if weights is not None:
        H = mobile_centered.T @ (weights * target_centered)
    else:
        H = mobile_centered.T @ target_centered

    # SVD
    U, S, Vt = torch.linalg.svd(H)
    R = U @ Vt

    # Handle reflection case
    if torch.det(R) < 0:
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

    # Handle masks - use subset for alignment computation
    if mobile_mask is not None and target_mask is not None:
        mobile_sub = mobile[mobile_mask]
        target_sub = target[target_mask]
        weights_sub = weights[mobile_mask] if weights is not None else None
    else:
        mobile_sub = mobile
        target_sub = target
        weights_sub = weights

    # Expand weights if needed
    if weights_sub is not None and weights_sub.ndim == 1:
        weights_sub = weights_sub.unsqueeze(-1)

    # Compute weighted centroids from subset
    if weights_sub is not None:
        weights_sum = weights_sub.sum(dim=0, keepdim=True).clamp(min=1e-8)
        centroid_mobile = (mobile_sub * weights_sub).sum(dim=0, keepdim=True) / weights_sum
        centroid_target = (target_sub * weights_sub).sum(dim=0, keepdim=True) / weights_sum
    else:
        centroid_mobile = mobile_sub.mean(dim=0, keepdim=True)
        centroid_target = target_sub.mean(dim=0, keepdim=True)

    # Center subset coordinates
    mobile_sub_centered = mobile_sub - centroid_mobile
    target_sub_centered = target_sub - centroid_target

    # Compute rotation matrix from subset
    R = kabsch_rotation_matrix(mobile_sub_centered, target_sub_centered, weights_sub)

    # Apply transformation to ALL mobile coordinates
    mobile_centered = mobile - centroid_mobile
    mobile_aligned = mobile_centered @ R + centroid_target

    return mobile_aligned

def compute_rmsd(coords1: Tensor, coords2: Tensor, mask: Optional[Tensor] = None) -> float:

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
    Compute CA/reference atom RMSD with different alignment modes.

    Args:
        coords1: [N, 3, 3] backbone coordinates (N-CA-C)
        coords2: [N, 3, 3] backbone coordinates
        mode: Alignment mode:
            - "all": Align all atoms, measure all RMSD
            - "target_align_binder_rmsd": Align by target, measure binder RMSD
            - "binder_align_ligand_com_rmsd": Align by binder, measure ligand COM distance
        n_target: Number of target residues (required for multi-chain modes)

    Returns:
        RMSD or distance value as float
    """
    # Extract CA atoms (middle atom)
    coords1 = coords1[:, 1, :].clone().float()
    coords2 = coords2[:, 1, :].clone().float()

    if mode == "target_align_binder_rmsd":
        # Create mask for target region
        n_total = coords1.shape[0]
        target_mask = torch.zeros(n_total, dtype=bool, device=coords1.device)
        target_mask[:n_target] = True

        # Align coords2 to coords1 using ONLY target region
        coords2_aligned = weighted_kabsch_align(
            coords2, coords1,
            mobile_mask=target_mask,
            target_mask=target_mask,
        )

        # Extract binder regions and measure RMSD
        binder_mask = ~target_mask
        return compute_rmsd(coords1, coords2_aligned, mask=binder_mask)

    elif mode == "binder_align_ligand_com_rmsd":
        # Split into ligand and binder
        ligand1, binder1 = coords1[:n_target], coords1[n_target:]
        ligand2, binder2 = coords2[:n_target], coords2[n_target:]

        # Compute ligand centers of mass
        ligand_com1 = ligand1.mean(dim=0)
        ligand_com2 = ligand2.mean(dim=0)

        # Align binder2 to binder1
        binder2_aligned = weighted_kabsch_align(binder2, binder1)

        # Get transformation parameters from binder alignment
        centroid_binder1 = binder1.mean(dim=0)
        centroid_binder2 = binder2.mean(dim=0)
        binder1_centered = binder1 - centroid_binder1
        binder2_centered = binder2 - centroid_binder2
        R = kabsch_rotation_matrix(binder2_centered, binder1_centered)

        # Apply same transformation to ligand COM
        ligand_com2_aligned = (ligand_com2 - centroid_binder2) @ R + centroid_binder1

        # Return distance between ligand COMs
        return torch.sqrt(torch.sum((ligand_com1 - ligand_com2_aligned) ** 2)).item()

    else:
        # Simple case: align everything, measure everything
        coords2_aligned = weighted_kabsch_align(coords2, coords1)
        return compute_rmsd(coords1, coords2_aligned)

def extract_backbone_from_cif(cif_file):
    """Extract N, CA, C backbone coordinates from CIF file."""
    import gemmi

    structure = gemmi.read_structure(str(cif_file))
    backbone_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                bb_atoms = []
                for atom_name in ['N', 'CA', 'C']:
                    atom = residue.find_atom(atom_name, '*')
                    if atom:
                        pos = atom.pos
                        bb_atoms.append([pos.x, pos.y, pos.z])
                    else:
                        # Missing atom - use dummy coords or skip residue
                        bb_atoms.append([0.0, 0.0, 0.0])

                if len(bb_atoms) == 3:
                    backbone_coords.append(bb_atoms)

    # Shape: [n_tokens, 3, 3] matching get_backbone_coords_from_result
    return torch.tensor(backbone_coords, dtype=torch.float32)

def optimize_protein_design(
    folder,
    designer,
    initial_seq,
    target_seq=None,
    target_pdb=None,
    target_chain=None,
    prefix="test",
    n_steps=5,
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    num_diffn_samples=1,
    temperature=0.1,
    use_esm=False,
    use_esm_target=False,
    use_alignment=True,
    align_to="all",
    scale_temp_by_plddt=False,
    partial_diffusion=0.0,
    pde_cutoff_intra=1.5,
    pde_cutoff_inter=3.0,
    omit_AA=None,
    randomize_template_sequence=True,
    cyclic=False,
    final_validation=True,
    verbose=False,
    viewer=None,
    render_freq=1,
):
    """
    Optimize protein design through iterative folding and sequence design.

    Args:
        folder: ChaiFolder instance
        designer: LigandMPNNWrapper instance
        initial_seq: Initial sequence to optimize
        target_seq: Target (protein sequence OR SMILES for ligand). If None with target_pdb, extracts from PDB.
        target_pdb: Optional template PDB
        target_chain: Chain ID in template PDB (required if target_pdb provided)
        prefix: Output file prefix
        n_steps: Number of optimization steps
        num_trunk_recycles: Trunk recycles per fold
        num_diffn_timesteps: Diffusion timesteps
        num_diffn_samples: Number of Diffusion samples per step
        temperature: MPNN sampling temperature
        use_esm: Use ESM for binder/unconditional (all iterations)
        use_esm_target: Use ESM for target protein (all iterations, ignored for ligands)
        use_alignment: Enable alignment during diffusion
        scale_temp_by_plddt: Scale MPNN temperature by inverse pLDDT
        partial_diffusion: Use partial diffusion refinement
        pde_cutoff_intra: Intra-chain PDE cutoff (Å)
        pde_cutoff_inter: Inter-chain PDE cutoff (Å)
        omit_AA: Amino acids to exclude from MPNN
        randomize_template_sequence: Randomize template sequence identity
        cyclic: Enable cyclic topology
        final_validation: Run final fold without templates
        verbose: Print detailed progress

    Supports:
        - Unconditional design (no target)
        - Protein-protein binder design
        - Protein-ligand binder design (target_seq = SMILES)
    """

    # Extract target sequence from PDB if not provided
    if target_pdb is not None and target_seq is None:
        if target_chain is None:
            raise ValueError("target_chain must be specified when using target_pdb")
        target_seq = extract_sequence_from_structure(target_pdb, target_chain)
        if verbose:
            print(f"{prefix} | Extracted target sequence from {target_pdb} chain {target_chain}: {target_seq[:60]}...")

    # Detect target type
    is_ligand_target = False
    if target_seq is not None:
        is_ligand_target = is_smiles(target_seq)

    is_binder_design = target_seq is not None
    mpnn_model_type = "ligand_mpnn" if is_ligand_target else "soluble_mpnn"
    target_entity_type = "ligand" if is_ligand_target else "protein"

    # Calculate PDE bins
    bin_centers = _bin_centers(0.0, 32.0, 64)
    pde_bins_intra = (bin_centers <= pde_cutoff_intra).sum().item()
    pde_bins_inter = (bin_centers <= pde_cutoff_inter).sum().item()

    if is_binder_design:
        if is_ligand_target:
            rmsd_mode = "binder_align_ligand_com_rmsd"
        else:
            rmsd_mode = "target_align_binder_rmsd"
    else:
        rmsd_mode = "all"

    def compute_pae_metrics(pae, n_target):
        """Compute PAE and iPAE"""
        if not is_binder_design:
            return {'pae': pae.mean().item(), 'ipae': None}

        mean_pae = pae.mean().item()
        if is_ligand_target:
            target_to_binder = pae[:n_target, n_target:].min(1).values.mean().item()
            binder_to_target = pae[n_target:, :n_target].min(0).values.mean().item()
        else:
            target_to_binder = pae[:n_target, n_target:].mean().item()
            binder_to_target = pae[n_target:, :n_target].mean().item()

        ipae = (target_to_binder + binder_to_target) / 2
        return {'pae': mean_pae, 'ipae': ipae}

    def compute_template_weight(prev_pde, n_target):
        """Compute PDE-based template weight"""
        if not is_binder_design:
            weight = prev_pde[..., :pde_bins_intra].sum(-1)

        else:
            n_total = prev_pde.shape[0]
            weight = torch.ones(n_total, n_total)
            weight[n_target:, n_target:] = prev_pde[n_target:, n_target:, :pde_bins_intra].sum(-1)
            weight[:n_target, n_target:] = prev_pde[:n_target, n_target:, :pde_bins_inter].sum(-1)
            weight[n_target:, :n_target] = prev_pde[n_target:, :n_target, :pde_bins_inter].sum(-1)

        return weight

    def fold_sequence(seq, prev=None, is_first_iteration=False):
        """Fold sequence and return metrics"""
        chains = []
        if is_binder_design:
            # Target chain
            align_target_weight = 10.0 if align_to in ["target","ligand"] else 1.0
            if is_first_iteration:
                if target_pdb is not None and not is_ligand_target:
                    # Protein target with template
                    target_opts = {
                        "use_esm": use_esm_target,
                        "template_pdb": target_pdb,
                        "template_chain_id": target_chain,
                        "align":align_target_weight,
                    }
                else:
                    # Protein target without template OR ligand target
                    target_opts = {"use_esm": use_esm_target and not is_ligand_target}
            else:
                if is_ligand_target:
                    # Ligand: no ESM
                    target_opts = {
                        "use_esm": False,
                        "align":align_target_weight,
                    }
                else:
                    # Protein target: use template + optional ESM
                    target_opts = {
                        "use_esm": use_esm_target,
                        "template_pdb": prev["pdb"],
                        "template_chain_id": "A",
                        "randomize_template_sequence": False,
                        "align":align_target_weight,
                    }
            chains.append([target_seq, "A", target_entity_type, target_opts])

            # Binder chain
            align_binder_weight = 10.0 if align_to == "binder" else 1.0
            binder_opts = {"use_esm": use_esm, "cyclic": cyclic, "align":align_binder_weight}
            if not is_first_iteration:
                binder_opts.update({
                    "template_pdb": prev["pdb"],
                    "template_chain_id": "B",
                    "randomize_template_sequence": randomize_template_sequence,
                })
            chains.append([seq, "B", "protein", binder_opts])
        else:
            # Unconditional
            opts = {"use_esm": use_esm, "cyclic": cyclic}
            if not is_first_iteration:
                opts.update({
                    "template_pdb": prev["pdb"],
                    "template_chain_id": "A",
                    "randomize_template_sequence": randomize_template_sequence,
                })
            chains.append([seq, "A", "protein", opts])

        # Fold
        if is_first_iteration:
            template_weight = None
        else:
            # Ensure prev state result is available
            if prev is None or "state" not in prev or prev["state"].result is None:
                 raise ValueError("Previous state or result is missing for template weight calculation.")

            template_weight = compute_template_weight(
                prev["state"].result["pde"],
                prev["n_target"]
            )

        folder.prep_inputs(chains)
        folder.get_embeddings()

        if viewer is not None and viewer._atom_types is None:
          # PATCH 9 (GLOBAL): self.state.batch -> self._current_batch
          # Use state.batch_inputs which is guaranteed to be present after prep_inputs
          token_exists = folder.state.batch_inputs["token_exists_mask"][0]
          n_total = token_exists.sum().item()
          if is_binder_design:
            n_binder = len(seq)
            n_target = n_total - n_binder
            if is_ligand_target:
              viewer._atom_types = ["L"] * n_target + ["P"] * n_binder
            else:
              viewer._atom_types = ["P"] * n_target + ["P"] * n_binder
            viewer._chains = ["A"] * n_target + ["B"] * n_binder

        folder.run_trunk(num_trunk_recycles=num_trunk_recycles,
                         template_weight=template_weight)

        # Partial diffusion
        refine_coords = None
        refine_step = None
        if partial_diffusion > 0 and prev is not None:
             # Ensure prev state result and batch_inputs are available
            if "state" not in prev or prev["state"].result is None or prev["state"].batch_inputs is None:
                 print("Warning: Skipping partial diffusion due to missing previous state data.")
            else:
                refine_coords = prepare_refinement_coords(
                    folder,
                    prev["state"].result,
                    # Pass the CPU batch_inputs dictionary directly
                    prev["state"].batch_inputs
                )
                refine_step = int(num_diffn_timesteps * partial_diffusion)


        # Sample
        folder.sample(
            num_diffn_timesteps=num_diffn_timesteps,
            num_diffn_samples=num_diffn_samples,
            use_alignment=use_alignment,
            refine_from_coords=refine_coords,
            refine_from_step=refine_step,
            viewer=viewer,
            render_freq=render_freq,
        )
        return folder.save_state()

    def design_sequence(step, prev):
        """Design sequence with MPNN"""
        temp_per_residue = None
        # Ensure prev state and result exist before accessing plddt
        if scale_temp_by_plddt and prev and "state" in prev and prev["state"].result and "plddt" in prev["state"].result:

            chain = "B" if is_binder_design else "A"
            # Get plddt per token directly
            plddt_per_token = prev["state"].result["plddt"].numpy()

            # Apply only to the designed chain if binder design
            if is_binder_design:
                 n_binder = len(prev["seq"]) # Use length of sequence from prev dict
                 # Need n_target to index correctly
                 if "n_target" not in prev:
                      # Attempt to calculate n_target if missing
                      if prev["state"].batch_inputs:
                           token_exists = prev["state"].batch_inputs["token_exists_mask"][0]
                           n_total = token_exists.sum().item()
                           prev["n_target"] = n_total - n_binder
                      else:
                           print("Warning: Cannot determine n_target for plddt scaling.")
                           plddt_per_token = np.array([]) # Avoid error below
                 if "n_target" in prev:
                      plddt_binder = plddt_per_token[prev["n_target"]:]
                      inv_plddt = np.square(1 - plddt_binder)
                 else: # Fallback if n_target couldn't be found
                      inv_plddt = np.array([])
            else:
                 inv_plddt = np.square(1 - plddt_per_token)


            temp_per_residue = {f"{chain}{i+1}": float(v) for i, v in enumerate(inv_plddt)}

        extra_args = {"--batch_size": 1}
        BASE_DIR = os.path.dirname(os.path.abspath(designer.run_py))
        MODEL_DIR = os.path.join(BASE_DIR, "model_params")
        if mpnn_model_type == "ligand_mpnn":
            extra_args["--checkpoint_ligand_mpnn"] = os.path.join(MODEL_DIR, "ligandmpnn_v_32_010_25.pt")
        else:
            extra_args["--checkpoint_soluble_mpnn"] = os.path.join(MODEL_DIR, "solublempnn_v_48_020.pt")
        if omit_AA:
            extra_args["--omit_AA"] = omit_AA

        sequences, _ = designer.run(
            model_type=mpnn_model_type,
            pdb_path=prev["pdb"],
            seed=111 + step,
            chains_to_design="B" if is_binder_design else "A",
            temperature=temperature,
            temperature_per_residue=temp_per_residue,
            extra_args=extra_args,
        )

        seq = sequences[0]
        if is_binder_design:
            # Handle potential ":" if MPNN outputs combined sequence
            if ":" in seq:
                 seq = seq.split(":")[-1]


        return seq

    def format_metrics(prev, rmsd=None):
        """Format metrics for printing"""
        # Ensure state and result are present
        if not prev or "state" not in prev or not prev["state"] or not prev["state"].result:
             print("Warning: Cannot format metrics, missing state or result.")
             return "Metrics unavailable"

        # Compute n_target from actual structure
        # Use batch_inputs from the state
        if prev["state"].batch_inputs:
             token_exists = prev["state"].batch_inputs["token_exists_mask"][0]
             n_total = token_exists.sum().item()
             # Ensure 'seq' is in prev for length calculation
             if "seq" in prev:
                  prev["n_target"] = (n_total - len(prev["seq"]))
             else:
                  prev["n_target"] = None # Indicate we couldn't calculate it
                  print("Warning: Cannot calculate n_target for metrics, 'seq' missing.")
        else:
             prev["n_target"] = None
             print("Warning: Cannot calculate n_target for metrics, 'batch_inputs' missing.")


        result = prev["state"].result
        # Use calculated n_target if available
        pae_metrics = compute_pae_metrics(result["pae"], prev["n_target"] if prev["n_target"] is not None else 0)

        out = {
            'plddt': result["plddt"].numpy().mean(),
            'ptm': result["ptm"].item(),
            'iptm': result["iptm"].item() if is_binder_design else 0.0,
            'pae': pae_metrics['pae'],
            'ipae': pae_metrics['ipae'],
            'ranking_score': result["ranking_score"],
        }
        msg = f"score={out['ranking_score']:.3f} plddt={out['plddt']:.1f} ptm={out['ptm']:.3f}"
        if is_binder_design:
            msg += f" iptm={out['iptm']:.3f} ipae={out['ipae']:.2f}" # Handle None ipae
        else:
            msg += f" pae={out['pae']:.2f}"
        if rmsd is not None:
            msg += f" rmsd={rmsd:.2f}"
        return msg

    def copy_prev(prev):
        x = {**prev}
        x["state"] = x["state"].copy()
        x["bb"] = x.get("bb", torch.empty(0, 3, 3)).clone() # Add safety check for bb
        return x

    # === MAIN LOOP ===

    # Step 0
    prev = {"seq":initial_seq}
    try:
        prev["state"] = fold_sequence(prev["seq"], is_first_iteration=True)
        # Check if folding succeeded and produced results
        if prev["state"] is None or prev["state"].result is None:
             raise RuntimeError("Initial folding failed to produce results.")

        prev["bb"] = get_backbone_coords_from_result(prev["state"])
        prev["pdb"] = prefix + "_s0.cif"
        folder.save(prev["pdb"]) # This requires state.batch_inputs to be set by fold_sequence

        msg = format_metrics(prev)
        print(f"{prefix} | Step 0: {msg}")

    except Exception as e:
        print(f"Error during initial folding (Step 0): {e}")
        # Decide how to handle: skip trial, return None, etc.
        return None # Example: return None if initial fold fails


    best_step = 0
    best = copy_prev(prev)
    # Optimization steps
    for step in range(n_steps):
        try: # Wrap step in try/except
            # Design sequence
            new_seq = design_sequence(step, prev)
            new = {"seq": new_seq}

            # Fold new sequence
            new["state"] = fold_sequence(new["seq"], prev)
            if new["state"] is None or new["state"].result is None:
                 print(f"Warning: Folding failed for step {step+1}, skipping evaluation.")
                 # Decide how to proceed: use previous state, break, continue?
                 # Option: keep previous state for next design, but don't update 'best'
                 # prev['seq'] = new_seq # Keep new sequence? Or revert? Depends on strategy
                 gc.collect()
                 torch.cuda.empty_cache()
                 continue # Skip rest of the loop for this step

            new["bb"] = get_backbone_coords_from_result(new["state"])
            new["pdb"] = f"{prefix}_s{step+1}.cif"
            folder.save(new["pdb"])

            # Compute RMSD (needs prev['n_target'] which is set in format_metrics)
            # Make sure prev['n_target'] was set successfully in the previous iteration
            if prev.get("n_target") is not None:
                rmsd = compute_ca_rmsd(prev["bb"], new["bb"], mode=rmsd_mode, n_target=prev["n_target"])
            else:
                rmsd = None # Cannot compute if n_target is missing

            msg = format_metrics(new, rmsd)
            print(f"{prefix} | Step {step+1}: {msg}")

            # Update best only if folding was successful
            if new["state"].result['ranking_score'] > best["state"].result['ranking_score']:
                best = copy_prev(new)
                best_step = step + 1

            prev = new # Update prev only if step was successful

        except Exception as e:
            print(f"Error during optimization step {step+1}: {e}")
            # Decide if you want to continue to the next step or break
            # break # Example: stop this trial if an error occurs

        finally: # Ensure cleanup happens even if there's an error
            gc.collect()
            torch.cuda.empty_cache()


    # Restore best (only if 'best' was successfully initialized)
    if best and best.get("state"):
        try:
            folder.restore_state(best["state"])
            folder.save(f"{prefix}_best.cif")
        except Exception as e:
            print(f"Error restoring/saving best state: {e}")
    else:
        print("Warning: No valid 'best' state found to restore.")
        return None # Or return the last valid 'prev' if desired


    if final_validation and best and best.get("state"): # Ensure 'best' is valid
        try: # Wrap validation in try/except
            # Validation
            chains = []
            if is_binder_design:
                # Target: use original context
                if is_ligand_target:
                    target_opts = {"use_esm": False}
                elif target_pdb is not None:
                    target_opts = {
                        "use_esm": use_esm_target,
                        "template_pdb": target_pdb,
                        "template_chain_id": target_chain,
                    }
                else:
                    target_opts = {"use_esm": use_esm_target}
                chains.append([target_seq, "A", target_entity_type, target_opts])

                # Binder: no template, but can use ESM
                binder_opts = {"use_esm": use_esm, "cyclic": cyclic}
                chains.append([best['seq'], "B", "protein", binder_opts])
            else:
                # Unconditional: no template, but can use ESM
                opts = {"use_esm": use_esm, "cyclic": cyclic}
                chains.append([best['seq'], "A", "protein", opts])

            folder.prep_inputs(chains)
            folder.get_embeddings()
            folder.run_trunk(num_trunk_recycles=num_trunk_recycles, template_weight=None)
            folder.sample(num_diffn_timesteps=num_diffn_timesteps,
                          num_diffn_samples=num_diffn_samples,
                          use_alignment=use_alignment)

            # Check if sampling produced a result
            if folder.state is None or folder.state.result is None:
                 print("Warning: Final validation sampling failed to produce results.")
                 # Handle appropriately, maybe skip RMSD calculation
            else:
                val = {
                    "seq":best["seq"],
                    "state":folder.save_state(),
                    "pdb":f"{prefix}_validation.cif",
                }
                folder.save(val["pdb"])
                val["bb"] = get_backbone_coords_from_result(val["state"])

                # Validation RMSD
                # Ensure best['n_target'] exists before calculating RMSD
                if best.get("n_target") is not None:
                     val_rmsd = compute_ca_rmsd(best["bb"], val["bb"], mode=rmsd_mode, n_target=best["n_target"])
                else:
                     val_rmsd = None # Cannot compute RMSD

                msg = format_metrics(val, val_rmsd)
                print(f"{prefix} | Validation: {msg}")

                best["val"] = val

        except Exception as e:
            print(f"Error during final validation: {e}")

        finally: # Cleanup after validation
            gc.collect()
            torch.cuda.empty_cache()

    return best