import random
import tempfile
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from einops import repeat, einsum as einops_einsum, rearrange

import math
import gemmi
import py3Dmol
import collections
import numpy as np

from chai_lab.chai1 import _bin_centers, DiffusionConfig, load_exported
from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES
from chai_lab.data.collate.collate import Collate
from chai_lab.data.dataset.all_atom_feature_context import AllAtomFeatureContext
from chai_lab.data.dataset.inference_dataset import Input, load_chains_from_raw
from chai_lab.data.dataset.msas.msa_context import MSAContext
from chai_lab.data.parsing.msas.data_source import MSADataSource
from chai_lab.data.dataset.embeddings.embedding_context import EmbeddingContext
from chai_lab.data.dataset.embeddings.esm import get_esm_embedding_context
from chai_lab.data.dataset.constraints.restraint_context import RestraintContext
from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import AllAtomResidueTokenizer
from chai_lab.data.dataset.structure.all_atom_structure_context import AllAtomStructureContext
from chai_lab.data.dataset.structure.chain import Chain
from chai_lab.data.dataset.templates.context import TemplateContext
from chai_lab.data.dataset.all_atom_feature_context import MAX_NUM_TEMPLATES
from chai_lab.data.features.generators.token_bond import TokenBondRestraint
from chai_lab.data.parsing.structure.entity_type import EntityType
from chai_lab.data.sources.rdkit import RefConformerGenerator
from chai_lab.data.io.cif_utils import save_to_cif, get_chain_letter
from chai_lab.data.io.pdb_utils import pdb_context_from_batch
from chai_lab.data.features.token_utils import get_token_reference_atom_positions_and_mask
from chai_lab.tools.rigid import Rigid
from chai_lab.utils.tensor_utils import move_data_to_device, cdist
from chai_lab.ranking.frames import get_frames_and_mask
from chai_lab.ranking.rank import rank
from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule
from chai_lab.model.utils import center_random_augmentation

from chai_lab.data.features.generators.base import EncodingType
from chai_lab.data.features.generators.atom_element import AtomElementOneHot
from chai_lab.data.features.generators.atom_name import AtomNameOneHot
from chai_lab.data.features.generators.blocked_atom_pair_distances import (
    BlockedAtomPairDistances,
    BlockedAtomPairDistogram,
)
from chai_lab.data.features.generators.docking import DockingRestraintGenerator
from chai_lab.data.features.generators.esm_generator import ESMEmbeddings
from chai_lab.data.features.generators.identity import Identity
from chai_lab.data.features.generators.is_cropped_chain import ChainIsCropped
from chai_lab.data.features.generators.missing_chain_contact import MissingChainContact
from chai_lab.data.features.generators.msa import (
    IsPairedMSAGenerator,
    MSADataSourceGenerator,
    MSADeletionMeanGenerator,
    MSADeletionValueGenerator,
    MSAFeatureGenerator,
    MSAHasDeletionGenerator,
    MSAProfileGenerator,
)
from chai_lab.data.features.generators.ref_pos import RefPos
from chai_lab.data.features.generators.relative_chain import RelativeChain
from chai_lab.data.features.generators.relative_entity import RelativeEntity
from chai_lab.data.features.generators.relative_sep import RelativeSequenceSeparation
from chai_lab.data.features.generators.relative_token import RelativeTokenSeparation
from chai_lab.data.features.generators.residue_type import ResidueType
from chai_lab.data.features.generators.structure_metadata import (
    IsDistillation,
    TokenBFactor,
    TokenPLDDT,
)
from chai_lab.data.features.generators.templates import (
    TemplateMaskGenerator,
    TemplateUnitVectorGenerator,
    TemplateDistogramGenerator,
    TemplateResTypeGenerator,
)
from chai_lab.data.features.generators.token_dist_restraint import TokenDistanceRestraint
from chai_lab.data.features.generators.token_pair_pocket_restraint import TokenPairPocketRestraint
from chai_lab.data.features.feature_factory import FeatureFactory
from chai_lab.data.features.feature_type import FeatureType

def copy_dict_of_tensors(d):
    if d is None:
        return None
    return {k: v.clone() if isinstance(v, Tensor) else v for k, v in d.items()}


@dataclass
class FoldingState:
    """Container for folding state"""
    feature_context: Optional[AllAtomFeatureContext] = None
    batch: Optional[Dict] = None
    model_size: Optional[int] = None
    init_reps: Optional[Dict[str, Tensor]] = None
    trunk_reps: Optional[Dict[str, Tensor]] = None
    result: Optional[Dict] = None
    chain_align_mask: Optional[List[bool]] = None
    chain_cyclic_mask: Optional[List[bool]] = None
    
    def copy(self) -> "FoldingState":
        """Deep copy state for saving/restoring"""
        
        return FoldingState(
            feature_context=self.feature_context,
            batch=copy_dict_of_tensors(self.batch),
            model_size=self.model_size,
            init_reps=copy_dict_of_tensors(self.init_reps),
            trunk_reps=copy_dict_of_tensors(self.trunk_reps),
            result=copy_dict_of_tensors(self.result),
            chain_align_mask=self.chain_align_mask.copy() if self.chain_align_mask else None,
            chain_cyclic_mask=self.chain_cyclic_mask.copy() if self.chain_cyclic_mask else None,
        )


class ChaiFolder:
    @staticmethod
    def und_self(x: torch.Tensor, pattern: str) -> torch.Tensor:
        return einops_einsum(x.float(), x.float(), pattern).bool()

    @dataclass
    class ChainOpts:
        use_esm: bool = False
        replace_x_with_mask: bool = False
        template_pdb: Optional[Path] = None
        template_chain_id: Optional[str] = None
        randomize_template_sequence: bool = False
        align: float = 1.0
        cyclic: bool = False

    def __init__(self,
                 device: str = "cuda:0",
                 allow_inter_chain_templates: bool = True,
                 pseudo_monomer: bool = False):
        self.device = torch.device(device)
        self.cpu_device = torch.device("cpu")
        self.tokenizer = AllAtomResidueTokenizer(RefConformerGenerator())
        self.models: Dict[str, Any] = {}
        self.allow_inter_chain_templates = allow_inter_chain_templates
        self.pseudo_monomer = pseudo_monomer
        self._load_all_models()
        self._build_feature_factory()
        self.state = FoldingState()

    def _load_all_models(self) -> None:
        model_keys = ["feature_embedding.pt", "token_embedder.pt", "trunk.pt",
                     "diffusion_module.pt", "confidence_head.pt", "bond_loss_input_proj.pt"]
        for key in model_keys:
            name = key.split(".")[0]
            self.models[name] = load_exported(key, self.device)

    def _build_feature_factory(self):
        """Build feature factory with inter-chain template flag"""
        feature_generators = dict(
            RelativeSequenceSeparation=RelativeSequenceSeparation(sep_bins=None),
            RelativeTokenSeparation=RelativeTokenSeparation(r_max=32),
            RelativeEntity=RelativeEntity(),
            RelativeChain=RelativeChain(),
            ResidueType=ResidueType(num_res_ty=32, key="token_residue_type"),
            ESMEmbeddings=ESMEmbeddings(),
            BlockedAtomPairDistogram=BlockedAtomPairDistogram(),
            InverseSquaredBlockedAtomPairDistances=BlockedAtomPairDistances(
                transform="inverse_squared",
                encoding_ty=EncodingType.IDENTITY,
            ),
            AtomRefPos=RefPos(),
            AtomRefCharge=Identity(
                key="inputs/atom_ref_charge",
                ty=FeatureType.ATOM,
                dim=1,
                can_mask=False,
            ),
            AtomRefMask=Identity(
                key="inputs/atom_ref_mask",
                ty=FeatureType.ATOM,
                dim=1,
                can_mask=False,
            ),
            AtomRefElement=AtomElementOneHot(max_atomic_num=128),
            AtomNameOneHot=AtomNameOneHot(),
            TemplateMask=TemplateMaskGenerator(
                allow_inter_chain=self.allow_inter_chain_templates
            ),
            TemplateUnitVector=TemplateUnitVectorGenerator(
                allow_inter_chain=self.allow_inter_chain_templates
            ),
            TemplateResType=TemplateResTypeGenerator(),
            TemplateDistogram=TemplateDistogramGenerator(
                allow_inter_chain=self.allow_inter_chain_templates
            ),
            TokenDistanceRestraint=TokenDistanceRestraint(
                include_probability=1.0,
                size=0.33,
                min_dist=6.0,
                max_dist=30.0,
                num_rbf_radii=6,
            ),
            DockingConstraintGenerator=DockingRestraintGenerator(
                include_probability=0.0,
                structure_dropout_prob=0.75,
                chain_dropout_prob=0.75,
            ),
            TokenPairPocketRestraint=TokenPairPocketRestraint(
                size=0.33,
                include_probability=1.0,
                min_dist=6.0,
                max_dist=20.0,
                coord_noise=0.0,
                num_rbf_radii=6,
            ),
            MSAProfile=MSAProfileGenerator(),
            MSADeletionMean=MSADeletionMeanGenerator(),
            IsDistillation=IsDistillation(),
            TokenBFactor=TokenBFactor(include_prob=0.0),
            TokenPLDDT=TokenPLDDT(include_prob=0.0),
            ChainIsCropped=ChainIsCropped(),
            MissingChainContact=MissingChainContact(contact_threshold=6.0),
            MSAOneHot=MSAFeatureGenerator(),
            MSAHasDeletion=MSAHasDeletionGenerator(),
            MSADeletionValue=MSADeletionValueGenerator(),
            IsPairedMSA=IsPairedMSAGenerator(),
            MSADataSource=MSADataSourceGenerator(),
        )
        self.feature_factory = FeatureFactory(feature_generators)

    def clear_state(self) -> None:
        self.state = FoldingState()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_state(self) -> FoldingState:
        """Save current state for later restoration"""
        return self.state.copy()

    def restore_state(self, state: FoldingState) -> None:
        """Restore a previously saved state"""
        self.state = state

    def _create_input(self, sequence: str, entity_type: str, name: str) -> Input:
        type_map = {
            "protein": EntityType.PROTEIN,
            "ligand": EntityType.LIGAND,
            "rna": EntityType.RNA,
            "dna": EntityType.DNA,
            "glycan": EntityType.MANUAL_GLYCAN,
        }
        return Input(sequence=sequence, entity_type=type_map[entity_type].value, entity_name=name)

    def _create_chains(self, sequences: List[Tuple[str, str, str, Dict]]) -> List[Chain]:
        """Create chains from sequence inputs"""
        inputs = [self._create_input(seq, etype, name) for seq, name, etype, _ in sequences]
        return load_chains_from_raw(inputs, identifier="chai-folder")

    def _parse_chain_opts(self, sequences: List[Tuple[str, str, str, Dict]]) -> List[ChainOpts]:
        """Parse chain options from sequence tuples"""
        opts_per_chain = []
        for *_unused, opt in sequences:
            opts_per_chain.append(ChaiFolder.ChainOpts(**opt) if opt else ChaiFolder.ChainOpts())
        return opts_per_chain

    def _apply_pseudo_monomer(self, structure_context: AllAtomStructureContext) -> None:
        """Apply pseudo_monomer mode: treat all chains as same entity/sym_id"""
        structure_context.token_entity_id = torch.zeros_like(structure_context.token_entity_id)
        structure_context.token_sym_id = torch.zeros_like(structure_context.token_sym_id)

    def _build_structure_context(
        self,
        chains: List[Chain],
        opts_per_chain: List[ChainOpts]
    ) -> AllAtomStructureContext:
        """Build merged structure context with optional pseudo_monomer"""
        structure_context = AllAtomStructureContext.merge([c.structure_context for c in chains])
        
        if self.pseudo_monomer:
            self._apply_pseudo_monomer(structure_context)
        
        structure_context.drop_glycan_leaving_atoms_inplace()
        return structure_context

    def _build_feature_contexts(
        self,
        chains: List[Chain],
        structure_context: AllAtomStructureContext,
        opts_per_chain: List[ChainOpts]
    ) -> AllAtomFeatureContext:
        """Build all feature contexts (MSA, templates, embeddings, restraints)"""
        n_tokens = structure_context.num_tokens
        
        msa_context, profile_msa_context = self._build_msa_context_per_chain(chains, opts_per_chain)
        template_context = self._build_template_context_per_chain(chains, opts_per_chain)
        embedding_context = self._build_embedding_context_per_chain(chains, opts_per_chain)
        restraint_context = RestraintContext.empty()
        
        return AllAtomFeatureContext(
            chains=chains,
            structure_context=structure_context,
            msa_context=msa_context,
            profile_msa_context=profile_msa_context,
            template_context=template_context,
            embedding_context=embedding_context,
            restraint_context=restraint_context,
        )

    def _collate_contexts(self, feature_context: AllAtomFeatureContext) -> Dict:
        """Collate feature context into batch"""
        collator = Collate(
            feature_factory=self.feature_factory,
            num_key_atoms=128,
            num_query_atoms=32
        )
        return move_data_to_device(collator([feature_context]), self.cpu_device)

    def _apply_cyclic_features(self):
        """Modify relative separation features for cyclic chains"""
        if not self.state.chain_cyclic_mask or not any(self.state.chain_cyclic_mask):
            return

        rel_sep_feat = self.state.batch["features"]["RelativeSequenceSeparation"]
        token_asym_id = self.state.batch["inputs"]["token_asym_id"][0]
        token_exists = self.state.batch["inputs"]["token_exists_mask"][0]

        center_bin = 33
        offsets = rel_sep_feat[0, :, :, 0] - center_bin

        for chain_idx, is_cyclic in enumerate(self.state.chain_cyclic_mask):
            if not is_cyclic:
                continue

            asym_id_value = chain_idx + 1
            chain_mask = (token_asym_id == asym_id_value) & token_exists
            indices = torch.where(chain_mask)[0]
            L = len(indices)

            if L <= 2:
                continue

            i, j = torch.meshgrid(indices, indices, indexing='ij')
            chain_offsets = offsets[i, j]

            abs_offset = torch.abs(chain_offsets)
            wrap_mask = abs_offset > (L / 2)
            cyclic_offsets = torch.where(
                wrap_mask,
                chain_offsets - torch.sign(chain_offsets) * L,
                chain_offsets
            )

            offsets[i, j] = cyclic_offsets

        rel_sep_feat[0, :, :, 0] = offsets + center_bin

    def prep_inputs(self, sequences: List[Tuple[str, str, str, Dict]]) -> None:
        """Main entry point: prepare inputs for folding"""
        with torch.no_grad():
            self.clear_state()
            
            # Parse inputs
            chains = self._create_chains(sequences)
            opts_per_chain = self._parse_chain_opts(sequences)
            
            # Store alignment/cyclic masks
            self.state.chain_align_mask = [opt.align for opt in opts_per_chain]
            self.state.chain_cyclic_mask = [opt.cyclic for opt in opts_per_chain]
            
            # Build contexts
            structure_context = self._build_structure_context(chains, opts_per_chain)
            n_tokens = structure_context.num_tokens
            
            # Determine model size
            possible = [s for s in AVAILABLE_MODEL_SIZES if s >= n_tokens]
            self.state.model_size = min(possible)
            
            # Build feature context and batch
            self.state.feature_context = self._build_feature_contexts(
                chains, structure_context, opts_per_chain
            )
            self.state.batch = self._collate_contexts(self.state.feature_context)
            
            # Apply cyclic modifications
            self._apply_cyclic_features()

    def _inject_bond_features(self, embedded: Dict[str, torch.Tensor]) -> None:
        """Inject bond features into token pair embeddings"""
        bond_ft_gen = TokenBondRestraint()
        bond_ft = bond_ft_gen.generate(batch=self.state.batch).data

        proj_out = self.models["bond_loss_input_proj"].forward(
            crop_size=self.state.model_size,
            move_to_device=self.device,
            return_on_cpu=False,
            input=bond_ft,
        )
        trunk_bond_feat, structure_bond_feat = proj_out.chunk(2, dim=-1)

        tp = embedded["TOKEN_PAIR"]
        pair_dtype = tp.dtype
        trunk_bond_feat = trunk_bond_feat.to(dtype=pair_dtype)
        structure_bond_feat = structure_bond_feat.to(dtype=pair_dtype)

        tp_trunk, tp_struct = tp.chunk(2, dim=-1)
        embedded["TOKEN_PAIR"] = torch.cat([
            tp_trunk + trunk_bond_feat,
            tp_struct + structure_bond_feat
        ], dim=-1)

    def get_embeddings(self) -> None:
        """Run feature embedding and token embedder"""
        if self.state.batch is None:
            raise RuntimeError("Must call prep_inputs() first")

        with torch.no_grad():
            inputs = move_data_to_device(self.state.batch["inputs"], self.device)
            features = move_data_to_device(self.state.batch["features"], self.device)

            embedded = self.models["feature_embedding"].forward(
                crop_size=self.state.model_size, **features
            )
            self._inject_bond_features(embedded)

            token_single_input_feats = embedded["TOKEN"]
            token_pair_trunk_feats, token_pair_struct_feats = embedded["TOKEN_PAIR"].chunk(2, -1)
            atom_single_trunk_feats, atom_single_struct_feats = embedded["ATOM"].chunk(2, -1)
            block_pair_trunk_feats, block_pair_struct_feats = embedded["ATOM_PAIR"].chunk(2, -1)

            token_single_init, token_single_struct, token_pair_init = (
                self.models["token_embedder"].forward(
                    crop_size=self.state.model_size,
                    token_single_input_feats=token_single_input_feats,
                    token_pair_input_feats=token_pair_trunk_feats,
                    atom_single_input_feats=atom_single_trunk_feats,
                    block_atom_pair_feat=block_pair_trunk_feats,
                    block_indices_h=inputs["block_atom_pair_q_idces"],
                    block_indices_w=inputs["block_atom_pair_kv_idces"],
                    atom_single_mask=inputs["atom_exists_mask"],
                    atom_token_indices=inputs["atom_token_index"],
                    block_atom_pair_mask=inputs["block_atom_pair_mask"],
                )
            )

            self.state.init_reps = move_data_to_device({
                'token_single_init': token_single_init,
                'token_single_struct': token_single_struct,
                'token_pair_init': token_pair_init,
                'token_pair_struct': token_pair_struct_feats,
                'atom_single_struct': atom_single_struct_feats,
                'atom_pair_struct': block_pair_struct_feats,
                'msa_feats': embedded["MSA"],
                'template_feats': embedded["TEMPLATES"],
            }, self.cpu_device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_trunk(self, num_trunk_recycles: int = 3, template_weight = None) -> None:
        """Run trunk"""
        if self.state.init_reps is None:
            self.get_embeddings()

        with torch.no_grad():
            inputs = move_data_to_device(self.state.batch["inputs"], self.device)

            token_single_init = self.state.init_reps['token_single_init'].to(self.device)
            token_pair_init = self.state.init_reps['token_pair_init'].to(self.device)

            template_input_masks = self.und_self(
                inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2"
            )
            token_pair_mask = self.und_self(
                inputs["token_exists_mask"], "b i, b j -> b i j"
            )

            s_single, s_pair = token_single_init, token_pair_init

            if template_weight is None:
                template_feats = self.state.init_reps['template_feats'].to(self.device)
            else:
                L = template_weight.shape[0]
                template_weight_expanded = template_weight[None, None, :, :, None]
                template_feats = self.state.init_reps['template_feats']
                template_feats[:, :, :L, :L, :] *= template_weight_expanded
                template_feats = template_feats.to(self.device)

            msa_feats = self.state.init_reps['msa_feats'].to(self.device)

            for n_recycle in range(num_trunk_recycles):
                s_single, s_pair = self.models["trunk"].forward(
                    crop_size=self.state.model_size,
                    token_single_trunk_initial_repr=token_single_init,
                    token_pair_trunk_initial_repr=token_pair_init,
                    token_single_trunk_repr=s_single,
                    token_pair_trunk_repr=s_pair,
                    msa_input_feats=msa_feats,
                    msa_mask=inputs["msa_mask"],
                    template_input_feats=template_feats,
                    template_input_masks=template_input_masks,
                    token_single_mask=inputs["token_exists_mask"],
                    token_pair_mask=token_pair_mask,
                )

            self.state.trunk_reps = {
                "token_single": s_single.to(self.cpu_device),
                "token_pair": s_pair.to(self.cpu_device),
                "token_single_struct": self.state.init_reps['token_single_struct'],
                "token_pair_struct": self.state.init_reps['token_pair_struct'],
                "atom_single_struct": self.state.init_reps['atom_single_struct'],
                "atom_pair_struct": self.state.init_reps['atom_pair_struct'],
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _weighted_rigid_align(self, true_coords, pred_coords, weights, mask):
        """Boltz's weighted alignment function"""
        with torch.no_grad():
            original_dtype = pred_coords.dtype
            true_coords_f32 = true_coords.to(torch.float32)
            pred_coords_f32 = pred_coords.to(torch.float32)
            weights_f32 = (mask * weights).unsqueeze(-1).to(torch.float32)

            weights_sum = weights_f32.sum(dim=1, keepdim=True).clamp(min=1e-8)
            true_centroid = (true_coords_f32 * weights_f32).sum(dim=1, keepdim=True) / weights_sum
            pred_centroid = (pred_coords_f32 * weights_f32).sum(dim=1, keepdim=True) / weights_sum

            true_centered = true_coords_f32 - true_centroid
            pred_centered = pred_coords_f32 - pred_centroid

            cov = pred_centered.transpose(-2, -1) @ (weights_f32 * true_centered)

            U, S, Vh = torch.linalg.svd(cov)
            V = Vh.mH

            R = U @ V.mH

            det = torch.det(R)

            if (det < 0).any():
                reflection_fix = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(R.shape[0], 1, 1)
                reflection_fix[:, -1, -1] = det.sign()
                rot_matrix = U @ reflection_fix @ V.mH
            else:
                rot_matrix = R

            aligned_coords = (pred_centered @ rot_matrix.mH) + true_centroid            
            return aligned_coords.to(original_dtype)

    def _compute_alignment_weights(
        self,
        atom_mask: Tensor,
        atom_token_indices: Tensor,
        token_entity_type: Tensor,
        token_asym_id: Tensor,
    ) -> Tensor:
        """Compute alignment weights for diffusion"""        
        weights = torch.ones_like(atom_mask, dtype=torch.float32)
        atom_asym_id = torch.gather(token_asym_id, 1, atom_token_indices.long())
        for chain_idx, align_weight in enumerate(self.state.chain_align_mask):
            asym_id_value = chain_idx + 1
            is_this_chain = (atom_asym_id == asym_id_value)
            weights[is_this_chain] = align_weight
        return weights

    def _run_diffusion(
        self,
        static_inputs: Dict[str, Tensor],
        token_entity_type: Tensor,
        token_asym_id: Tensor,
        sigmas: Tensor,
        use_alignment: bool = True,
        initial_coords: Optional[Tensor] = None,
    ) -> Tensor:
        """Run diffusion with optional alignment"""
        _, num_atoms = static_inputs["atom_single_mask"].shape

        if initial_coords is not None:
            atom_pos = initial_coords
            if atom_pos.dim() == 2:
                atom_pos = atom_pos.unsqueeze(0)
        else:
            atom_pos = sigmas[0] * torch.randn(1, num_atoms, 3, device=self.device)

        num_timesteps = len(sigmas) - 1
        gammas = torch.where(
            (sigmas >= DiffusionConfig.S_tmin) & (sigmas <= DiffusionConfig.S_tmax),
            min(DiffusionConfig.S_churn / num_timesteps, math.sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
        diff_mod = self.models["diffusion_module"]

        def _denoise(atom_pos: Tensor, sigma: Tensor) -> Tensor:
            atom_noised_coords = atom_pos.unsqueeze(1)
            noise_sigma = sigma.view(1, 1)
            return diff_mod.forward(
                atom_noised_coords=atom_noised_coords.float(),
                noise_sigma=noise_sigma.float(),
                crop_size=self.state.model_size,
                **static_inputs,
            )

        atom_mask = static_inputs["atom_single_mask"]

        # Compute alignment weights once
        weights = None
        if use_alignment:
            weights = self._compute_alignment_weights(
                atom_mask=atom_mask,
                atom_token_indices=static_inputs["atom_token_indices"],
                token_entity_type=token_entity_type,
                token_asym_id=token_asym_id,
            )

        for sigma_curr, sigma_next, gamma_curr in sigmas_and_gammas:
            atom_pos = center_random_augmentation(atom_pos, atom_single_mask=atom_mask)

            noise = DiffusionConfig.S_noise * torch.randn(atom_pos.shape, device=atom_pos.device)
            sigma_hat = sigma_curr + gamma_curr * sigma_curr
            atom_pos_noise = (sigma_hat**2 - sigma_curr**2).clamp_min(1e-6).sqrt()
            atom_pos_hat = atom_pos + noise * atom_pos_noise

            denoised_pos = _denoise(atom_pos=atom_pos_hat, sigma=sigma_hat)

            if use_alignment:
                atom_pos_hat = self._weighted_rigid_align(
                    true_coords=denoised_pos,
                    pred_coords=atom_pos_hat,
                    weights=weights,
                    mask=atom_mask,
                )

            d_i = (atom_pos_hat - denoised_pos) / sigma_hat
            atom_pos = atom_pos_hat + (sigma_next - sigma_hat) * d_i

            if sigma_next != 0 and DiffusionConfig.second_order:
                denoised_pos_2 = _denoise(atom_pos=atom_pos, sigma=sigma_next)

                if use_alignment:
                    atom_pos = self._weighted_rigid_align(
                        true_coords=denoised_pos_2,
                        pred_coords=atom_pos,
                        weights=weights,
                        mask=atom_mask,
                    )
                    d_i_prime = (atom_pos - denoised_pos_2) / sigma_next
                else:
                    d_i_prime = (atom_pos - denoised_pos_2) / sigma_next

                atom_pos = atom_pos + (sigma_next - sigma_hat) * ((d_i_prime + d_i) / 2)

        return atom_pos

    def sample(self,
              num_diffn_timesteps: int = 200,
              num_diffn_samples: int = 1,
              use_alignment: bool = True,
              refine_from_coords: Optional[Tensor] = None,
              refine_from_step: Optional[int] = None) -> None:
        """Run diffusion sampling and optionally score the structure"""
        if self.state.trunk_reps is None:
            self.run_trunk()

        with torch.no_grad():
            inputs = move_data_to_device(self.state.batch["inputs"], self.device)
            reps_gpu = move_data_to_device(self.state.trunk_reps, self.device)

            static_diffusion_inputs = dict(
                token_single_trunk_repr=reps_gpu["token_single"].float(),
                token_pair_trunk_repr=reps_gpu["token_pair"].float(),
                token_single_initial_repr=reps_gpu["token_single_struct"].float(),
                token_pair_initial_repr=reps_gpu["token_pair_struct"].float(),
                atom_single_input_feats=reps_gpu["atom_single_struct"].float(),
                atom_block_pair_input_feats=reps_gpu["atom_pair_struct"].float(),
                atom_single_mask=inputs["atom_exists_mask"],
                atom_block_pair_mask=inputs["block_atom_pair_mask"],
                token_single_mask=inputs["token_exists_mask"],
                block_indices_h=inputs["block_atom_pair_q_idces"],
                block_indices_w=inputs["block_atom_pair_kv_idces"],
                atom_token_indices=inputs["atom_token_index"],
            )

            schedule = InferenceNoiseSchedule(
                s_max=DiffusionConfig.S_tmax,
                s_min=4e-4,
                p=7.0,
                sigma_data=DiffusionConfig.sigma_data,
            )
            full_sigmas = schedule.get_schedule(
                device=self.device, num_timesteps=num_diffn_timesteps
            )
            best_result = None
            for s in range(num_diffn_samples):
                if refine_from_coords is not None:
                    atom_mask = inputs["atom_exists_mask"].squeeze(0)
                    n_atoms_padded = atom_mask.shape[0]
    
                    padded_coords = torch.zeros(n_atoms_padded, 3, device=self.device)
                    padded_coords[atom_mask] = refine_from_coords.to(self.device)
    
                    starting_sigma = full_sigmas[refine_from_step]
                    noise = torch.randn_like(padded_coords)
                    padded_coords = padded_coords + starting_sigma * noise
    
                    sigmas = full_sigmas[refine_from_step:]
                else:
                    _, num_atoms = static_diffusion_inputs["atom_single_mask"].shape
                    padded_coords = full_sigmas[0] * torch.randn(num_atoms, 3, device=self.device)
                    sigmas = full_sigmas
    
                padded_coords = self._run_diffusion(
                    static_inputs=static_diffusion_inputs,
                    token_entity_type=inputs["token_entity_type"],
                    token_asym_id=inputs["token_asym_id"],
                    sigmas=sigmas,
                    use_alignment=use_alignment,
                    initial_coords=padded_coords,
                )
                result = self._score(padded_coords, reps_gpu)
                if best_result is None or result["ranking_score"] > best_result["ranking_score"]:
                    best_result = result
                    atom_mask = inputs["atom_exists_mask"].squeeze(0)
                    best_result['coords'] = padded_coords.squeeze(0)[atom_mask].cpu()
                

            self.state.result = best_result

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _score(self, padded_atom_pos: Tensor, trunk_reps_gpu: Dict) -> Dict:
        """Score a single padded structure using confidence head"""
        inputs = move_data_to_device(self.state.batch["inputs"], self.device)
        conf_head = self.models["confidence_head"]

        pae_logits, pde_logits, plddt_logits = conf_head.forward(
            crop_size=self.state.model_size,
            token_single_input_repr=trunk_reps_gpu["token_single_struct"],
            token_single_trunk_repr=trunk_reps_gpu["token_single"],
            token_pair_trunk_repr=trunk_reps_gpu["token_pair"],
            token_single_mask=inputs["token_exists_mask"],
            atom_single_mask=inputs["atom_exists_mask"],
            atom_coords=padded_atom_pos,
            token_reference_atom_index=inputs["token_ref_atom_index"],
            atom_token_index=inputs["atom_token_index"],
            atom_within_token_index=inputs["atom_within_token_index"],
        )

        _, valid_frames_mask = get_frames_and_mask(
            padded_atom_pos,
            inputs["token_asym_id"],
            inputs["token_residue_index"],
            inputs["token_backbone_frame_mask"],
            inputs["token_centre_atom_index"],
            inputs["token_exists_mask"],
            inputs["atom_exists_mask"],
            inputs["token_backbone_frame_index"],
            inputs["atom_token_index"],
        )

        ranking_outputs = rank(
            padded_atom_pos,
            atom_mask=inputs["atom_exists_mask"],
            atom_token_index=inputs["atom_token_index"],
            token_exists_mask=inputs["token_exists_mask"],
            token_asym_id=inputs["token_asym_id"],
            token_entity_type=inputs["token_entity_type"],
            token_valid_frames_mask=valid_frames_mask,
            lddt_logits=plddt_logits,
            lddt_bin_centers=_bin_centers(0, 1, plddt_logits.shape[-1]).to(plddt_logits.device),
            pae_logits=pae_logits,
            pae_bin_centers=_bin_centers(0.0, 32.0, 64).to(pae_logits.device),
        )

        atom_mask = inputs["atom_exists_mask"].squeeze(0).cpu()
        token_mask = inputs["token_exists_mask"].squeeze(0).cpu()

        def softmax_einsum_and_cpu(logits: Tensor, bin_mean: Tensor, pattern: str) -> Tensor:
            res = einops_einsum(logits.float().softmax(dim=-1), bin_mean.to(logits.device), pattern)
            return res.to(device="cpu")

        pae_scores = softmax_einsum_and_cpu(
            pae_logits[:, token_mask, :, :][:, :, token_mask, :],
            _bin_centers(0.0, 32.0, 64),
            "b n1 n2 d, d -> b n1 n2",
        )
        plddt_scores_atom = softmax_einsum_and_cpu(
            plddt_logits,
            _bin_centers(0, 1, plddt_logits.shape[-1]),
            "b a d, d -> b a",
        )

        pde = pde_logits.cpu()[:,token_mask,:,:][:,:,token_mask,:].float().softmax(dim=-1)

        [indices] = inputs["atom_token_index"].cpu()
        def avg_per_token_1d(x):
            n = torch.bincount(indices[atom_mask], weights=x[atom_mask])
            d = torch.bincount(indices[atom_mask]).clamp(min=1)
            return n / d
        plddt_scores = avg_per_token_1d(plddt_scores_atom.squeeze())
        return dict(
            plddt_per_atom=plddt_scores_atom.squeeze()[atom_mask].cpu(),
            plddt=plddt_scores,
            pae=pae_scores.squeeze(),
            pde=pde.squeeze(),
            ptm=ranking_outputs.ptm_scores.complex_ptm.cpu(),
            iptm=ranking_outputs.ptm_scores.interface_ptm.cpu(),
            per_chain_ptm=ranking_outputs.ptm_scores.per_chain_ptm.cpu(),
            per_chain_pair_iptm=ranking_outputs.ptm_scores.per_chain_pair_iptm.cpu(),
            total_clashes=ranking_outputs.clash_scores.total_clashes.cpu(),
            total_inter_chain_clashes=ranking_outputs.clash_scores.total_inter_chain_clashes.cpu(),
            chain_chain_clashes=ranking_outputs.clash_scores.chain_chain_clashes.cpu(),
            has_inter_chain_clashes=ranking_outputs.clash_scores.has_inter_chain_clashes.cpu(),
            complex_plddt=ranking_outputs.plddt_scores.complex_plddt.cpu(),
            per_chain_plddt=ranking_outputs.plddt_scores.per_chain_plddt.cpu(),
            ranking_score=ranking_outputs.aggregate_score.cpu().item(),
        )

    def save(self, filename: Union[str, Path], use_entity_names: bool = False) -> Path:
        """Save the current structure"""
        if self.state.result is None:
            raise RuntimeError("No structure has been sampled yet")

        filename = Path(filename).with_suffix(".cif")
        return self._save_structure(self.state.result, filename, use_entity_names_as_chain_ids=use_entity_names)

    def plot(self, width: int = 640, height: int = 480,
            style: str = "cartoon", color_by: str = "none"):
        """Plot the current structure"""
        if self.state.result is None:
            raise RuntimeError("No structure has been sampled yet")

        res = self.state.result

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        path = self._save_structure(
            result_item=res,
            out_path=tmp_path,
            use_entity_names_as_chain_ids=True,
        )
        cif_text = path.read_text()

        view = py3Dmol.view(width=width, height=height)
        view.addModel(cif_text, "cif")

        color_by = (color_by or "none").lower()
        if color_by == "plddt":
            view.setStyle({"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]}, {
                style: {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "roygb",
                        "min": 50,
                        "max": 90,
                    }
                }
            })
        elif color_by == "chain":
            view.setStyle({"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
                        {style: {"colorscheme": "chain"}})
        else:
            view.setStyle({"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
                        {style: {}})

        view.setStyle({"or": [{"resn": ["UNL", "LIG"]}, {"hetflag": True}]},
                      {"stick": {"radius": 0.2}})

        view.zoomTo()
        os.remove(path)
        return view.show()

    def _save_structure(self, result_item: dict, out_path: Path,
                        use_entity_names_as_chain_ids: bool = True) -> Path:
        coords = result_item["coords"]
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        elif coords.dim() == 3 and coords.shape[0] != 1:
            coords = coords[:1]
        coords = coords.to(torch.float32)

        plddt = result_item.get("plddt_per_atom", None)
        bfactors = None
        if plddt is not None:
            if plddt.dim() != 1:
                plddt = plddt.view(-1)
            bfactors = (plddt * 100.0).unsqueeze(0).to(torch.float32)

        output_inputs = move_data_to_device(self.state.batch["inputs"], self.cpu_device)
        ctx = pdb_context_from_batch(output_inputs)
        asym_ids = sorted(ctx.asym_id2entity_type.keys())

        if use_entity_names_as_chain_ids:
            desired = [ch.entity_data.entity_name for ch in self.state.feature_context.chains]
            mapping = {asym_id: (desired[i] if i < len(desired) else get_chain_letter(i + 1))
                       for i, asym_id in enumerate(asym_ids)}
        else:
            mapping = {asym_id: get_chain_letter(i + 1) for i, asym_id in enumerate(asym_ids)}

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_cif(
            coords=coords.cpu(),
            output_batch=output_inputs,
            write_path=out_path,
            asym_entity_names=mapping,
            bfactors=None if bfactors is None else bfactors.cpu(),
        )
        return out_path

    def _build_embedding_context_per_chain(self, chains: List[Chain],
                                         opts_per_chain: List["ChaiFolder.ChainOpts"]) -> EmbeddingContext:
        per_chain_embeds: List[Tensor] = []
        for ch, opt in zip(chains, opts_per_chain, strict=True):
            if opt.use_esm and ch.entity_data.entity_type == EntityType.PROTEIN:
                if opt.replace_x_with_mask:
                    modified_full_sequence = [residue if residue != "X" else "<mask>"
                                            for residue in ch.entity_data.full_sequence]
                    modified_entity_data = replace(ch.entity_data, full_sequence=modified_full_sequence)
                    temp_chain = Chain(entity_data=modified_entity_data, structure_context=ch.structure_context)
                    ctx_one = get_esm_embedding_context([temp_chain], device=self.device)
                else:
                    ctx_one = get_esm_embedding_context([ch], device=self.device)
                per_chain_embeds.append(ctx_one.esm_embeddings)
            else:
                zeros = EmbeddingContext.empty(n_tokens=ch.structure_context.num_tokens).esm_embeddings
                per_chain_embeds.append(zeros)

        merged = torch.cat(per_chain_embeds, dim=0)
        return EmbeddingContext(esm_embeddings=merged)

    def _build_msa_context_per_chain(self, chains: List[Chain],
                                     opts_per_chain: List["ChaiFolder.ChainOpts"]) -> Tuple[MSAContext, MSAContext]:
        per_chain_msa: List[MSAContext] = []

        for ch, opt in zip(chains, opts_per_chain, strict=True):
            sc = ch.structure_context
            n = sc.num_tokens
            msa = MSAContext.create_empty(n_tokens=n, depth=1)
            per_chain_msa.append(msa)

        msa_context = MSAContext.cat(per_chain_msa, dim=1)
        profile_msa_context = MSAContext.cat(per_chain_msa, dim=1)
        return msa_context, profile_msa_context

    def _template_context_from_structure_context(self,
                                                 template_sc: AllAtomStructureContext,
                                                 randomize_template_sequence: bool = False) -> TemplateContext:
        coords, coord_mask = get_token_reference_atom_positions_and_mask(
            atom_pos=template_sc.atom_gt_coords.unsqueeze(0),
            atom_mask=template_sc.atom_exists_mask.unsqueeze(0),
            token_reference_atom_index=template_sc.token_ref_atom_index.unsqueeze(0),
            token_exists_mask=template_sc.token_exists_mask.unsqueeze(0),
        )
        coords = coords.squeeze(0)
        coord_mask = coord_mask.squeeze(0)

        distances = cdist(coords.unsqueeze(0)).squeeze(0)
        mask_2d = coord_mask[:, None] & coord_mask[None, :]
        distances = distances.masked_fill(~mask_2d, 100.0)

        bb_idx = template_sc.token_backbone_frame_index
        n_idx, ca_idx, c_idx = [t.squeeze(-1) for t in torch.split(bb_idx, 1, dim=-1)]

        def _g(idx):
            pos, _ = get_token_reference_atom_positions_and_mask(
                atom_pos=template_sc.atom_gt_coords.unsqueeze(0),
                atom_mask=template_sc.atom_exists_mask.unsqueeze(0),
                token_reference_atom_index=idx.unsqueeze(0),
                token_exists_mask=template_sc.token_backbone_frame_mask.unsqueeze(0),
            )
            return pos.squeeze(0)

        n_pos = _g(n_idx)
        ca_pos = _g(ca_idx)
        c_pos = _g(c_idx)

        eps = 1e-12
        rigids = Rigid.make_transform_from_reference(n_xyz=n_pos, ca_xyz=ca_pos, c_xyz=c_pos, eps=eps)
        points = rigids.get_trans()
        rigid_vec = rigids[..., None].invert_apply(points)

        bb_mask2d = template_sc.token_backbone_frame_mask.squeeze(0).bool()
        bb_mask2d = bb_mask2d[:, None] & bb_mask2d[None, :]
        invd = torch.rsqrt(eps + (rigid_vec ** 2).sum(-1)) * bb_mask2d
        unit_vec = rigid_vec * invd[..., None]

        template_restype = template_sc.token_residue_type
        if randomize_template_sequence:
            template_restype = torch.randint(0,20,template_restype.shape)

        return TemplateContext(
            template_restype=template_restype.squeeze(0).to(torch.int32).unsqueeze(0),
            template_pseudo_beta_mask=coord_mask.bool().unsqueeze(0),
            template_backbone_frame_mask=template_sc.token_backbone_frame_mask.squeeze(0).bool().unsqueeze(0),
            template_distances=distances.unsqueeze(0),
            template_unit_vector=unit_vec.unsqueeze(0),
        )

    def _clean_mse_residues(self, structure: gemmi.Structure) -> None:
        """
        Replace MSE (selenomethionine) with MET (methionine) in-place.
        MSE causes tokenization issues - gets atom-level tokenization without backbone frames.
        """
        mse_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.name.strip().upper() == 'MSE':
                        residue.name = 'MET'
                        mse_count += 1
                        for atom in residue:
                            if atom.name.strip().upper() == 'SE':
                                atom.name = 'SD'
                                atom.element = gemmi.Element('S')

        if mse_count > 0:
            print(f"Cleaned {mse_count} MSE residues → MET in template")

    def _parse_template_chains(
        self,
        template_pdb: Path,
        template_chain_ids: List[str]
    ) -> List[AllAtomStructureContext]:
        """Parse template chains, bypassing entity system"""
        from chai_lab.data.parsing.structure.residue import get_residues
        from chai_lab.data.parsing.structure.all_atom_entity_data import AllAtomEntityData

        structure = gemmi.read_structure(str(template_pdb))
        self._clean_mse_residues(structure)

        available_chains: Dict[str, gemmi.Chain] = {}
        for model in structure:
            for chain_in_model in model:
                if chain_in_model.name not in available_chains:
                    available_chains[chain_in_model.name] = chain_in_model

        template_structure_contexts = []
        for chain_id in template_chain_ids:
            gemmi_chain = available_chains.get(chain_id)
            assert gemmi_chain is not None, \
                f"Template chain '{chain_id}' not found in {template_pdb}."

            subchains = list(gemmi_chain.subchains())
            assert len(subchains) > 0, \
                f"Template chain '{chain_id}' in {template_pdb} has no subchains."
            subchain = subchains[0]

            residues_list = list(subchain.first_conformer())
            for idx, res in enumerate(residues_list, start=1):
                res.label_seq = idx

            full_sequence = [res.name for res in residues_list]

            residues = get_residues(
                subchain=subchain,
                full_sequence=full_sequence,
                entity_type=EntityType.PROTEIN,
            )

            entity_data = AllAtomEntityData(
                residues=residues,
                full_sequence=full_sequence,
                entity_name=f"template_{chain_id}",
                entity_id=0,
                entity_type=EntityType.PROTEIN,
                source_pdb_chain_id=chain_id,
                subchain_id=subchain.subchain_id(),
                resolution=structure.resolution,
                release_datetime=None,
                pdb_id=structure.name,
                method="",
            )

            tpl_sc = self.tokenizer.tokenize_entity(entity_data)
            assert tpl_sc is not None, \
                f"Failed to tokenize template chain {chain_id} from {template_pdb}."

            template_structure_contexts.append(tpl_sc)

        return template_structure_contexts

    def _group_consecutive_templates(
        self,
        chains: List[Chain],
        opts: List["ChaiFolder.ChainOpts"]
    ) -> List[Dict]:
        """Group consecutive chains that can share template features"""
        groups = []
        i = 0

        while i < len(chains):
            is_templatable_protein = (
                chains[i].entity_data.entity_type == EntityType.PROTEIN and
                opts[i].template_pdb is not None and
                opts[i].template_chain_id is not None
            )

            if is_templatable_protein:
                start = i
                current_pdb = opts[i].template_pdb

                i += 1
                while (i < len(chains) and
                       chains[i].entity_data.entity_type == EntityType.PROTEIN and
                       opts[i].template_pdb == current_pdb and
                       opts[i].template_chain_id is not None):
                    i += 1

                groups.append({
                    'type': 'multi_chain' if i - start > 1 else 'single',
                    'start': start,
                    'end': i
                })
            else:
                groups.append({
                    'type': 'single',
                    'start': i,
                    'end': i + 1
                })
                i += 1

        return groups

    def _build_template_for_group(
        self,
        chains: List[Chain],
        opts: List[ChainOpts]
    ) -> TemplateContext:
        """Build template for one or more chains (unified handler)"""
        
        # Check if group has any valid templates
        has_template = any(
            ch.entity_data.entity_type == EntityType.PROTEIN and
            opt.template_pdb is not None and
            opt.template_chain_id is not None
            for ch, opt in zip(chains, opts)
        )
        
        if not has_template:
            # Empty template for non-protein or protein without template
            total_tokens = sum(ch.structure_context.num_tokens for ch in chains)
            return TemplateContext.empty(n_templates=1, n_tokens=total_tokens)
        
        # Parse template chains (grouping guarantees same template_pdb)
        template_pdb = next(opt.template_pdb for opt in opts if opt.template_pdb)
        template_chain_ids = [opt.template_chain_id for opt in opts if opt.template_chain_id]
        template_scs = self._parse_template_chains(template_pdb, template_chain_ids)
        
        # Merge if multiple
        merged_sc = (AllAtomStructureContext.merge(template_scs) 
                     if len(template_scs) > 1 else template_scs[0])
        
        # Build template context
        tpl_context = self._template_context_from_structure_context(
            merged_sc,
            randomize_template_sequence=opts[0].randomize_template_sequence
        )
        
        # Map to query tokens with offsets
        all_indices, offset = [], 0
        for ch, tpl_sc in zip(chains, template_scs):
            _, inv = torch.unique(ch.structure_context.token_residue_index, return_inverse=True)
            all_indices.append(inv + offset)
            offset += tpl_sc.num_tokens
        
        return tpl_context.index_select(torch.cat(all_indices))

    def _build_template_context_per_chain(
        self,
        chains: List[Chain],
        opts: List["ChaiFolder.ChainOpts"]
    ) -> TemplateContext:
        """Build template contexts with automatic multi-chain grouping"""
        groups = self._group_consecutive_templates(chains, opts)
        return TemplateContext.merge([
            self._build_template_for_group(
                chains[g['start']:g['end']], 
                opts[g['start']:g['end']]
            ) for g in groups
        ])
