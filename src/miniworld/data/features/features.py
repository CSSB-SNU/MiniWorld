from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from team_gm import BaseBatch, typecheck


@typecheck
@dataclass()
class SequenceFeatures(BaseBatch):
    """Sequence features."""

    token_type: Int[torch.Tensor, "B L_res"]


@typecheck
@dataclass()
class StructureFeatures(BaseBatch):
    """Structure features."""

    atom_pos: Float[torch.Tensor, "B L_atom 3"]
    atom_pos_mask: Float[torch.Tensor, "B L_atom"]
    atom_mask: Bool[torch.Tensor, "B L_atom"]
    token_bond: Int[torch.Tensor, "B n_token_bond 2"]
    token_mask: Bool[torch.Tensor, "B L_res"]
    atom_bond: Int[torch.Tensor, "B n_atom_bond 6"]


@typecheck
@dataclass()
class MultiStateStructureFeatures(BaseBatch):
    """Structure features."""

    atom_pos: Float[torch.Tensor, "B N_str L_atom 3"]
    atom_pos_mask: Float[torch.Tensor, "B N_str L_atom"]
    atom_mask: Bool[torch.Tensor, "B L_atom"]
    token_bond: Int[torch.Tensor, "B n_token_bond 3"]
    token_mask: Bool[torch.Tensor, "B L_res"]
    atom_bond: Int[torch.Tensor, "B n_atom_bond 6"]


@typecheck
@dataclass()
class ReferenceFeatures(BaseBatch):
    """Reference features."""

    pos: Float[torch.Tensor, "B L_atom 3"]
    mask: Float[torch.Tensor, "B L_atom"]
    element: Float[torch.Tensor, "B L_atom"]
    charge: Float[torch.Tensor, "B L_atom"]
    space_uid: Int[torch.Tensor, "B L_atom"]


@typecheck
@dataclass()
class MSAFeatures(BaseBatch):
    """MSA features."""

    aligned_sequences: Int[torch.Tensor, "B N_msa L_token"]
    mask: Bool[torch.Tensor, "B N_msa"]
    has_deletion: Int[torch.Tensor, "B N_msa L_token"]
    deletion_value: Float[torch.Tensor, "B N_msa L_token"]
    profile: Float[torch.Tensor, "B L_token d_profile"]
    deletion_mean: Float[torch.Tensor, "B L_token"]


@typecheck
@dataclass()
class TemplateFeatures(BaseBatch):
    """Template features."""

    mask: Bool[torch.Tensor, "B N_temp"]
    ids: Int[torch.Tensor, "B N_temp L_res"]
    res_type: Int[torch.Tensor, "B N_temp L_res"]
    cb_xyz: Float[torch.Tensor, "B N_temp L_res 3"]
    cb_mask: Bool[torch.Tensor, "B N_temp L_res"]
    bb_xyz: Float[torch.Tensor, "B N_temp L_res 3 3"]
    bb_mask: Bool[torch.Tensor, "B N_temp L_res 3"]


@typecheck
@dataclass()
class ChainFeatures(BaseBatch):
    """Chain features."""

    entity_type: Int[torch.Tensor, "B L_chain"]


@typecheck
@dataclass()
class SchemeFeatures(BaseBatch):
    """Scheme features."""

    token_residue_idx: Int[torch.Tensor, "B L_token"]
    token_idx: Int[torch.Tensor, "B L_token"]
    token_asym_id: Int[torch.Tensor, "B L_token"]
    token_entity_id: Int[torch.Tensor, "B L_token"]
    token_sym_id: Int[torch.Tensor, "B L_token"]
    atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"]
