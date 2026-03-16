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
class SchemeFeatures(BaseBatch):
    """Scheme features."""

    token_residue_idx: Int[torch.Tensor, "B L_token"]
    token_idx: Int[torch.Tensor, "B L_token"]
    token_asym_id: Int[torch.Tensor, "B L_token"]
    token_entity_id: Int[torch.Tensor, "B L_token"]
    token_sym_id: Int[torch.Tensor, "B L_token"]
    atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"]
    edge_index: Int[torch.Tensor, "B E"]


@typecheck
@dataclass()
class MSAFeatures(BaseBatch):
    """MSA features."""

    aligned_sequences: Int[torch.Tensor, "B N_sampled N_msa L_token"]
    msa_mask: Bool[torch.Tensor, "B N_sampled N_msa L_token"]
    has_deletion: Int[torch.Tensor, "B N_sampled N_msa L_token"]
    deletion_value: Float[torch.Tensor, "B N_sampled N_msa L_token"]
    profile: Float[torch.Tensor, "B L_token d_profile"]
    deletion_mean: Float[torch.Tensor, "B L_token"]


@typecheck
@dataclass()
class ChainFeatures(BaseBatch):
    """Chain features."""

    entity_type: Int[torch.Tensor, "B L_chain"]
    contact_edges: Int[torch.Tensor, "B N_contact 2"]


@dataclass(kw_only=True)
class Batch(BaseBatch):
    """Batch of features."""

    name: list

    # additional info for making cif files
    heteros: list
    atom_ids: list
    chem_comp_ids: list

    sequence: SequenceFeatures
    structure: StructureFeatures
    reference: ReferenceFeatures
    scheme: SchemeFeatures
    msa: MSAFeatures
    chain: ChainFeatures

    @property
    def shape(self) -> torch.Size:
        """Return the shape of atom mask."""
        return self.structure.atom_mask.shape

    @property
    def device(self) -> torch.device:
        """Return the device."""
        return self.structure.atom_mask.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype."""
        return self.structure.atom_pos.dtype

    @property
    def token_length(self) -> int:
        """Return the token length."""
        if len(self.reference.pos.shape) == 2:
            return self.scheme.token_idx.shape[0]
        if len(self.reference.pos.shape) == 3:
            return self.scheme.token_idx.shape[1]
        msg = "Cannot infer token length from reference positions."
        raise ValueError(msg)

    @property
    def atom_length(self) -> int:
        """Return the atom length."""
        if len(self.reference.pos.shape) == 2:
            return self.reference.pos.shape[0]
        if len(self.reference.pos.shape) == 3:
            return self.reference.pos.shape[1]
        msg = "Cannot infer atom length from reference positions."
        raise ValueError(msg)
