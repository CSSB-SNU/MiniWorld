from dataclasses import dataclass

import torch
from jaxtyping import Int
from team_gm import BaseBatch, typecheck

from .features import (
    ChainFeatures,
    MSAFeatures,
    ReferenceFeatures,
    SequenceFeatures,
    StructureFeatures,
)


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
