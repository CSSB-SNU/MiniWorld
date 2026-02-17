from dataclasses import dataclass

import torch
from jaxtyping import Int
from team_gm import BaseBatch, typecheck

from .features import (
    MSAFeatures,
    MultiStateStructureFeatures,
    ReferenceFeatures,
    SequenceFeatures,
)


@typecheck
@dataclass
class SchemeFeatures(BaseBatch):
    """Scheme features."""

    residue_idx: Int[torch.Tensor, "B L_res"]
    residue_idx_mono: Int[torch.Tensor, "B L_res"]
    residue_asym_id: Int[torch.Tensor, "B L_res"]
    residue_entity_id: Int[torch.Tensor, "B L_res"]
    residue_sym_id: Int[torch.Tensor, "B L_res"]
    atom_to_residue_idx_map: Int[torch.Tensor, "B L_atom"]

@dataclass(kw_only=True)
class Batch(BaseBatch):
    """Batch of features."""

    name: list  # (...)

    # additional info for making cif files
    heteros: list
    atom_ids: list
    chem_comp_ids: list

    sequence: SequenceFeatures
    structure: MultiStateStructureFeatures
    reference: ReferenceFeatures
    scheme: SchemeFeatures
    msa: MSAFeatures

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
    def residue_length(self) -> int:
        """Return the residue length."""
        if len(self.reference.pos.shape) == 2:
            return self.scheme.residue_idx.shape[0]
        if len(self.reference.pos.shape) == 3:
            return self.scheme.residue_idx.shape[1]
        msg = "Cannot infer residue length from reference positions."
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
