from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from team_gm import BaseBatch, typecheck

from .features import (
    ChainFeatures,
    MSAFeatures,
    ReferenceFeatures,
    SequenceFeatures,
    StructureFeatures,
)

if TYPE_CHECKING:
    from jaxtyping import Int


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
    def msa_depth(self) -> int:
        """Return the MSA depth."""
        return self.msa.aligned_sequences.shape[1]

    @property
    def msa_number(self) -> int:
        """Return the number of sampled MSAs."""
        return self.msa.aligned_sequences.shape[0]

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

    @staticmethod
    def empty(n_msa: int, msa_depth: int, n_tokens: int, n_atoms: int) -> Batch:
        """Create an empty batch with the given dimensions (batch_size=1)."""
        return Batch(
            name=[""],
            heteros=[None],
            atom_ids=[None],
            chem_comp_ids=[None],
            sequence=SequenceFeatures(
                token_type=torch.zeros((1, n_tokens), dtype=torch.long),
            ),
            structure=StructureFeatures(
                atom_pos=torch.zeros((1, n_atoms, 3), dtype=torch.float),
                atom_pos_mask=torch.zeros((1, n_atoms), dtype=torch.float),
                atom_mask=torch.zeros((1, n_atoms), dtype=torch.bool),
                token_bond=torch.zeros((1, 0, 2), dtype=torch.long),
                token_mask=torch.zeros((1, n_tokens), dtype=torch.bool),
                atom_bond=torch.zeros((1, 0, 6), dtype=torch.long),
            ),
            reference=ReferenceFeatures(
                pos=torch.zeros((1, n_atoms, 3), dtype=torch.float),
                mask=torch.zeros((1, n_atoms), dtype=torch.float),
                element=torch.zeros((1, n_atoms), dtype=torch.float),
                charge=torch.zeros((1, n_atoms), dtype=torch.float),
                space_uid=torch.zeros((1, n_atoms), dtype=torch.long),
            ),
            scheme=SchemeFeatures(
                token_residue_idx=torch.zeros((1, n_tokens), dtype=torch.long),
                token_idx=torch.zeros((1, n_tokens), dtype=torch.long),
                token_asym_id=torch.zeros((1, n_tokens), dtype=torch.long),
                token_entity_id=torch.zeros((1, n_tokens), dtype=torch.long),
                token_sym_id=torch.zeros((1, n_tokens), dtype=torch.long),
                atom_to_token_idx_map=torch.zeros((1, n_atoms), dtype=torch.long),
                edge_index=torch.zeros((1, 2), dtype=torch.long),
            ),
            msa=MSAFeatures(
                aligned_sequences=torch.zeros(
                    (1, n_msa, msa_depth, n_tokens),
                    dtype=torch.long,
                ),
                msa_mask=torch.zeros((1, n_msa, msa_depth, n_tokens), dtype=torch.bool),
                has_deletion=torch.zeros(
                    (1, n_msa, msa_depth, n_tokens),
                    dtype=torch.bool,
                ),
                deletion_value=torch.zeros(
                    (1, n_msa, msa_depth, n_tokens),
                    dtype=torch.float,
                ),
                profile=torch.zeros((1, n_tokens, 20), dtype=torch.float),
                deletion_mean=torch.zeros((1, n_tokens), dtype=torch.float),
            ),
            chain=ChainFeatures(
                entity_type=torch.zeros((1, 1), dtype=torch.long),
                contact_edges=torch.zeros((1, 1, 2), dtype=torch.long),
            ),
        )
