from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from team_gm import BaseBatch, typecheck


@typecheck
@dataclass(frozen=True)
class SequenceFeatures(BaseBatch):
    """Sequence features."""

    residue_type: Int[torch.Tensor, "B L_res"]


@typecheck
@dataclass(frozen=True)
class StructureFeatures(BaseBatch):
    """Structure features."""

    atom_pos: Float[torch.Tensor, "B L_atom 3"]
    atom_pos_mask: Float[torch.Tensor, "B L_atom"]
    atom_mask: Bool[torch.Tensor, "B L_atom"]
    residue_bond: Int[torch.Tensor, "B n_residue_bond 3"]
    residue_mask: Bool[torch.Tensor, "B L_res"]
    atom_bond: Int[torch.Tensor, "B n_atom_bond 6"]


@typecheck
@dataclass(frozen=True)
class ReferenceFeatures(BaseBatch):
    """Reference features."""

    pos: Float[torch.Tensor, "B L_atom 3"]
    mask: Float[torch.Tensor, "B L_atom"]
    element: Float[torch.Tensor, "B L_atom"]
    charge: Float[torch.Tensor, "B L_atom"]
    space_uid: Int[torch.Tensor, "B L_atom"]


@typecheck
@dataclass(frozen=True)
class SchemeFeatures(BaseBatch):
    """Scheme features."""

    crop_indices: Int[torch.Tensor, "B L_res"]
    residue_idx: Int[torch.Tensor, "B L_res"]
    residue_idx_mono: Int[torch.Tensor, "B L_res"]
    residue_asym_id: Int[torch.Tensor, "B L_res"]
    residue_entity_id: Int[torch.Tensor, "B L_res"]
    residue_sym_id: Int[torch.Tensor, "B L_res"]
    atom_to_residue_idx_map: Int[torch.Tensor, "B L_atom"]
    edge_index: Int[torch.Tensor, "B E"]


@typecheck
@dataclass(frozen=True)
class MSAFeatures(BaseBatch):
    """MSA features."""

    aligned_sequences: Int[torch.Tensor, "B N_sampled N_msa L_token"]
    has_deletion: Int[torch.Tensor, "B N_sampled N_msa L_token"]
    deletion_value: Float[torch.Tensor, "B N_sampled N_msa L_token"]
    profile: Float[torch.Tensor, "B L_token d_profile"]
    deletion_mean: Float[torch.Tensor, "B L_token"]


@typecheck
@dataclass(frozen=True)
class ChainFeatures(BaseBatch):
    """Chain features."""

    entity_type: Int[torch.Tensor, "B L_chain"]
    contact_edges: Int[torch.Tensor, "B N_contact 2"]


@dataclass(kw_only=True, frozen=True)
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
    def shape(self) -> tuple[int]:
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
        return None

    @property
    def atom_length(self) -> int:
        """Return the atom length."""
        if len(self.reference.pos.shape) == 2:
            return self.reference.pos.shape[0]
        if len(self.reference.pos.shape) == 3:
            return self.reference.pos.shape[1]
        return None


@dataclass(kw_only=True, frozen=True)
class NoisyBatch(Batch):
    """Batch with noise for diffusion model."""

    # Tensor of timesteps
    t: torch.Tensor  # (..., 1)

    # Noisy rigids of frame atoms
    x_t: torch.Tensor  # (..., L, 3)
    x_mask: torch.Tensor # (..., L)

    # Self condition rigid
    x_sc: torch.Tensor | None = None  # (..., L, 3)

    def __post_init__(self) -> None:
        """Post-initialization to move tensors to correct device and dtype."""
        object.__setattr__(self, "t", self.t.to(self.device, dtype=self.dtype))
        object.__setattr__(self, "x_t", self.x_t.to(self.device, dtype=self.dtype))
        if self.x_sc is not None:
            object.__setattr__(
                self,
                "x_sc",
                self.x_sc.to(self.device, dtype=self.dtype),
            )
