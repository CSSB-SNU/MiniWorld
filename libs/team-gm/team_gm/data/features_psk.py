import torch
from dataclasses import dataclass
from jaxtyping import Float, Int, Bool

from team_gm import BaseBatch, typecheck


@typecheck
@dataclass(frozen=True)
class SequenceFeatures(BaseBatch):
    atom_residue_type: Int[torch.Tensor, "B L_atom"]
    residue_type: Int[torch.Tensor, "B L_res"]
    residue_ccd_idx: Int[torch.Tensor, "B L_res"]

@typecheck
@dataclass(frozen=True)
class StructureFeatures(BaseBatch):
    residue_pos: Float[torch.Tensor, "B L_token 3"]
    residue_mask: Bool[torch.Tensor, "B L_token"]
    atom_pos: Float[torch.Tensor, "B L_atom 3"]
    atom_mask: Bool[torch.Tensor, "B L_atom"]
    residue_bond: Int[torch.Tensor, "B n_residue_bond 3"]
    atom_bond: Int[torch.Tensor, "B n_atom_bond 6"]


@typecheck
@dataclass(frozen=True)
class ReferenceFeatures(BaseBatch):
    pos: Float[torch.Tensor, "B L_atom 3"]
    mask: Float[torch.Tensor, "B L_atom"]
    element: Float[torch.Tensor, "B L_atom"]
    charge: Float[torch.Tensor, "B L_atom"]
    space_uid: Int[torch.Tensor, "B L_atom"]



@typecheck
@dataclass(frozen=True)
class SchemeFeatures(BaseBatch):
    crop_indices: Int[torch.Tensor, "B L_res"]
    residue_idx: Int[torch.Tensor, "B L_res"]
    residue_idx_mono: Int[torch.Tensor, "B L_res"]
    residue_asym_id: Int[torch.Tensor, "B L_res"]
    residue_entity_id: Int[torch.Tensor, "B L_res"]
    residue_sym_id: Int[torch.Tensor, "B L_res"]
    atom_to_residue_idx_map: Int[torch.Tensor, "B L_atom"]
    residue_chain_break: list[dict[str, tuple[int, int]]]  # {chain_id: (start, end)}


@typecheck
@dataclass(frozen=True)
class MSAFeatures(BaseBatch):
    aligned_sequences: Int[torch.Tensor, "B N_sampled N_msa L_token"]
    has_deletion: Int[torch.Tensor, "B N_sampled N_msa L_token"]
    deletion_value: Float[torch.Tensor, "B N_sampled N_msa L_token"]
    profile: Float[torch.Tensor, "B L_token d_profile"]
    deletion_mean: Float[torch.Tensor, "B L_token"]

@dataclass(kw_only=True)
class Batch(BaseBatch):
    # Input data name.
    name: str | list  # (...)

    sequence : SequenceFeatures
    structure: StructureFeatures
    reference: ReferenceFeatures
    scheme: SchemeFeatures
    msa: MSAFeatures

    @property
    def shape(self) -> tuple[int]:
        return self.structure.atom_mask.shape

    @property
    def device(self) -> torch.device:
        return self.structure.atom_mask.device

    @property
    def dtype(self) -> torch.dtype:
        return self.structure.atom_pos.dtype

    @property
    def residue_length(self) -> int:
        if len(self.reference.ref_pos.shape) == 2:
            return self.scheme.residue_idx.shape[0]
        elif len(self.reference.ref_pos.shape) == 3:
            return self.scheme.residue_idx.shape[1]

    @property
    def atom_length(self) -> int:
        if len(self.reference.ref_pos.shape) == 2:
            return self.reference.ref_pos.shape[0]
        elif len(self.referenceref_pos.shape) == 3:
            return self.referenceref_pos.shape[1]


@dataclass(kw_only=True)
class NoisyBatch(Batch):
    # Tensor of timesteps
    t: torch.Tensor  # (..., 1)

    # Noisy rigids of frame atoms
    x_t: torch.Tensor  # (..., L, 3)

    # Self condition rigid
    x_sc: torch.Tensor | None = None  # (..., L, 3)

    def __post_init__(self):
        self.t = self.t.to(self.device, dtype=self.dtype)
        self.x_t = self.x_t.to(self.device, dtype=self.dtype)
        if self.x_sc is not None:
            self.x_sc = self.x_sc.to(self.device, dtype=self.dtype)
        super().__post_init__()