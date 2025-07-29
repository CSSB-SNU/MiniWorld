import torch

from typing_extensions import Self
from dataclasses import dataclass, fields
from functools import cached_property
from team_gm.utils.data_utils import auto_tensor_collate, move_data_to_device
from jaxtyping import Float, Int, Bool
from team_gm import typecheck

def _auto_collate(data_list: list):
    assert all(isinstance(data_list[0], type(data)) for data in data_list)

    if isinstance(data_list[0], torch.Tensor):
        return auto_tensor_collate(data_list)
    # TODO: make it more general
    elif data_list[0] is None:
        return None
    else:
        return data_list


class BaseBatch:
    def to(self, device: str | torch.device):
        if not isinstance(device, torch.device):
            device = torch.device(device)

        return self.__class__(
            **{
                f.name: move_data_to_device(getattr(self, f.name), device)
                for f in fields(self)
            }
        )

    @classmethod
    def collate_fn(cls, batch_list: list[Self]) -> Self:
        if not batch_list:
            raise ValueError("batch_list cannot be empty.")
        if not all(isinstance(b, cls) for b in batch_list):
            raise TypeError(
                f"Expected all items in batch_list to be of type {cls.__name__}, "
                f"but got {[type(b) for b in batch_list]}."
            )

        collated_data = {}
        for f in fields(cls):
            data_list = [getattr(b, f.name) for b in batch_list]
            data_type = type(data_list[0])
            if not all(isinstance(d, data_type) for d in data_list):
                raise TypeError(
                    f"Expected all items in data_list for field '{f.name}' to be of "
                    f"type {data_type.__name__}, but got {[type(d) for d in data_list]}."
                )
            if data_type is torch.Tensor:
                collated_data[f.name] = auto_tensor_collate(data_list, 0)
            elif issubclass(data_type, BaseBatch):
                collated_data[f.name] = data_type.collate_fn(data_list)
            elif data_type is type(None):
                collated_data[f.name] = None
            elif data_type is list:
                collated_data[f.name] = [
                    item for sublist in data_list for item in sublist
                ]
            else:
                raise TypeError(
                    f"Unsupported data type for field '{f.name}': {data_type}."
                )

        return cls(**collated_data)

    def duplicate(self, num: int) -> Self:
        return self.collate_fn([self] * num)

    @classmethod
    def from_sample(cls, **kwargs) -> Self:
        batched_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                batched_kwargs[k] = v.unsqueeze(0)
            elif isinstance(v, BaseBatch):
                batched_kwargs[k] = v
            elif isinstance(v, type(None)):
                batched_kwargs[k] = None
            else:
                batched_kwargs[k] = [v]

        return cls(**batched_kwargs)

    @cached_property
    def batch_size(self) -> int:
        batch_size_dict = {}
        for f in fields(self):
            data = getattr(self, f.name)
            if isinstance(data, torch.Tensor):
                batch_size_dict[f.name] = data.shape[0]
            elif isinstance(data, BaseBatch):
                batch_size_dict[f.name] = data.batch_size
            elif data is None:
                pass
            elif isinstance(data, list):
                batch_size_dict[f.name] = len(data)
            else:
                raise TypeError(
                    f"Unsupported data type for field '{f.name}': {type(data)}."
                )

        if not batch_size_dict:
            return 0

        batch_size = next(iter(batch_size_dict.values()))
        if not all(size == batch_size for size in batch_size_dict.values()):
            raise ValueError(
                "Batch size is not consistent across all fields. "
                f"Sizes found: {batch_size_dict}."
            )
        return batch_size


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
    mask: Bool[torch.Tensor, "B L_atom"]
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
    residue_chain_break: dict[str, tuple[int, int]]  # {chain_id: (start, end)}


@typecheck
@dataclass(frozen=True)
class MSAFeatures(BaseBatch):
    aligned_sequences: Int[torch.Tensor, "B N_sampled N_msa L_token"]
    has_deletion: Bool[torch.Tensor, "B N_sampled N_msa L_token"]
    deletion_value: Float[torch.Tensor, "B N_sampled N_msa L_token"]
    profile: Float[torch.Tensor, "B N_msa L_token d_profile"]
    deletion_mean: Float[torch.Tensor, "B N_msa L_token"]

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
