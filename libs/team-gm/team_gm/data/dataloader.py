import torch
import random

from torch.utils.data import DataLoader
from collections.abc import Mapping
from pathlib import PosixPath, Path
from dataclasses import dataclass
from jaxtyping import Float

from team_gm import typecheck
from . import chemical
from .features import Batch, SequenceFeatures, StructureFeatures


@typecheck
@dataclass
class InputData:
    seq: str
    xyz: Float[torch.Tensor, "L 14 3"]
    mask: Float[torch.Tensor, "L 14"]
    bfac: Float[torch.Tensor, "L 14"]
    occ: Float[torch.Tensor, "L 14"]
    name: str

    def __post_init__(self):
        if len(self.seq) != len(self.xyz):
            raise ValueError(
                f"Sequence length {len(self.seq)} does not match XYZ length "
                f"{len(self.xyz)}."
            )

    def __len__(self) -> int:
        return len(self.seq)

    @staticmethod
    def from_input_path(input_path: str | PosixPath) -> "InputData":
        input_path = Path(input_path)
        data = torch.load(input_path)
        return InputData(
            seq=data["seq"],
            xyz=data["xyz"],
            mask=data["mask"],
            bfac=data["bfac"],
            occ=data["occ"],
            name=input_path.stem,
        )

    def crop_chain(self, crop_size: int = 256, unclamp: bool = False) -> "InputData":
        # TODO: Use more sphisticated method to crop chain

        if len(self) <= crop_size:
            return self

        res_mask = ~self.mask[:, :3].logical_not().any(dim=-1)
        exists_idx = res_mask[:-crop_size].nonzero()[:, 0]
        if len(exists_idx) == 0:
            exists_idx = torch.tensor([len(res_mask) - crop_size])

        if unclamp:
            exists_idx = exists_idx[: random.randint(1, len(exists_idx))]
        crop_idx = random.choice(exists_idx).item()
        return InputData(
            seq=self.seq[crop_idx : crop_idx + crop_size],
            xyz=self.xyz[crop_idx : crop_idx + crop_size],
            mask=self.mask[crop_idx : crop_idx + crop_size],
            bfac=self.bfac[crop_idx : crop_idx + crop_size],
            occ=self.occ[crop_idx : crop_idx + crop_size],
            name=self.name,
        )


class ProteinData(torch.utils.data.Dataset):
    def __init__(
        self,
        item_dict: Mapping[int, list[tuple[list[str, str], int]]],
        input_dir_path: str | PosixPath,
        crop_chain: bool = True,
        crop_length: int = 256,
    ):
        self.pdb_ids = list(item_dict.keys())
        self.item_dict = item_dict
        self.input_dir_path = Path(input_dir_path)
        if not self.input_dir_path.exists():
            raise ValueError(f"Data directory {input_dir_path} does not exist.")
        self.crop_chain = crop_chain
        self.crop_length = crop_length

    def create_dataloader(self, **kwargs):
        params = {
            "shuffle": False,
            "num_workers": 4,
            "pin_memory": True,
            "collate_fn": Batch.collate_fn,
        }
        params.update(kwargs)
        return DataLoader(self, **params)

    @staticmethod
    def from_input_data(
        input_data: InputData,
        crop_chain: bool = False,
        crop_length: int = 256,
    ) -> Batch:
        # Crop input data
        if crop_chain and len(input_data.seq) > crop_length:
            input_data = input_data.crop_chain(crop_length)

        # Get SequenceFeatures
        L, N = len(input_data), chemical.RES_ATOM_NUM
        char_to_idx = lambda char: chemical.RES_NAMES.index(
            chemical.RESCHAR_TO_RESNAME[char]
        )
        res_type = torch.tensor([char_to_idx(s) for s in input_data.seq])
        # TODO: fix this assumption
        seq_idx = torch.arange(L)

        sequence = SequenceFeatures.from_sample(
            res_type=res_type,
            seq_idx=seq_idx,
        )

        # Get StructureFeatures
        atom_pos = torch.full((L, N, 3), float("NaN"), dtype=torch.float32)
        atom_pos[:, :14, :] = input_data.xyz
        atom_CoM = torch.nanmean(atom_pos[:, chemical.PROT_ATOM_CENTER_IDX], 0)
        atom_pos = atom_pos - atom_CoM

        atom_mask = torch.full((L, N), False)
        atom_mask[:, :14] = input_data.mask
        assert atom_pos[atom_mask].isnan().sum() == 0

        frame_idxs = chemical.RES_TYPE_RIGIDGROUP_ATOM_IDX[res_type]
        structure = StructureFeatures.from_sample(
            atom_pos=atom_pos,
            atom_mask=atom_mask,
            frame_idxs=frame_idxs,
        )

        return Batch(
            sequence=sequence,
            structure=structure,
        )

    @staticmethod
    def from_input_path(
        input_path: str | PosixPath,
        crop_chain: bool = False,
        crop_length: int = 256,
    ) -> Batch:
        input_data = InputData.from_input_path(input_path)
        return ProteinData.from_input_data(input_data, crop_chain, crop_length)

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx: int) -> Batch:
        pdb_id = self.pdb_ids[idx]
        (pdb_name, cluster_id), _ = random.choice(self.item_dict[pdb_id])
        input_path = self.input_dir_path / pdb_name[1:3] / f"{pdb_name}.pt"
        return ProteinData.from_input_path(input_path, self.crop_chain, self.crop_length)
