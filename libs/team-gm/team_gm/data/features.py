import torch
import numpy as np

from dataclasses import dataclass, replace
from scipy.spatial.transform import Rotation
from functools import cached_property
from jaxtyping import Float, Int, Bool, Shaped

from team_gm import BaseBatch, typecheck
from team_gm.utils.frame_utils import atom_to_frame, create_rigid
from team_gm.utils.rigid_utils import Rigid
from team_gm.data import chemical


@typecheck
@dataclass(frozen=True)
class SequenceFeatures(BaseBatch):
    res_type: Int[torch.Tensor, "B L"]
    seq_idx: Int[torch.Tensor, "B L"]

    @cached_property
    @typecheck
    def res_name(self) -> Shaped[np.ndarray, "B L"]:
        res_type = self.res_type.cpu()
        return np.array(chemical.RES_NAMES)[res_type]


@typecheck
@dataclass(frozen=True)
class StructureFeatures(BaseBatch):
    atom_pos: Float[torch.Tensor, "B L N 3"]
    atom_mask: Bool[torch.Tensor, "B L N"]
    frame_idxs: Int[torch.Tensor, "B L 3"]

    @cached_property
    @typecheck
    def res_mask(self) -> Bool[torch.Tensor, "B L"]:
        return torch.gather(self.atom_mask, -1, self.frame_idxs).all(-1)

    @cached_property
    def rigid(self) -> Rigid:
        frame_pos = torch.gather(
            self.atom_pos, -2, self.frame_idxs.unsqueeze(-1).tile(3)
        )
        rigid = atom_to_frame(*frame_pos.unbind(-2))
        trans = rigid.get_trans()
        trans[~self.res_mask] = 0.0
        rot_mats = rigid.get_rots().get_rot_mats()
        rot_mats[~self.res_mask] = torch.eye(3, device=self.atom_pos.device)
        return create_rigid(rot_mats, trans)

    @torch.no_grad()
    def augmentation(self, trans_scale: float = 1.0) -> "StructureFeatures":
        B = self.batch_size
        device = self.atom_pos.device

        rot_aug = Rotation.random(B).as_matrix()
        rot_aug = torch.from_numpy(rot_aug).reshape(B, 3, 3)
        rot_aug = rot_aug.to(device).float()
        trans_aug = trans_scale * torch.randn(B)
        trans_aug = trans_aug.to(device)

        atom_CoM = self.atom_pos.nanmean(dim=1).nanmean(dim=1)
        atom_pos = self.atom_pos - atom_CoM[:, None, None]
        atom_pos = torch.einsum("bij, blnj->blni", rot_aug, atom_pos)
        atom_pos = atom_pos + trans_aug[:, None, None, None]
        return replace(self, atom_pos=atom_pos)


@dataclass
class Batch(BaseBatch):
    sequence: SequenceFeatures
    structure: StructureFeatures
