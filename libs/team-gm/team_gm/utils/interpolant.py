import torch

from pathlib import Path
from dataclasses import dataclass
from jaxtyping import Float, Bool

from team_gm import typecheck
from . import so3_utils
from .frame_utils import centered_gaussian, create_rigid
from .rigid_utils import Rigid
from .data_utils import NM_TO_ANG_SCALE


@typecheck
@dataclass
class RigidPath:
    t: Float[torch.Tensor, "B"]  # noqa: F821
    trans_0: Float[torch.Tensor, "B L 3"]
    trans_t: Float[torch.Tensor, "B L 3"]
    rot_mats_0: Float[torch.Tensor, "B L 3 3"]
    rot_mats_t: Float[torch.Tensor, "B L 3 3"]
    res_mask: Bool[torch.Tensor, "B L"]

    @property
    def rigid_t(self) -> Rigid:
        return create_rigid(rot_mats=self.rot_mats_t, trans=self.trans_t)

    @property
    def rigid_sc(self) -> Rigid | None:
        if not hasattr(self, "_rigid_sc") or self._rigid_sc is None:
            return None
        return self._rigid_sc

    @rigid_sc.setter
    def rigid_sc(self, value: Rigid | None) -> None:
        if value is None:
            self._rigid_sc = None
        elif isinstance(value, Rigid):
            if value.shape != self.res_mask.shape:
                raise ValueError(
                    f"Expected rigid shape {self.res_mask.shape}, got {value.shape}"
                )
            self._rigid_sc = value
        else:
            raise TypeError(f"Expected Rigid or None, got {type(value)}")


igso3 = so3_utils.SampleIGSO3(
    num_omega=1000,
    sigma_grid=torch.linspace(0.1, 1.5, 1000),
    cache_dir=Path(__file__).parent / ".cache",
)


# TODO: Make a general diffusion path sampling function
@typecheck
def sample_rigid_path(
    rigid: Rigid,  # (B, L)
    res_mask: Bool[torch.Tensor, "B L"],
    equal_timestep: bool = False,
) -> RigidPath:
    B, L = res_mask.shape
    device = res_mask.device

    if rigid.shape != (B, L):
        raise ValueError(f"rigid must be {B, L}, got {rigid.shape}")

    t = torch.rand(B, device=device)
    if equal_timestep:
        t[:] = t[0].item()

    trans_0 = centered_gaussian(res_mask) * NM_TO_ANG_SCALE
    trans_0 = trans_0.nan_to_num(0)
    trans_t = (1 - t[..., None, None]) * trans_0 + t[..., None, None] * rigid.get_trans()

    noisy_rotmats = igso3.sample(torch.tensor([1.5]), B * L).to(device)
    noisy_rotmats = noisy_rotmats.reshape(B, L, 3, 3)
    rot_mats_1 = rigid.get_rots().get_rot_mats()
    rot_mats_0 = torch.einsum("...ij,...jk->...ik", rot_mats_1, noisy_rotmats)
    rot_mats_t = so3_utils.geodesic_t(t[..., None, None], rot_mats_1, rot_mats_0)
    rot_mats_t[~res_mask] = torch.eye(3, device=device)

    return RigidPath(
        t=t,
        trans_0=trans_0,
        trans_t=trans_t,
        rot_mats_0=rot_mats_0,
        rot_mats_t=rot_mats_t,
        res_mask=res_mask,
    )
