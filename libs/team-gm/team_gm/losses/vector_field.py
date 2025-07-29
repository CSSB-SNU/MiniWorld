import torch

from jaxtyping import Float, Bool

from team_gm import typecheck
from team_gm.utils.so3_utils import calc_rot_vf


@typecheck
def cal_trans_vf_loss(
    trans_gt: Float[torch.Tensor, "* L 3"],
    trans_pred: Float[torch.Tensor, "* L 3"],
    loss_mask: Bool[torch.Tensor, "* L"],
) -> Float[torch.Tensor, "*"]:
    """Calculate translation vector field loss.

    Parameters
    ----------
    trans_gt: FloatTensor, (..., L, 3)
        Tensor of ground truth frame translation.
    trans_pred: FloatTensor, (..., L, 3)
        Tensor of prediction frame translation.
    loss_mask: BoolTensor, (..., L)
        Tensor of masking certain residue loss.
    """

    trans_vf_loss = (trans_gt - trans_pred) ** 2
    trans_vf_loss = trans_vf_loss.masked_fill(~loss_mask[..., None], 0)
    trans_vf_loss = trans_vf_loss.sum(dim=(-1, -2))
    trans_vf_loss /= loss_mask.sum(dim=-1) * 3
    return trans_vf_loss


@typecheck
def cal_rots_vf_loss(
    rot_mats_t: Float[torch.Tensor, "* L 3 3"],
    rot_mats_gt: Float[torch.Tensor, "* L 3 3"],
    rot_mats_pred: Float[torch.Tensor, "* L 3 3"],
    loss_mask: Bool[torch.Tensor, "* L"],
) -> Float[torch.Tensor, "*"]:
    """Calculate rotation vector field loss.

    Parameters
    ----------
    rot_mats_t: FloatTensor, (..., L, 3, 3)
        Tensor of noisy frame rotation.
    rot_mats_gt: FloatTensor, (..., L, 3, 3)
        Tensor of ground truth frame rotation.
    rot_mats_pred: FloatTensor, (..., L, 3, 3)
        Tensor of prediction frame rotation.
    loss_mask: BoolTensor, (..., L)
        Tensor of masking certain residue loss.
    """

    rots_vf_gt = calc_rot_vf(rot_mats_t, rot_mats_gt)
    rots_vf_pred = calc_rot_vf(rot_mats_t, rot_mats_pred)
    rots_vf_loss = (rots_vf_gt - rots_vf_pred) ** 2
    rots_vf_loss = rots_vf_loss.masked_fill(~loss_mask[..., None], 0)
    rots_vf_loss = rots_vf_loss.sum(dim=(-1, -2))
    rots_vf_loss /= loss_mask.sum(dim=-1) * 3
    return rots_vf_loss
