import torch
import numpy as np

from jaxtyping import Float, Bool
from collections.abc import Sequence

from team_gm import typecheck
from team_gm.utils.data_utils import to_numpy


Array = np.ndarray | torch.Tensor

def cal_atom_lddt(
    pred_atom_pos: Float[Array, "L 3"],
    gt_atom_pos: Float[Array, "L 3"],
    atom_mask: Bool[Array, "L"],  # noqa: F821
    max_distance: float = 15.0,
    distance_bins: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """Calculate lDDT score of two sets of atom positions.

    Parameters
    ----------
    pred_positions: ndarray or FloatTensor, [L, 3]
        Predicted atom positions.
    gt_positions: ndarray or FloatTensor, [L, 3]
        Ground truth atom positions.
    atom_mask: ndarray or BoolTensor, [L]
        Mask of valid atoms.
    max_distance: float, optional
        Maximum distance to consider for lDDT calculation.
        Default is 15.0.
    distance_bins: Sequence[float], optional
        Distance bins to use for lDDT calculation.
        Default is (0.5, 1.0, 2.0, 4.0).

    Returns
    -------
    lddt: float
        lDDT score.
    """
    if isinstance(pred_atom_pos, np.ndarray):
        pred_atom_pos = torch.from_numpy(pred_atom_pos)
    if isinstance(gt_atom_pos, np.ndarray):
        gt_atom_pos = torch.from_numpy(gt_atom_pos)
    if isinstance(atom_mask, np.ndarray):
        atom_mask = torch.from_numpy(atom_mask)

    # distance matrix
    pred_dist = pred_atom_pos[None, :] - pred_atom_pos[:, None, :]
    pred_dist = torch.norm(pred_dist, dim=-1)
    gt_dist = gt_atom_pos[None, :] - gt_atom_pos[:, None, :]
    gt_dist = torch.norm(gt_dist, dim=-1)


    # get valid pairs
    pair_mask = atom_mask[:, None] * atom_mask[None, :]
    pair_mask &= gt_dist > 0
    pair_mask &= gt_dist < max_distance

    delta = torch.abs(pred_dist - gt_dist)
    lddt = torch.zeros_like(atom_mask).float()
    for distance_bin in distance_bins:
        condition = ((delta <= distance_bin) * pair_mask).sum(-1).float()
        condition /= pair_mask.sum(-1) + 1e-8
        lddt += condition / len(distance_bins)
    lddt = (lddt * atom_mask).sum() / (atom_mask.sum() + 1e-8)
    return float(lddt)

if __name__ == "__main__":
    loaded = torch.load("debug_output.pt", map_location="cpu", weights_only=False)
    atom_pos_pred = loaded["atom_pos_pred"]
    atom_pos_gt = loaded["atom_pos_gt"]
    atom_mask = loaded["atom_mask"]
    lddt = cal_atom_lddt(atom_pos_pred[0], atom_pos_gt, atom_mask)