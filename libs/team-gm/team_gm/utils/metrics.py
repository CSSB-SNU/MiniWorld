import torch
import numpy as np

from jaxtyping import Float, Bool
from collections.abc import Sequence

from team_gm import typecheck
from team_gm.utils.data_utils import to_numpy


Array = np.ndarray | torch.Tensor


# Derived from https://github.com/nghiaho12/rigid_transform_3D
@typecheck
def align_pos(
    prb_pos: Float[np.ndarray, "L 3"],
    ref_pos: Float[np.ndarray, "L 3"],
) -> tuple[
    Float[np.ndarray, "L 3"],
    Float[np.ndarray, "3 3"],
    Float[np.ndarray, "3"],
]:
    """Align probe positions to reference positions.

    Parameters
    ----------
    prb_pos: np.ndarray, [L, 3]
        Probe atom positions.
    ref_pos: np.ndarray, [L, 3]
        Reference atom positions.

    Returns
    -------
    aligned_prb_pos: np.ndarray, [L, 3]
        Aligned probe atom positions.
    R: np.ndarray, [3, 3]
        Rotation matrix.
    T: np.ndarray, [3]
        Translation vector.
    """
    if np.isnan(prb_pos).any() or np.isnan(ref_pos).any():
        raise ValueError(f"NaN in input positions. {prb_pos} {ref_pos}")
    prb_CoM = np.mean(prb_pos, axis=0)
    ref_CoM = np.mean(ref_pos, axis=0)

    # find rotation
    H = (prb_pos - prb_CoM).T @ (ref_pos - ref_CoM)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    T = -R @ prb_CoM + ref_CoM
    aligned_prb_pos = R @ prb_pos.T + T.reshape(3, 1)

    return aligned_prb_pos.T, R, T


@typecheck
def cal_aligned_rmsd(
    prb_pos: Float[Array, "L 3"],
    ref_pos: Float[Array, "L 3"],
    res_mask: Bool[Array, "L"] | None = None,  # noqa: F821
) -> float:
    """Calculate RMSD of two sets of atom positions.
    Positions will be aligned before calculating RMSD.

    Parameters
    ----------
    prb_pos: ndarray or FloatTensor, (L, 3)
        Predicted atom positions.
    ref_pos: ndarray or FloatTensor, (L, 3)
        Reference atom positions.
    res_mask: ndarray or BoolTensor, (L)
        Mask of valid residues.

    Returns
    -------
    rmsd: float
        RMSD of two sets of atom positions.
    """
    if isinstance(prb_pos, torch.Tensor):
        prb_pos = to_numpy(prb_pos)
    if isinstance(ref_pos, torch.Tensor):
        ref_pos = to_numpy(ref_pos)
    if isinstance(res_mask, torch.Tensor):
        res_mask = to_numpy(res_mask)

    if res_mask is not None:
        non_gap_idx = np.where(~np.isnan(ref_pos).any(-1) & res_mask)[0]
    else:
        non_gap_idx = np.where(~np.isnan(ref_pos).any(-1))[0]
    if np.isnan(prb_pos[non_gap_idx]).any():
        raise ValueError(f"NaN in predicted positions. {prb_pos[non_gap_idx]}")
    aligned_prb_pos, _, _ = align_pos(prb_pos[non_gap_idx], ref_pos[non_gap_idx])
    rmsd = np.mean(np.linalg.norm(aligned_prb_pos - ref_pos[non_gap_idx], axis=-1))
    return rmsd.item()


@typecheck
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
    atom_mask: ndarray or BoolTensor, [L,]
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
    pred_dist = torch.cdist(pred_atom_pos, pred_atom_pos, p=2)
    gt_dist = torch.cdist(gt_atom_pos, gt_atom_pos, p=2) # [L, L]

    # get valid pairs
    pair_mask = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(0) # [L, L]
    pair_mask &= gt_dist > 0
    pair_mask &= gt_dist < max_distance

    delta = torch.abs(pred_dist - gt_dist)
    lddt_per_atom = torch.zeros_like(atom_mask, dtype=torch.float32)
    denom = pair_mask.sum(dim=1).clamp(min=1e-6).float()  # number of valid pairs per atom

    for d in distance_bins:
        agree = ((delta < d) & pair_mask).sum(dim=1).float()
        lddt_per_atom += agree.div(denom)

    # --- Average over bins, then over atoms ---
    lddt_per_atom.div_(len(distance_bins))
    global_lddt = lddt_per_atom.mean().item()
    return global_lddt
