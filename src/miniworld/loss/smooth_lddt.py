from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from team_gm import typecheck

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jaxtyping import Bool, Float


@torch.compile
@typecheck
def cal_smooth_lddt(
    pred_coord: Float[torch.Tensor, "... N 3"],
    gt_coord: Float[torch.Tensor, "... N 3"],
    is_nucleotide: Bool[torch.Tensor, "... N"],
    mask: Bool[torch.Tensor, "... N"],
    distance_bins: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    nucleotide_cutoff: float = 30.0,
    non_nucleotide_cutoff: float = 15.0,
) -> Float[torch.Tensor, ""]:
    """Smooth lDDT loss (AF3 Algorithm 27).

    Computes sigmoid-smoothed lDDT at multiple distance thresholds. Inclusion radius is
    per atom-i: 30A for nucleotides, 15A for others.

    Parameters
    ----------
    pred_coord
        Predicted atom coordinates.
    gt_coord
        Ground-truth atom coordinates.
    is_nucleotide
        Per-atom flag for nucleotide atoms (DNA/RNA), which use a wider inclusion
        radius.
    mask
        Valid atom mask.
    distance_bins
        Distance thresholds for sigmoid scoring.
    nucleotide_cutoff
        Inclusion radius for nucleotide atom pairs.
    non_nucleotide_cutoff
        Inclusion radius for non-nucleotide atom pairs.

    """
    pred_dist = torch.cdist(pred_coord, pred_coord)
    gt_dist = torch.cdist(gt_coord, gt_coord)

    dist_diff = torch.abs(pred_dist - gt_dist)
    score = sum(torch.sigmoid(thres - dist_diff) for thres in distance_bins)
    score = score / len(distance_bins)

    is_nuc = is_nucleotide.unsqueeze(-1)
    cutoff_mask = (gt_dist < nucleotide_cutoff) & is_nuc
    cutoff_mask = cutoff_mask | ((gt_dist < non_nucleotide_cutoff) & ~is_nuc)

    diag_mask = ~torch.eye(mask.shape[-1], dtype=torch.bool, device=mask.device)
    mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    mask_2d = mask_2d & cutoff_mask & diag_mask

    score = score * mask_2d
    lddt = score.sum(dim=(-1, -2)) / mask_2d.float().sum(dim=(-1, -2)).clamp(min=1)
    return (1 - lddt).mean()
