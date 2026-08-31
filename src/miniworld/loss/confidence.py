"""Confidence targets + losses for phase4 (pLDDT / PDE / PAE).

All targets are computed from a *predicted* structure vs the ground truth under
``no_grad`` (the structure model is frozen); the confidence head is then trained with
per-bin cross-entropy against these bucketed targets.

  * **pLDDT** — per-atom lDDT of pred vs GT, bucketed into ``n_bins`` over [0, 1].
    Mirrors :func:`miniworld.loss.metrics.cal_atom_lddt` but returns the per-atom
    vector (its internal ``per_atom_lddt``) instead of the scalar.
  * **PDE** — |d_ij^pred - d_ij^gt| between per-token representative atoms, bucketed
    over [0, ``pde_max``].
  * **PAE** — frame-aligned error; requires per-token backbone frames, which are NOT
    a dataset feature yet (Stage B). :func:`pae_target_bins` is the seam: it returns
    ``None`` until ``token_frame`` inputs are supplied, so the PAE loss is skipped
    (weight 0) without breaking the training format.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int


# ---------------------------------------------------------------------------
# Per-token representative atom positions (CB / pseudo-beta)
# ---------------------------------------------------------------------------
def representative_positions(
    atom_pos: Float[torch.Tensor, "N L_atom 3"],
    atom_pos_mask: Bool[torch.Tensor, "N L_atom"],
    atom_to_token_idx: Int[torch.Tensor, "N L_atom"],
    atom_is_rep: Bool[torch.Tensor, "N L_atom"],
    token_num: int,
) -> tuple[Float[torch.Tensor, "N L 3"], Bool[torch.Tensor, "N L"]]:
    """Gather one representative-atom position per token (capture-safe).

    Same lowest-index-valid-rep-atom gather as
    :func:`miniworld.utils.structure.distance.get_representative_distances`, but
    returns the ``[N, token_num, 3]`` positions (not the clamped distance matrix), so
    the caller can form unclamped distances / frame-relative errors.
    """
    device = atom_pos.device
    n, length, _ = atom_pos.shape
    rep_valid = atom_pos_mask & atom_is_rep
    tok_idx = atom_to_token_idx.clamp(0, token_num - 1)

    atom_ids = torch.arange(length, device=device).unsqueeze(0).expand(n, length)
    cand = torch.where(rep_valid, atom_ids, torch.full_like(atom_ids, length))
    rep_idx = torch.full((n, token_num), length, dtype=cand.dtype, device=device)
    rep_idx.scatter_reduce_(1, tok_idx, cand, reduce="amin", include_self=True)
    tok_valid = rep_idx < length
    gather_idx = rep_idx.clamp(max=length - 1)
    rep_pos = torch.gather(atom_pos, 1, gather_idx.unsqueeze(-1).expand(n, token_num, 3))
    return rep_pos, tok_valid


# ---------------------------------------------------------------------------
# pLDDT target (per-atom lDDT, bucketed)
# ---------------------------------------------------------------------------
def per_atom_lddt(
    pred_atom_pos: Float[torch.Tensor, "N L_atom 3"],
    gt_atom_pos: Float[torch.Tensor, "L_atom 3"],
    atom_mask: Bool[torch.Tensor, "L_atom"],
    max_distance: float = 15.0,
    distance_bins: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> tuple[Float[torch.Tensor, "N L_atom"], Bool[torch.Tensor, "L_atom"]]:
    """Per-atom lDDT in [0, 1] and the per-atom validity mask (>=1 valid neighbor).

    O(L_atom**2) like :func:`metrics.cal_atom_lddt`; call under ``no_grad`` on a crop.
    """
    device = pred_atom_pos.device
    pred = pred_atom_pos.to(device=device, dtype=torch.float32)
    gt = gt_atom_pos.to(device=device, dtype=torch.float32)
    mask = atom_mask.to(device=device, dtype=torch.bool)

    pred_dist = torch.cdist(pred, pred)                # [N, L, L]
    gt_dist = torch.cdist(gt[None], gt[None])[0]       # [L, L]

    pair_mask = mask[:, None] & mask[None, :]
    pair_mask = pair_mask & (gt_dist > 0.0) & (gt_dist < max_distance)

    delta = torch.abs(pred_dist - gt_dist)             # [N, L, L]
    bins = torch.tensor(distance_bins, dtype=torch.float32, device=device)
    cond = (delta.unsqueeze(-1) <= bins) & pair_mask.unsqueeze(-1)  # [N, L, L, K]
    num_in_bin = cond.sum(dim=2)                        # [N, L, K]
    total = pair_mask.sum(dim=-1, keepdim=True).float() # [L, 1]
    frac = num_in_bin.float() / (total + 1e-8)          # [N, L, K]
    lddt = frac.mean(dim=-1)                            # [N, L]

    valid = mask & (pair_mask.sum(dim=-1) > 0)          # [L]
    return lddt, valid


def plddt_target_bins(
    lddt: Float[torch.Tensor, "N L_atom"],
    n_bins: int = 50,
) -> Int[torch.Tensor, "N L_atom"]:
    """Bucket per-atom lDDT in [0, 1] into ``n_bins`` uniform bins."""
    return (lddt.clamp(0.0, 1.0 - 1e-6) * n_bins).long()


# ---------------------------------------------------------------------------
# PDE target (representative distance error, bucketed)
# ---------------------------------------------------------------------------
def pde_target_bins(
    pred_rep_pos: Float[torch.Tensor, "N L 3"],
    gt_rep_pos: Float[torch.Tensor, "N L 3"],
    tok_valid: Bool[torch.Tensor, "N L"],
    n_bins: int = 64,
    pde_max: float = 32.0,
) -> tuple[Int[torch.Tensor, "N L L"], Bool[torch.Tensor, "N L L"]]:
    """|d_ij^pred - d_ij^gt| bucketed over [0, ``pde_max``], with pair-valid mask."""
    pred_d = torch.cdist(pred_rep_pos, pred_rep_pos)  # [N, L, L]
    gt_d = torch.cdist(gt_rep_pos, gt_rep_pos)        # [N, L, L]
    err = torch.abs(pred_d - gt_d)
    step = pde_max / n_bins
    bins = (err / step).clamp(0, n_bins - 1).long()
    pair_mask = tok_valid[:, :, None] & tok_valid[:, None, :]
    return bins, pair_mask


def pred_rep_distance(
    rep_pos: Float[torch.Tensor, "N L 3"],
    tok_valid: Bool[torch.Tensor, "N L"],
    min_distance: float = 2.0,
    max_distance: float = 22.0,
) -> Float[torch.Tensor, "N L L"]:
    """Predicted representative-atom distance matrix (clamped) for the head input."""
    dist = torch.cdist(rep_pos, rep_pos)
    pair_mask = tok_valid[:, :, None] & tok_valid[:, None, :]
    dist = dist.masked_fill(~pair_mask, max_distance)
    return dist.clamp(min_distance, max_distance)


# ---------------------------------------------------------------------------
# PAE target (frame-aligned error) — STAGE B seam
# ---------------------------------------------------------------------------
def pae_target_bins(
    pred_rep_pos: Float[torch.Tensor, "N L 3"],
    gt_rep_pos: Float[torch.Tensor, "N L 3"],
    tok_valid: Bool[torch.Tensor, "N L"],
    token_frame: object | None = None,
    n_bins: int = 64,
    pae_max: float = 32.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Frame-aligned error target, bucketed. Returns ``None`` until frames exist.

    PAE needs a per-token backbone frame ``T_i`` (built from N/CA/C atoms) to express
    ``x_j`` in ``i``'s frame: ``e_ij = || T_i^{-1} x_j^pred - T_i^{-1} x_j^gt ||``.
    Per-token frames are not a dataset feature yet, so this returns ``None`` (PAE loss
    skipped). Wire ``token_frame`` (rotations [N,L,3,3] + translations [N,L,3] for pred
    and gt) here for Stage B.
    """
    if token_frame is None:
        return None
    raise NotImplementedError(
        "PAE frame-aligned target: supply per-token N/CA/C frames (Stage B).",
    )


# ---------------------------------------------------------------------------
# Cross-entropy losses (masked)
# ---------------------------------------------------------------------------
def masked_ce(
    logits: torch.Tensor,        # [..., C]
    target: torch.Tensor,        # [...]
    mask: torch.Tensor,          # [...]
) -> torch.Tensor:
    """Mean cross-entropy over masked positions."""
    c = logits.shape[-1]
    flat_logits = logits.reshape(-1, c)
    flat_target = target.reshape(-1).clamp(0, c - 1)
    flat_mask = mask.reshape(-1).float()
    ce = F.cross_entropy(flat_logits.float(), flat_target, reduction="none")
    return (ce * flat_mask).sum() / (flat_mask.sum() + 1e-8)
