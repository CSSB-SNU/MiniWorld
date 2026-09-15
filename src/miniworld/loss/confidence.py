"""Confidence targets + losses for phase 3 (pLDDT / PDE / PAE).

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
    atom_is_nuc: Bool[torch.Tensor, "L_atom"] | None = None,
    atom_is_ligand: Bool[torch.Tensor, "L_atom"] | None = None,
    max_distance: float = 15.0,
    nuc_max_distance: float = 30.0,
    distance_bins: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> tuple[Float[torch.Tensor, "N L_atom"], Bool[torch.Tensor, "L_atom"]]:
    """Per-atom lDDT in [0, 1] and the per-atom validity mask (>=1 valid neighbor).

    AF3 §4.3.1 neighbor rules:
      * inclusion radius depends on the **neighbor** atom j: 30 Å if j is a
        nucleic-acid atom, else 15 Å (``atom_is_nuc``);
      * for a **ligand** atom i, only ligand-polymer contacts count — intra-ligand
        (ligand-ligand) neighbor pairs are excluded (``atom_is_ligand``).
    Both extra masks are optional; omitting them recovers the flat-15 Å all-atom lDDT.
    O(L_atom**2) like :func:`metrics.cal_atom_lddt`; call under ``no_grad`` on a crop.
    """
    device = pred_atom_pos.device
    pred = pred_atom_pos.to(device=device, dtype=torch.float32)
    gt = gt_atom_pos.to(device=device, dtype=torch.float32)
    mask = atom_mask.to(device=device, dtype=torch.bool)

    pred_dist = torch.cdist(pred, pred)                # [N, L, L]
    gt_dist = torch.cdist(gt[None], gt[None])[0]       # [L, L]

    pair_mask = mask[:, None] & mask[None, :]
    pair_mask = pair_mask & (gt_dist > 0.0)
    # Per-neighbor (column j) inclusion radius: 30 Å for NA neighbors, else 15 Å.
    if atom_is_nuc is not None:
        is_nuc = atom_is_nuc.to(device=device, dtype=torch.bool)
        radius_j = torch.where(is_nuc, gt_dist.new_tensor(nuc_max_distance),
                               gt_dist.new_tensor(max_distance))  # [L]
        pair_mask = pair_mask & (gt_dist < radius_j[None, :])
    else:
        pair_mask = pair_mask & (gt_dist < max_distance)
    # Ligand atom i: keep only polymer neighbors (drop ligand-ligand pairs).
    if atom_is_ligand is not None:
        is_lig = atom_is_ligand.to(device=device, dtype=torch.bool)
        pair_mask = pair_mask & ~(is_lig[:, None] & is_lig[None, :])

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
def token_frames(
    frame_atom_pos: Float[torch.Tensor, "N L 3 3"],
    frame_valid: Bool[torch.Tensor, "L"],
    colinear_cos: float = 0.9063,  # cos(25 deg): AF3 near-colinear invalidity
) -> tuple[Float[torch.Tensor, "N L 3 3"], Float[torch.Tensor, "N L 3"], Bool[torch.Tensor, "N L"]]:
    """Build per-token rigid frames (R, t) from three atoms (Gram-Schmidt).

    ``frame_atom_pos[..., k, :]`` are the k=0,1,2 frame atoms — (N, CA, C) for
    protein tokens, (C1', C3', C4') for nucleotides (AF3 §4.3.2). The centre atom
    (k=1) is the translation; e1 points to atom 2, e2 is atom 0 orthogonalised.
    A frame is invalid when its token has no 3-atom frame (``frame_valid``) or the
    three atoms are near-colinear (angle < 25°).
    """
    a0 = frame_atom_pos[..., 0, :]
    a1 = frame_atom_pos[..., 1, :]  # centre
    a2 = frame_atom_pos[..., 2, :]
    v1 = a2 - a1
    v2 = a0 - a1
    e1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
    v2n = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / (u2.norm(dim=-1, keepdim=True) + 1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    rot = torch.stack([e1, e2, e3], dim=-1)  # [N, L, 3, 3] columns = basis
    colinear = (e1 * v2n).sum(-1).abs() > colinear_cos  # [N, L]
    valid = frame_valid.to(dtype=torch.bool)[None, :] & ~colinear
    return rot, a1, valid


def pae_target_bins(
    pred_rep_pos: Float[torch.Tensor, "N L 3"],
    gt_rep_pos: Float[torch.Tensor, "N L 3"],
    tok_valid: Bool[torch.Tensor, "N L"],
    pred_frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gt_frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_bins: int = 64,
    pae_max: float = 32.0,
) -> tuple[Int[torch.Tensor, "N L L"], Bool[torch.Tensor, "N L L"]]:
    """Frame-aligned error target, bucketed over [0, ``pae_max``].

    ``e_ij = || R_i^{-1}(x_j - t_i) |_pred - R_i^{-1}(x_j - t_i) |_gt ||`` where the
    frame ``(R_i, t_i)`` is token i's and ``x_j`` is token j's representative atom.
    ``*_frame`` are ``(R [N,L,3,3], t [N,L,3], valid [N,L])`` from :func:`token_frames`.
    Pair mask = frame i valid & rep j valid.
    """
    r_pred, t_pred, fvalid_pred = pred_frame
    r_gt, t_gt, fvalid_gt = gt_frame
    # x_j in frame i: R_i^T (x_j - t_i). rep pos [N,L,3] -> broadcast j over i.
    def _local(rot: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        d = x[:, None, :, :] - t[:, :, None, :]        # [N, Li, Lj, 3]
        # rot columns are the basis; R^T d = einsum over basis axis
        return torch.einsum("nikc,nijc->nijk", rot, d)  # [N, Li, Lj, 3]

    local_pred = _local(r_pred, t_pred, pred_rep_pos)
    local_gt = _local(r_gt, t_gt, gt_rep_pos)
    err = (local_pred - local_gt).norm(dim=-1)          # [N, Li, Lj]
    step = pae_max / n_bins
    bins = (err / step).clamp(0, n_bins - 1).long()
    frame_valid = (fvalid_pred & fvalid_gt)             # [N, L] on i
    pair_mask = frame_valid[:, :, None] & tok_valid[:, None, :]
    return bins, pair_mask


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
