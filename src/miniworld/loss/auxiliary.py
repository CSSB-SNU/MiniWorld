import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from team_gm import typecheck

from miniworld.utils.structure import (
    extract_contact_map,
    get_shortest_distances,
    get_shortest_distances_from_multistructures,
)


@typecheck
def cal_all_atom_loss(
    atom_pos_gt: Float[torch.Tensor, "* L N 3"],
    atom_pos_pred: Float[torch.Tensor, "* L N 3"],
    atom_mask: Bool[torch.Tensor, "* L N"],
    loss_mask: Bool[torch.Tensor, "* L"],
) -> Float[torch.Tensor, "*"]:
    """Calculate all atom MSE loss.

    Parameters
    ----------
    atom_pos_gt: FloatTensor, (..., L, N, 3)
        Tensor of ground truth atom coordinate.
    atom_pos_pred: FloatTensor, (..., L, N, 3)
        Tensor of prediction atom coordinate.
    atom_mask: BoolTensor, (..., L, N)
        Tensor of masking atom.
    loss_mask: BoolTensor, (..., L)
        Tensor of masking certain residue loss.

    """
    atom_mask = atom_mask * loss_mask[..., None]
    all_atom_loss = torch.norm(
        (atom_pos_gt - atom_pos_pred).masked_fill(~atom_mask[..., None], 0),
        dim=-1,
    )
    all_atom_loss = all_atom_loss.sum(dim=(-1, -2))
    all_atom_loss /= atom_mask.sum(dim=(-1, -2))
    return all_atom_loss


@typecheck
def cal_dist_mat_loss(
    atom_pos_gt: Float[torch.Tensor, "* L N 3"],
    atom_pos_pred: Float[torch.Tensor, "* L N 3"],
    atom_mask: Bool[torch.Tensor, "* L N"],
    loss_mask: Bool[torch.Tensor, "* L"],
    threshold: float = 6,
) -> Float[torch.Tensor, "*"]:
    """Calculate distogram matrix MSE loss.

    Parameters
    ----------
    atom_pos_gt: FloatTensor, (..., L, N, 3)
        Tensor of ground truth atom coordinate.
    atom_pos_pred: FloatTensor, (..., L, N, 3)
        Tensor of prediction atom coordinate.
    atom_mask: BoolTensor, (..., L, N)
        Tensor of masking atom.
    loss_mask: BoolTensor, (..., L)
        Tensor of masking certain residue loss.
    threshold: float
        Threshold of define neighbor atoms.

    """
    atom_mask = atom_mask * loss_mask[..., None]
    pair_dist_mask = atom_mask[..., None, :, None] * atom_mask[..., None, :, None, :]
    pair_dist_mask = torch.tril(pair_dist_mask, -1)

    atom_pos_gt = atom_pos_gt.masked_fill(~atom_mask[..., None], 0)
    pair_dists_gt = torch.norm(
        atom_pos_gt[..., None, :, None, :] - atom_pos_gt[..., None, :, None, :, :],
        dim=-1,
    )

    atom_pos_pred = atom_pos_pred.masked_fill(~atom_mask[..., None], 0)
    pair_dists_pred = torch.norm(
        atom_pos_pred[..., None, :, None, :] - atom_pos_pred[..., None, :, None, :, :],
        dim=-1,
    )

    # No loss on anything > threshold A
    proximity_mask = pair_dists_gt < threshold
    pair_dist_mask = pair_dist_mask * proximity_mask
    dist_mat_loss = (pair_dists_gt - pair_dists_pred) ** 2
    dist_mat_loss = dist_mat_loss * pair_dist_mask
    dist_mat_loss = torch.sum(dist_mat_loss, dim=(-1, -2, -3, -4))
    dist_mat_loss /= torch.sum(pair_dist_mask, dim=(-1, -2, -3, -4))
    return dist_mat_loss


@typecheck
def cal_distogram_loss(
    logit_pred: Float[torch.Tensor, "* L L D"],
    residue_pos_gt: Float[torch.Tensor, "* L 3"],
    residue_pos_mask: Bool[torch.Tensor, "* L"],
) -> Float[torch.Tensor, "*"]:
    """Pairwise distances are bucketized into D bins using (D-1) bin edges.

    - Edges default to linspace(2.0, 22.0, D-1) -> bins: [0,e0), [e0,e1), ..., [e_{-1}, +inf)
    - Only i<j pairs (upper triangle) are used; invalid coords are masked out.
    """
    *lead, L, _, D = logit_pred.shape
    device = logit_pred.device

    # Distances
    pos = torch.where(
        residue_pos_mask[..., None],
        residue_pos_gt,
        torch.zeros_like(residue_pos_gt),
    )
    dist = torch.cdist(pos, pos)  # (*, L, L)

    # Targets (AF2-style binning; last bin is overflow)
    edges = torch.linspace(2.0, 22.0, D - 1, device=device)
    target = torch.bucketize(dist, edges)  # (*, L, L), int64 in [0, D-1]

    # Valid pair mask: i<j and both residues valid
    tri = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    pair_valid = (
        residue_pos_mask.unsqueeze(-1) & residue_pos_mask.unsqueeze(-2)
    ) & tri  # (*, L, L)

    # CE loss per pair
    logits = logit_pred.permute(*range(len(lead)), -1, -3, -2)  # (*, D, L, L)
    ce = F.cross_entropy(logits, target, reduction="none")  # (*, L, L)

    # Masked reduction per leading sample
    denom = pair_valid.sum(dim=(-2, -1)).clamp_min(1).to(ce.dtype)  # (*,)
    num = (ce * pair_valid).sum(dim=(-2, -1))  # (*,)
    return num / denom  # (*,)


@typecheck
def cal_atom_distogram_loss(
    logit_pred: Float[torch.Tensor, "* L L D"],
    atom_pos: Float[torch.Tensor, "* L_atom 3"],
    atom_pos_mask: Bool[torch.Tensor, "* L_atom"],
    atom_to_res_idx: Int[torch.Tensor, "* L_atom"],
    min_distance: float = 2.0,
    max_distance: float = 22.0,
) -> Float[torch.Tensor, "*"]:
    """Calculate residue level distogram loss from atom positions."""
    *lead, L, _, D = logit_pred.shape
    device = logit_pred.device


    if atom_pos.dim() == 3:
        # Single structure
        residue_dists, residue_pair_mask = get_shortest_distances(
            atom_pos=atom_pos,
            atom_pos_mask=atom_pos_mask,
            atom_to_res_idx=atom_to_res_idx,
            min_distance=min_distance,
            max_distance=max_distance,
        )  # (L_max, L_max), (L_max, L_max)
    else:
        residue_dists, residue_pair_mask = get_shortest_distances_from_multistructures(
            atom_pos=atom_pos,
            atom_pos_mask=atom_pos_mask,
            atom_to_res_idx=atom_to_res_idx,
            min_distance=min_distance,
            max_distance=max_distance,
        )  # (..., R_max, R_max), (..., R_max, R_max)

    # Targets (AF2-style binning; last bin is overflow)
    edges = torch.linspace(min_distance, max_distance, D - 1, device=device)
    target = torch.bucketize(residue_dists, edges)  # (*, L, L), int64 in [0, D-1]

    # Valid pair mask: i<j and both residues valid
    tri = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    residue_pair_mask = residue_pair_mask.to(torch.bool)
    residue_pair_mask = residue_pair_mask & tri  # (*, L, L)
    # CE loss per pair
    logits = logit_pred.permute(*range(len(lead)), -1, -3, -2)  # (*, D, L, L)
    ce = F.cross_entropy(logits, target, reduction="none")  # (*, L, L)

    # Masked reduction per leading sample
    denom = residue_pair_mask.sum(dim=(-2, -1)).clamp_min(1).to(ce.dtype)  # (*,)
    num = (ce * residue_pair_mask).sum(dim=(-2, -1))  # (*,)
    return num / denom  # (*,)




@typecheck
def cal_contact_map_focal_loss(
    logit_pred: Float[torch.Tensor, "* L L 2"],
    atom_pos: Float[torch.Tensor, "* N L_atom 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L_atom"],
    atom_to_res_idx: Int[torch.Tensor, "* L_atom"],
    cutoff: float = 6.0,
    min_distance: float = 2.0,
    max_distance: float = 22.0,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Float[torch.Tensor, "*"]:
    """Focal loss for residue-level contact map prediction (2-class logits).

    Args:
        logit_pred: Contact logits, last dim is class (0: non-contact, 1: contact).
            Shape: (*, L, L, 2)
        atom_pos: Atomic coordinates. Shape: (B, N, L_atom, 3)
        atom_pos_mask: Atom mask. True for valid atoms. Shape: (B, N, L_atom)
        atom_to_res_idx: Residue index per atom position. Shape: (B, L_atom)
        cutoff: Distance cutoff (Å) for defining contacts.
        min_distance: Minimum allowed distance for distance computation.
        max_distance: Maximum allowed distance for distance computation.
        alpha: Focal loss alpha parameter.
        gamma: Focal loss gamma parameter.

    Returns:
        Focal loss averaged over valid residue pairs. Shape: (*,)
    """
    *lead, L, _, C = logit_pred.shape
    if C != 2:
        msg = f"logit_pred last dim must be 2 (got {C})"
        raise ValueError(msg)
    device = logit_pred.device

    contact_target, residue_pair_mask = extract_contact_map(
        atom_pos=atom_pos,
        atom_pos_mask=atom_pos_mask,
        atom_to_res_idx=atom_to_res_idx,
        cutoff=cutoff,
        min_distance=min_distance,
        max_distance=max_distance,
    )  # (*, L, L), bool
    target = contact_target.long()  # (*, L, L), in {0,1}

    # 3) Valid residue pairs: both residues exist and i < j (upper triangle)
    tri = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=device),
        diagonal=1,
    )  # (L, L)
    residue_pair_mask = residue_pair_mask.to(torch.bool)  # (*, L, L)
    valid_pair_mask = residue_pair_mask & tri  # (*, L, L)

    # 4) Focal loss on 2-class logits
    # logits: (*, 2, L, L) for cross_entropy
    logits = logit_pred.permute(*range(len(lead)), -1, -3, -2)  # (*, 2, L, L)

    # Standard CE loss per pair
    ce = F.cross_entropy(
        logits,
        target,
        reduction="none",
    )  # (*, L, L)

    # p_t = exp(-CE) (probability of the true class)
    p_t = torch.exp(-ce)  # (*, L, L)

    # alpha_t depending on target class
    alpha_tensor_pos = torch.as_tensor(alpha, dtype=ce.dtype, device=device)
    alpha_tensor_neg = torch.as_tensor(1.0 - alpha, dtype=ce.dtype, device=device)
    alpha_factor = torch.where(
        contact_target, alpha_tensor_pos, alpha_tensor_neg
    )  # (*, L, L)

    modulating_factor = (1.0 - p_t) ** gamma  # (*, L, L)

    focal = alpha_factor * modulating_factor * ce  # (*, L, L)

    # 5) Masked reduction over residue pairs
    focal = focal * valid_pair_mask.to(focal.dtype)  # (*, L, L)

    denom = valid_pair_mask.sum(dim=(-2, -1)).clamp_min(1).to(focal.dtype)  # (*,)
    num = focal.sum(dim=(-2, -1))  # (*,)

    return num / denom  # (*,)
@typecheck
def cal_contact_map_weighted_bce_loss(
    logit_pred: Float[torch.Tensor, "* L L 2"],
    atom_pos: Float[torch.Tensor, "* N L_atom 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L_atom"],
    atom_to_res_idx: Int[torch.Tensor, "* L_atom"],
    cutoff: float = 6.0,
    min_distance: float = 2.0,
    max_distance: float = 22.0,
    pos_weight: float = 1.0,
    long_range_min_seq_sep: int = 16,
    long_range_sigmoid_k: float = 1.0,
    long_range_sigmoid_amp: float = 0.0,
) -> Float[torch.Tensor, "*"]:
    """Weighted BCE loss for residue-level contact map prediction (2-class logits).

    Args:
        logit_pred: Contact logits, last dim is class (0: non-contact, 1: contact).
            Shape: (*, L, L, 2)
        atom_pos: Atomic coordinates. Shape: (*, N, L_atom, 3)
        atom_pos_mask: Atom mask. True for valid atoms. Shape: (*, N, L_atom)
        atom_to_res_idx: Residue index per atom position. Shape: (*, L_atom)
        cutoff: Distance cutoff (Å) for defining contacts.
        min_distance: Minimum allowed distance for distance computation.
        max_distance: Maximum allowed distance for distance computation.
        pos_weight: Multiplicative weight for positive class (contact). Negative class weight is 1.0.
        long_range_min_seq_sep: Sequence separation where long-range boosting starts (≈3x via sigmoid).
        long_range_sigmoid_k: Slope of the sigmoid used for long-range boost; larger -> sharper transition.
        long_range_sigmoid_amp: Amplitude of the long-range boost; 0 disables it (default), 2 roughly matches the previous ~3x.

    Returns:
        Weighted BCE loss averaged over valid residue pairs. Shape: (*,)
    """
    *lead, L, _, C = logit_pred.shape
    if C != 2:
        msg = f"logit_pred last dim must be 2 (got {C})"
        raise ValueError(msg)
    device = logit_pred.device

    contact_target, residue_pair_mask = extract_contact_map(
        atom_pos=atom_pos,
        atom_pos_mask=atom_pos_mask,
        atom_to_res_idx=atom_to_res_idx,
        cutoff=cutoff,
        min_distance=min_distance,
        max_distance=max_distance,
    )  # (*, L, L), bool
    target = contact_target.to(logit_pred.dtype)  # (*, L, L), in {0,1}

    # 3) Valid residue pairs: both residues exist and i < j (upper triangle)
    tri = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=device),
        diagonal=1,
    )  # (L, L)
    residue_pair_mask = residue_pair_mask.to(torch.bool)  # (*, L, L)
    valid_pair_mask = residue_pair_mask & tri  # (*, L, L)

    # 4) Weighted BCE on 2-class logits
    # Convert 2-class logits to a single binary logit (pos vs neg), equivalent to 2-class softmax
    bin_logit = logit_pred[..., 1] - logit_pred[..., 0]  # (*, L, L)

    # Standard BCE loss per pair
    bce = F.binary_cross_entropy_with_logits(
        bin_logit,
        target,
        reduction="none",
    )  # (*, L, L)

    # Positive weighting
    pos_w = torch.as_tensor(pos_weight, dtype=bce.dtype, device=device)
    weight = torch.where(contact_target, pos_w, torch.ones_like(bce))  # (*, L, L)
    bce = bce * weight  # (*, L, L)

    # Long-range boost: smoothly scale loss toward ~3x when |i-j| exceeds the threshold
    idx = torch.arange(L, device=device)
    seq_sep = (idx[None, :, None] - idx[None, None, :]).abs()  # (1, L, L)
    sigmoid_k = torch.as_tensor(long_range_sigmoid_k, dtype=bce.dtype, device=device)
    sigmoid_amp = torch.as_tensor(long_range_sigmoid_amp, dtype=bce.dtype, device=device)
    long_range_weight = 1.0 + sigmoid_amp * torch.sigmoid(
        (seq_sep - long_range_min_seq_sep).to(bce.dtype) * sigmoid_k
    )  # (1, L, L) -> in [1, 1+amp]
    bce = bce * long_range_weight  # (*, L, L)

    # 5) Masked reduction over residue pairs
    bce = bce * valid_pair_mask.to(bce.dtype)  # (*, L, L)

    denom = valid_pair_mask.sum(dim=(-2, -1)).clamp_min(1).to(bce.dtype)  # (*,)
    num = bce.sum(dim=(-2, -1))  # (*,)

    return num / denom  # (*,)

@typecheck
def _extract_long_range_contact_targets_and_mask(
    atom_pos: Float[torch.Tensor, "* N L_atom 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L_atom"],
    atom_to_res_idx: Int[torch.Tensor, "* L_atom"],
    cutoff: float,
    min_distance: float,
    max_distance: float,
    min_seq_sep: int,
    L: int,
    device: torch.device,
) -> tuple[Bool[torch.Tensor, "* L L"], Bool[torch.Tensor, "* L L"]]:
    """Helper: extract long-range contact targets and valid-pair mask."""
    with torch.no_grad():
        residue_dists, residue_pair_mask = get_shortest_distances_from_multistructures(
            atom_pos=atom_pos,
            atom_pos_mask=atom_pos_mask,
            atom_to_res_idx=atom_to_res_idx,
            min_distance=min_distance,
            max_distance=max_distance,
        )  # (*, L, L), (*, L, L)

    contact_target = residue_dists <= cutoff  # (*, L, L), bool

    # i < j
    tri = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=device),
        diagonal=1,
    )

    # |i - j| >= min_seq_sep
    idx = torch.arange(L, device=device)
    seq_sep_mask = (idx[None, :, None] - idx[None, None, :]).abs() >= min_seq_sep
    seq_sep_mask = seq_sep_mask.squeeze(0)  # (L, L)

    valid_pair_mask = (
        residue_pair_mask.to(torch.bool) & tri & seq_sep_mask
    )  # (*, L, L)

    return contact_target, valid_pair_mask


@typecheck
def cal_long_range_precision(
    logit_pred: Float[torch.Tensor, "* L L 2"],
    atom_pos: Float[torch.Tensor, "* N L_atom 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L_atom"],
    atom_to_res_idx: Int[torch.Tensor, "* L_atom"],
    cutoff: float = 6.0,
    min_distance: float = 2.0,
    max_distance: float = 22.0,
    min_seq_sep: int = 16,
) -> Float[torch.Tensor, "*"]:
    *lead, L, _, C = logit_pred.shape
    if C != 2:
        raise ValueError("logit_pred last dim must be 2")
    device = logit_pred.device

    contact_target, valid_pair_mask = _extract_long_range_contact_targets_and_mask(
        atom_pos=atom_pos,
        atom_pos_mask=atom_pos_mask,
        atom_to_res_idx=atom_to_res_idx,
        cutoff=cutoff,
        min_distance=min_distance,
        max_distance=max_distance,
        min_seq_sep=min_seq_sep,
        L=L,
        device=device,
    )

    pred_contact = logit_pred[..., 1] > logit_pred[..., 0]  # (*, L, L)

    tp = (pred_contact & contact_target & valid_pair_mask).sum(dim=(-2, -1))
    fp = (pred_contact & ~contact_target & valid_pair_mask).sum(dim=(-2, -1))

    denom = (tp + fp).clamp_min(1).to(tp.dtype)
    return tp.to(denom.dtype) / denom


@typecheck
def cal_long_range_recall(
    logit_pred: Float[torch.Tensor, "* L L 2"],
    atom_pos: Float[torch.Tensor, "* N L_atom 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L_atom"],
    atom_to_res_idx: Int[torch.Tensor, "* L_atom"],
    cutoff: float = 6.0,
    min_distance: float = 2.0,
    max_distance: float = 22.0,
    min_seq_sep: int = 16,
) -> Float[torch.Tensor, "*"]:
    *lead, L, _, C = logit_pred.shape
    if C != 2:
        raise ValueError("logit_pred last dim must be 2")
    device = logit_pred.device

    contact_target, valid_pair_mask = _extract_long_range_contact_targets_and_mask(
        atom_pos=atom_pos,
        atom_pos_mask=atom_pos_mask,
        atom_to_res_idx=atom_to_res_idx,
        cutoff=cutoff,
        min_distance=min_distance,
        max_distance=max_distance,
        min_seq_sep=min_seq_sep,
        L=L,
        device=device,
    )

    pred_contact = logit_pred[..., 1] > logit_pred[..., 0]

    tp = (pred_contact & contact_target & valid_pair_mask).sum(dim=(-2, -1))
    fn = (~pred_contact & contact_target & valid_pair_mask).sum(dim=(-2, -1))

    denom = (tp + fn).clamp_min(1).to(tp.dtype)
    return tp.to(denom.dtype) / denom


@typecheck
def cal_long_range_f1(
    logit_pred: Float[torch.Tensor, "* L L 2"],
    atom_pos: Float[torch.Tensor, "* N L_atom 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L_atom"],
    atom_to_res_idx: Int[torch.Tensor, "* L_atom"],
    cutoff: float = 6.0,
    min_distance: float = 2.0,
    max_distance: float = 22.0,
    min_seq_sep: int = 16,
) -> Float[torch.Tensor, "*"]:
    precision = cal_long_range_precision(
        logit_pred,
        atom_pos,
        atom_pos_mask,
        atom_to_res_idx,
        cutoff,
        min_distance,
        max_distance,
        min_seq_sep,
    )
    recall = cal_long_range_recall(
        logit_pred,
        atom_pos,
        atom_pos_mask,
        atom_to_res_idx,
        cutoff,
        min_distance,
        max_distance,
        min_seq_sep,
    )

    denom = (precision + recall).clamp_min(1e-8)
    return 2.0 * precision * recall / denom


@typecheck
def cal_long_range_auroc(
    logit_pred: Float[torch.Tensor, "* L L 2"],
    atom_pos: Float[torch.Tensor, "* N L_atom 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L_atom"],
    atom_to_res_idx: Int[torch.Tensor, "* L_atom"],
    cutoff: float = 6.0,
    min_distance: float = 2.0,
    max_distance: float = 22.0,
    min_seq_sep: int = 16,
) -> Float[torch.Tensor, "*"]:
    *lead, L, _, C = logit_pred.shape
    if C != 2:
        raise ValueError("logit_pred last dim must be 2")
    device = logit_pred.device

    contact_target, valid_pair_mask = _extract_long_range_contact_targets_and_mask(
        atom_pos=atom_pos,
        atom_pos_mask=atom_pos_mask,
        atom_to_res_idx=atom_to_res_idx,
        cutoff=cutoff,
        min_distance=min_distance,
        max_distance=max_distance,
        min_seq_sep=min_seq_sep,
        L=L,
        device=device,
    )

    # Binary score: P(contact)
    prob = torch.softmax(logit_pred, dim=-1)[..., 1]  # (*, L, L)

    # Flatten per sample
    prob = prob[valid_pair_mask]
    target = contact_target[valid_pair_mask]

    # Handle degenerate cases
    if prob.numel() == 0:
        return torch.zeros((), device=device)

    # Rank-based AUROC (equivalent to Mann–Whitney U)
    order = torch.argsort(prob)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(
        1,
        order.numel() + 1,
        device=device,
        dtype=ranks.dtype,
    )

    pos = target
    n_pos = pos.sum()
    n_neg = pos.numel() - n_pos

    if n_pos == 0 or n_neg == 0:
        return torch.ones((), device=device)

    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return auc
