import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from team_gm import typecheck

from miniworld.utils.structure import (
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

    # Compute residue-level shortest distances and mask
    residue_dists, residue_pair_mask = get_shortest_distances(
        atom_pos=atom_pos,
        atom_pos_mask=atom_pos_mask,
        atom_to_res_idx=atom_to_res_idx,
        min_distance=min_distance,
        max_distance=max_distance,
    )  # residue_dists: (*, L, L), residue_pair_mask: (*, L, L)

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
        alpha: Focal loss α parameter.
        gamma: Focal loss γ parameter.

    Returns:
        Focal loss averaged over valid residue pairs. Shape: (*,)
    """
    *lead, L, _, C = logit_pred.shape
    if C != 2:
        raise ValueError(f"logit_pred last dim must be 2 (got {C})")
    device = logit_pred.device

    # 1) Residue-level shortest distances and mask
    with torch.no_grad():
        residue_dists, residue_pair_mask = get_shortest_distances_from_multistructures(
            atom_pos=atom_pos,
            atom_pos_mask=atom_pos_mask,
            atom_to_res_idx=atom_to_res_idx,
            min_distance=min_distance,
            max_distance=max_distance,
        )  # (..., R_max, R_max), (..., R_max, R_max)

    if residue_dists.shape[-1] != L:
        raise ValueError(
            f"Size mismatch: logit_pred L={L}, "
            f"residue_dists last dim={residue_dists.shape[-1]}"
        )

    # 2) Contact targets (0/1) from distances
    contact_target = residue_dists <= cutoff  # (*, L, L), bool
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

    # α_t depending on target class
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
