import torch

from jaxtyping import Float, Bool

from team_gm import typecheck


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
