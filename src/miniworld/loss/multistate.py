import torch
import torch.nn.functional as F

from jaxtyping import Float, Bool, Int

from team_gm import typecheck

from MiniWorld.utils.structure.sdist import get_shortest_distances

@typecheck
def cal_shortest_distogram_loss(
    logit_pred: Float[torch.Tensor, "* L L D"],
    atom_pos: Float[torch.Tensor, "* N L 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L"],
    atom_to_res_idx: Int[torch.Tensor, "* L"],
    min_distance: float = 2.0,
    max_distance: float = 22.0,
) -> Float[torch.Tensor, "*"]:
    """
    - Pairwise distances are bucketized into D bins using (D-1) bin edges.
    - Edges default to linspace(2.0, 22.0, D-1) -> bins: [0,e0), [e0,e1), ..., [e_{-1}, +inf)
    - Only i<j pairs (upper triangle) are used; invalid coords are masked out.
    """
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
    loss = num / denom  # (*,)

    return loss
