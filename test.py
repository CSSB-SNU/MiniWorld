import torch
import torch.nn.functional as F
import numpy as np

from jaxtyping import Float, Bool
from collections.abc import Sequence

from team_gm import typecheck
from team_gm.utils.data_utils import to_numpy

@torch.no_grad()
def group_by_expert_no_loop(y: torch.Tensor, topk_indices: torch.Tensor, padding: int = 4):
    """
    y:            (M, D)
    topk_indices: (M, K) with expert ids in [0, E-1]
    padding:      group size multiple per expert
    returns:
      sorted_y: (sum_e ceil(cnt_e/padding)*padding, D)
      idx_map:  (same,) original row index; -1 where padded
      expert_map: (same,) expert id for each row (pads included)
    """
    device = y.device
    M, D = y.shape
    E = int(topk_indices.amax().item()) + 1

    # membership[m, e] == True if row m is routed to expert e (appears once per expert)
    # use one_hot to avoid explicit loops
    membership = F.one_hot(topk_indices, num_classes=E).bool().any(dim=1)  # (M, E)

    # per-expert counts and padding
    counts = membership.sum(dim=0, dtype=torch.long)                       # (E,)
    pad = (padding - (counts % padding)) % padding                         # (E,)
    final_counts = counts + pad                                            # (E,)
    total = int(final_counts.sum().item())

    # block start offset for each expert in the final concatenation
    start = torch.cumsum(final_counts, dim=0) - final_counts               # (E,)

    # ranks within each expert (stable by original row order)
    cumsum_col = membership.to(torch.int64).cumsum(dim=0)                  # (M, E)
    me = membership.nonzero(as_tuple=False)                                 # (N_sel, 2) -> (m, e)
    m_sel, e_sel = me[:, 0], me[:, 1]
    rank_in_e = cumsum_col[m_sel, e_sel] - 1                                # (N_sel,)
    pos = start[e_sel] + rank_in_e                                          # (N_sel,)

    # allocate and scatter
    sorted_y = y.new_zeros((total, D))
    sorted_y[pos] = y[m_sel]

    idx_map = torch.full((total,), -1, device=device, dtype=torch.long)
    idx_map[pos] = m_sel

    # expert id for every output row (including padded rows)
    expert_map = torch.repeat_interleave(torch.arange(E, device=device, dtype=torch.long), final_counts)

    return sorted_y, idx_map, expert_map

if __name__ == "__main__":
    M = 8
    D = 2
    k = 2
    y = torch.arange(M * D).view(M, D).float()
    E = 4
    score = torch.randn(M, E, dtype=y.dtype, device=y.device)
    _, topk_indices = torch.topk(score, k=k, dim=-1)
    sorted_y = []
    idx_map = []
    expert_map = []
    padding = 4
    for expert_idx in range(topk_indices.max().item()+1):
        selected = y[(topk_indices == expert_idx).any(dim=-1)]
        original_idx = torch.nonzero((topk_indices == expert_idx).any(dim=-1), as_tuple=False).view(-1)
        # pad remainder by 16
        _pad = len(selected) % padding
        if _pad > 0:
            _pad = padding - _pad
            selected = torch.cat([selected, torch.zeros(_pad, selected.shape[1], device=y.device)], dim=0)
            original_idx = torch.cat([original_idx, torch.zeros(_pad, dtype=original_idx.dtype, device=y.device)-1], dim=0)
        sorted_y.append(selected)
        idx_map.append(original_idx)
        expert_map.append(torch.ones(len(original_idx), dtype=torch.int64, device=y.device) * expert_idx)
    sorted_y = torch.cat(sorted_y, dim=0)
    idx_map = torch.cat(idx_map, dim=0)
    expert_map = torch.cat(expert_map, dim=0)

    # vectorized_version
    sorted_y_v, idx_map_v, expert_map_v = group_by_expert_no_loop(y, topk_indices, padding=padding)

    # print diff
    assert (sorted_y == sorted_y_v).all()
    assert (idx_map == idx_map_v).all()
    assert (expert_map == expert_map_v).all()
    breakpoint()