import torch
import torch.nn.functional as F


def route(router_weight, x, k):
    # (E, N) @ (M, N) -> (M, E)
    logit = x.matmul(router_weight.T)
    # add small noise to handle tie
    # logit += torch.randn_like(logit) * 1e-6
    topk_logit, topk_indices = torch.topk(logit, k=k, dim=-1)
    topk_score = F.softmax(topk_logit, dim=-1)
    return topk_score, topk_indices  # (M, k)


def loss_free_route(expert_frequency, router_weight, x, k):
    # (E, N) @ (M, N) -> (M, E)
    # bias = torch.randn(1, router_weight.size(0), device=x.device, dtype=x.dtype) * 0.01
    # Option 1.
    logit = F.linear(x, router_weight, bias=None)

    # biased logit
    bias = expert_frequency.log().unsqueeze(0)  # (1, E)
    biased_logit = logit - bias

    # _, topk_indices = torch.topk(logit, k=k, dim=-1)
    # topk_logit = logit.gather(dim=-1, index=topk_indices)
    # topk_score = F.softmax(topk_logit, dim=-1)
    _, topk_indices = torch.topk(biased_logit, k=k, dim=-1)

    topk_logit = logit.gather(dim=-1, index=topk_indices)
    topk_score = F.softmax(topk_logit, dim=-1)
    return topk_score, topk_indices  # (M, k)


def group_by_expert(
    y: torch.Tensor,
    topk_score: torch.Tensor,
    topk_indices: torch.Tensor,
    padding: int = 128,
):
    """
    y:            (M, D)
    topk_score:   (M, K)
    topk_indices: (M, K) with expert ids in [0, E-1]
    padding:      group size multiple per expert
    returns:
      sorted_y: (sum_e ceil(cnt_e/padding)*padding, D)
      sorted_score: (same,) expert score for each row (pads included, 0 on pads)
      idx_map:  (same,) original row index; -1 where padded
      expert_map: (same,) expert id for each row (pads included)
    """
    device = y.device
    M, D = y.shape
    _, K = topk_indices.shape
    E = int(topk_indices.amax().item()) + 1

    # membership[m, e] == True if row m is routed to expert e (appears once per expert)
    scores = torch.zeros(M, E, dtype=topk_score.dtype, device=topk_score.device)
    membership = torch.zeros(M, E, dtype=torch.bool, device=topk_indices.device)
    scores.scatter_(1, topk_indices, topk_score)
    membership.scatter_(1, topk_indices, True)

    pos, sel, expert_map, sorted_score = [], [], [], []

    # Note : I don't like the for loop, but it is not that slow thoguh.
    pos_offset = 0
    for ee in range(E):
        _sel = torch.where(membership[:, ee])[0]
        # padding
        _sel = F.pad(_sel, (0, padding - len(_sel) % padding), value=-1)
        _pos = torch.arange(len(_sel), device=device)
        _pos += pos_offset
        _expert = torch.full_like(_sel, ee)
        _score = scores[_sel, ee]
        pos_offset += len(_sel)

        pos.append(_pos)
        sel.append(_sel)
        expert_map.append(_expert)
        sorted_score.append(_score)

    pos, sel, expert_map, sorted_score = map(
        torch.cat, (pos, sel, expert_map, sorted_score)
    )
    total = pos.shape[0]

    # allocate and scatter features
    sorted_y = torch.zeros((total, D), device=device, dtype=y.dtype)
    valid = sel >= 0
    sorted_y[pos[valid]] = y[sel[valid]]

    idx_map = torch.full((total,), -1, device=device, dtype=torch.long)
    idx_map[pos] = sel

    return sorted_y, sorted_score, idx_map, expert_map, pos, sel


def scatter_expert(
    shape: torch.Size,
    y: torch.Tensor,
    idx_map: torch.Tensor,
):
    valid = idx_map >= 0
    output = torch.zeros(shape, device=y.device, dtype=y.dtype).index_add_(
        0, idx_map[valid], y[valid]
    )
    return output


def stack_expert(
    shape: torch.Size,  # (L, E) or (L, E, D)
    y: torch.Tensor,  # (L_pooled,) or (L_pooled, D)
    idx_map: torch.Tensor,  # (L_pooled,) long in [0..L-1] or -1
    expert_map: torch.Tensor,  # (L_pooled,) long in [0..E-1] or -1
) -> torch.Tensor:
    """
    Map pooled rows to (L, E) or (L, E, D) by (idx_map, expert_map).
    Duplicates are summed. Invalid entries (=-1) ignored.
    """
    assert len(shape) in (2, 3), f"shape must be (L,E) or (L,E,D), got {tuple(shape)}"

    if len(shape) == 3:
        L, E, D = shape
        out = torch.zeros((L * E, D), device=y.device, dtype=y.dtype)
        need_squeeze = False
    else:
        L, E = shape
        D = None
        out = torch.zeros(
            (L * E, 1), device=y.device, dtype=y.dtype
        )  # work in 2D then squeeze
        need_squeeze = True

    valid = (idx_map >= 0) & (expert_map >= 0)
    if not valid.any():
        return out.view(shape) if not need_squeeze else out.view(L, E)

    idx = idx_map[valid].to(torch.long)  # (N,)
    exp = expert_map[valid].to(torch.long)  # (N,)
    flat_idx = idx * E + exp  # (N,)

    vals = y[valid]
    # Ensure 2D values for index_add_ into (..., F)
    if vals.dim() == 1:
        vals = vals.unsqueeze(-1)  # (N,) -> (N,1)

    if D is not None:
        assert vals.size(-1) == D, f"D mismatch: y has {vals.size(-1)} vs target {D}"
    else:
        # Target is (L,E) -> only accept last-dim 1; if larger, it's ambiguous
        assert vals.size(-1) == 1, (
            f"For (L,E) output, y must be (N,) or (N,1), got last-dim {vals.size(-1)}"
        )

    out.index_add_(0, flat_idx, vals)

    if need_squeeze:
        return out.view(L, E)  # (L,E)
    else:
        return out.view(L, E, D)  # (L,E,D)