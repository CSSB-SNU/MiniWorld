"""Sequence-local plus structure-nearest neighbor indices for the SWA atom attention.

Moved from ``miniworld_engine.modules.swa_atom_attention.module`` (deleted in miniworld-engine 2.4.2 as dead code in the engine; MiniWorld's
diffusion module is its only caller).
"""

import torch
from jaxtyping import Bool, Float, Int


@torch.no_grad()
def build_local_structure_neighbor_indices(
    x_t: Float[torch.Tensor, "N S 3"],
    valid: Bool[torch.Tensor, "N S"],
    *,
    seq_neighbors: int = 128,
    structure_neighbors: int = 128,
    query_chunk_size: int = 128,
) -> tuple[Int[torch.Tensor, "N S K"], Bool[torch.Tensor, "N S K"]]:
    """Build sequence-local plus x_t-nearest neighbor indices.

    The sequence-local set is the nearest ``seq_neighbors`` atoms in valid atom
    order and includes self. The structure set takes the nearest
    ``structure_neighbors`` atoms by current noisy coordinates after excluding
    the sequence-local set, so the concatenated set is duplicate-free.

    This avoids full [N, S, S] distance tensors; each batch row and query chunk
    is processed independently to keep peak memory bounded.
    """
    n, s = valid.shape
    device = valid.device
    if seq_neighbors < 1:
        raise ValueError("seq_neighbors must be >= 1 so every valid atom includes self.")
    if structure_neighbors < 0:
        raise ValueError("structure_neighbors must be >= 0.")

    seq_k = min(seq_neighbors, s)
    struct_k = min(structure_neighbors, s)
    total_k = seq_k + struct_k
    neighbor_idx = torch.zeros(n, s, total_k, dtype=torch.int32, device=device)
    neighbor_mask = torch.zeros(n, s, total_k, dtype=torch.bool, device=device)

    coords = x_t.to(torch.float32)
    for b in range(n):
        valid_pos = torch.nonzero(valid[b], as_tuple=False).flatten()
        m = int(valid_pos.numel())
        if m == 0:
            continue

        row_seq_k = min(seq_k, m)
        ranks = torch.arange(m, device=device)
        rank_dist = (ranks[:, None] - ranks[None, :]).abs().to(torch.float32)
        # Stable tie-break toward lower atom index for deterministic neighbors.
        rank_dist = rank_dist + valid_pos.view(1, m).to(torch.float32) * 1e-6
        _, seq_rank_idx = torch.topk(rank_dist, k=row_seq_k, dim=-1, largest=False)
        seq_atom_idx = valid_pos[seq_rank_idx]
        neighbor_idx[b, valid_pos, :row_seq_k] = seq_atom_idx.to(torch.int32)
        neighbor_mask[b, valid_pos, :row_seq_k] = True

        if struct_k == 0:
            continue

        row_struct_k = min(struct_k, max(m - row_seq_k, 0))
        if row_struct_k == 0:
            continue

        row_coords = coords[b, valid_pos]
        for q_start in range(0, m, query_chunk_size):
            q_stop = min(q_start + query_chunk_size, m)
            q_pos = valid_pos[q_start:q_stop]
            d2 = torch.cdist(row_coords[q_start:q_stop], row_coords, p=2).square()
            seq_excluded = torch.zeros(q_stop - q_start, m, dtype=torch.bool, device=device)
            seq_excluded.scatter_(1, seq_rank_idx[q_start:q_stop, :row_seq_k], True)
            d2 = d2.masked_fill(seq_excluded, float("inf"))
            d2 = d2 + valid_pos.view(1, m).to(torch.float32) * 1e-6
            struct_vals, struct_rank_idx = torch.topk(
                d2, k=row_struct_k, dim=-1, largest=False
            )
            offset = seq_k
            neighbor_idx[b, q_pos, offset:offset + row_struct_k] = valid_pos[
                struct_rank_idx
            ].to(torch.int32)
            neighbor_mask[b, q_pos, offset:offset + row_struct_k] = torch.isfinite(
                struct_vals
            )

    return neighbor_idx, neighbor_mask
