"""Isolate FA4 sliding-window + seqused_k behavior vs hand references.

Reference A (position-window): query p attends key j iff |p-j|<=w AND j<L.
Reference B (rank-window on packed): same as A when front-packed (position==rank).
We also try measuring the window on the PACKED (nonzero) layout to see if FA's
varlen window is anchored differently for fixed-stride vs packed.
"""
import torch, torch.nn.functional as F
from flash_attn.cute import flash_attn_varlen_func

dev = "cuda"
torch.manual_seed(0)
N, S, H, D, W = 3, 32, 2, 16, 4
L = torch.tensor([32, 20, 12], dtype=torch.int32, device=dev)   # incl. one full row
q = torch.randn(N, S, H, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(N, S, H, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(N, S, H, D, device=dev, dtype=torch.bfloat16)
scale = D ** -0.5
cu = torch.arange(0, (N + 1) * S, S, dtype=torch.int32, device=dev)
pos = torch.arange(S, device=dev)
mvalid = (pos[None, :] < L[:, None])


def ref_pos_window():
    sc = torch.einsum('nqhd,nkhd->nhqk', q.float(), k.float()) * scale
    within = (pos[:, None] - pos[None, :]).abs() <= W           # [S,S]
    allowed = within[None] & mvalid[:, None, :]                 # keys valid & in window
    sc = sc.masked_fill(~allowed.view(N, 1, S, S), float('-inf'))
    return torch.einsum('nhqk,nkhd->nqhd', F.softmax(sc, -1), v.float())


def fa_fixed_stride(**kw):
    o = flash_attn_varlen_func(
        q.reshape(N * S, H, D), k.reshape(N * S, H, D), v.reshape(N * S, H, D),
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S,
        softmax_scale=scale, window_size=(W, W), **kw)
    if isinstance(o, tuple):
        o = o[0]
    return o.reshape(N, S, H, D).float()


def fa_packed():  # OLD path: nonzero pack, window over packed order
    idx = torch.nonzero(mvalid.flatten(), as_tuple=False).flatten()
    cup = F.pad(torch.cumsum(L.to(torch.int32), 0, dtype=torch.int32), (1, 0))
    qb, kb, vb = (t.reshape(N * S, H, D)[idx] for t in (q, k, v))
    o = flash_attn_varlen_func(qb, kb, vb, cu_seqlens_q=cup, cu_seqlens_k=cup,
                               max_seqlen_q=S, max_seqlen_k=S, softmax_scale=scale,
                               window_size=(W, W))
    if isinstance(o, tuple):
        o = o[0]
    full = o.new_zeros(N * S, H, D)
    full[idx] = o
    return full.view(N, S, H, D).float()


r = ref_pos_window()
m = mvalid.view(N, S, 1, 1).expand(N, S, H, D)


def mdiff(x):  # max abs diff over VALID query rows only (select, avoid NaN*0)
    d = torch.where(m, (x - r).abs(), torch.zeros_like(x))
    return d.max().item()


print(f"{'OLD packed vs ref':32s} {mdiff(fa_packed()):.5f}")
print(f"{'fixed-stride seqused_k':32s} {mdiff(fa_fixed_stride(seqused_k=L)):.5f}")
print(f"{'fixed-stride seqused_k+q':32s} {mdiff(fa_fixed_stride(seqused_k=L, seqused_q=L)):.5f}")
print("\n~0 = matches position-window+valid reference.")
