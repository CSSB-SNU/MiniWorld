"""Pin down the CORRECT FA4 seqused API usage vs a hand-computed masked reference.

Tiny, global (no window) so the reference is unambiguous: each row attends only
to its first L[row] keys. Tests which kwargs make FA match.
"""
import torch, torch.nn.functional as F
from flash_attn.cute import flash_attn_varlen_func

dev = "cuda"
torch.manual_seed(0)
N, S, H, D = 2, 8, 2, 16
L = torch.tensor([5, 3], dtype=torch.int32, device=dev)   # valid counts per row
q = torch.randn(N, S, H, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(N, S, H, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(N, S, H, D, device=dev, dtype=torch.bfloat16)
scale = D ** -0.5
cu = torch.arange(0, (N + 1) * S, S, dtype=torch.int32, device=dev)
mvalid = (torch.arange(S, device=dev)[None, :] < L[:, None])   # [N,S] front-packed


def fa(**kw):
    o = flash_attn_varlen_func(
        q.reshape(N * S, H, D), k.reshape(N * S, H, D), v.reshape(N * S, H, D),
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S,
        softmax_scale=scale, **kw)
    if isinstance(o, tuple):
        o = o[0]
    return o.reshape(N, S, H, D).float()


def ref():  # manual masked softmax over first L keys (global)
    sc = torch.einsum('nqhd,nkhd->nhqk', q.float(), k.float()) * scale
    sc = sc.masked_fill(~mvalid.view(N, 1, 1, S), float('-inf'))
    return torch.einsum('nhqk,nkhd->nqhd', F.softmax(sc, -1), v.float())


r = ref()
m = mvalid.view(N, S, 1, 1)
for name, kw in [
    ("no mask", {}),
    ("seqused_k only", dict(seqused_k=L)),
    ("seqused_k + seqused_q", dict(seqused_k=L, seqused_q=L)),
]:
    try:
        o = fa(**kw)
        print(f"{name:24s} max|FA - masked_ref| on valid = {((o - r).abs() * m).max().item():.5f}")
    except Exception as e:
        print(f"{name:24s} ERROR: {type(e).__name__}: {e}")
print("\nWhichever ~0 is the correct seqused usage. 'no mask' should be LARGE.")
