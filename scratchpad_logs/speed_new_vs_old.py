"""Speed: OLD packed varlen (nonzero+gather+FA+scatter) vs NEW seqused (deployed)."""
import statistics as st
import torch, torch.nn.functional as F
from flash_attn.cute import flash_attn_varlen_func
from team_gm.modules.layers import swa_atom_attention as M

dev = "cuda"
torch.manual_seed(0)
H, D, W, BLOCKS = 4, 32, 64, 3
scale = D ** -0.5


def old_full(q, k, v, valid, s):
    n = q.shape[0]
    seqlens = valid.sum(-1, dtype=torch.int32)
    idx = torch.nonzero(valid.flatten(), as_tuple=False).flatten()          # dynamic!
    cu = F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))
    qb, kb, vb = (t.reshape(n * s, H, D).to(torch.bfloat16)[idx] for t in (q, k, v))
    o = flash_attn_varlen_func(qb, kb, vb, cu_seqlens_q=cu, cu_seqlens_k=cu,
                               max_seqlen_q=s, max_seqlen_k=s, softmax_scale=scale, window_size=(W, W))
    o = o[0] if isinstance(o, tuple) else o
    full = o.new_zeros(n * s, H, D); full[idx] = o
    return full.view(n, s, H, D)


def new_full(q, k, v, valid, s):
    n = q.shape[0]
    _, _, seqused, cu, max_s, val = M.build_attention_params(
        torch.zeros(n, s, D // 2, device=dev), torch.zeros(n, s, D // 2, device=dev), valid, 1)
    return M.flash_window_seqused(q, k, v, cu, seqused, max_s, val, n, s, scale, W)


def bench(fn, q, k, v, valid, s, it=50):
    for _ in range(10):
        for _ in range(BLOCKS): fn(q, k, v, valid, s)
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(BLOCKS): fn(q, k, v, valid, s)
        b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)


print(f"{'N':>3} {'S':>5} {'OLD packed':>11} {'NEW seqused':>12} {'speedup':>8}")
for n, s in [(4, 3072), (16, 3072), (48, 3072)]:
    L = torch.randint(s // 2, s, (n,), device=dev).to(torch.int32)
    valid = torch.arange(s, device=dev)[None, :] < L[:, None]
    q = torch.randn(n, s, H, D, device=dev); k = torch.randn(n, s, H, D, device=dev); v = torch.randn(n, s, H, D, device=dev)
    o_old = bench(old_full, q, k, v, valid, s)
    o_new = bench(new_full, q, k, v, valid, s)
    print(f"{n:>3} {s:>5} {o_old:>10.3f}ms {o_new:>11.3f}ms {o_old/o_new:>7.2f}x")
