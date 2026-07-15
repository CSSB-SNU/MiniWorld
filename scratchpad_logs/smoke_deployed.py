"""End-to-end smoke test of the DEPLOYED swa_atom_attention module: build_attention_params
(new static seqused params) + flash_window_seqused, vs old packed varlen + SDPA band.
"""
import torch, torch.nn.functional as F
from flash_attn.cute import flash_attn_varlen_func
from team_gm.modules.layers import swa_atom_attention as M

dev = "cuda"
torch.manual_seed(0)
N, S, H, D, W = 4, 512, 4, 32, 64
scale = D ** -0.5
L = torch.randint(S // 2, S, (N,), device=dev).to(torch.int32)
valid = torch.arange(S, device=dev)[None, :] < L[:, None]  # front-packed
q = torch.randn(N, S, H, D, device=dev)
k = torch.randn(N, S, H, D, device=dev)
v = torch.randn(N, S, H, D, device=dev)
cos = torch.zeros(N, S, D // 2, device=dev); sin = torch.zeros_like(cos)  # unused by attn core

# DEPLOYED path
_, _, seqused, cu, max_s, val = M.build_attention_params(cos, sin, valid, num_aug=1)
new = M.flash_window_seqused(q, k, v, cu, seqused, max_s, val, N, S, scale, W).float()

# OLD packed varlen (reference the code used to run)
idx = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
cup = F.pad(torch.cumsum(valid.sum(-1, dtype=torch.int32), 0, dtype=torch.int32), (1, 0))
qb, kb, vb = (t.reshape(N * S, H, D).to(torch.bfloat16)[idx] for t in (q, k, v))
o = flash_attn_varlen_func(qb, kb, vb, cu_seqlens_q=cup, cu_seqlens_k=cup,
                           max_seqlen_q=S, max_seqlen_k=S, softmax_scale=scale, window_size=(W, W))
o = o[0] if isinstance(o, tuple) else o
old = o.new_zeros(N * S, H, D); old[idx] = o; old = old.view(N, S, H, D).float()

m = valid.view(N, S, 1, 1)
print(f"build_attention_params returns: seqused.shape={tuple(seqused.shape)} "
      f"cu_seqlens={cu.tolist()[:4]}... max_seqlen={max_s}")
print(f"max|DEPLOYED_new - OLD_packed| on valid = {((new - old).abs() * m).max().item():.6f}")
print("PASS" if ((new - old).abs() * m).max().item() < 0.05 else "FAIL")
