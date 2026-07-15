"""Find a flash_window variant that gives FINITE gradients AND correct forward.
The deployed version (seqused_q+k) has finite forward but NaN grads (padding-query
rows skipped by seqused_q -> NaN saved LSE -> NaN dk/dv on backward)."""
import torch, torch.nn.functional as F
from flash_attn.cute import flash_attn_varlen_func

dev = "cuda"
torch.manual_seed(0)
N, S, H, D, W = 4, 512, 4, 32, 64
scale = D ** -0.5
L = torch.randint(S // 2, S, (N,), device=dev).to(torch.int32)
valid = torch.arange(S, device=dev)[None, :] < L[:, None]
cu = torch.arange(0, (N + 1) * S, S, dtype=torch.int32, device=dev)
m = valid[..., None, None]
q0 = torch.randn(N, S, H, D, device=dev); k0 = torch.randn(N, S, H, D, device=dev); v0 = torch.randn(N, S, H, D, device=dev)


def sdpa_ref():  # correct reference (position/rank window, valid keys), padding rows 0
    rank = torch.cumsum(valid.long(), 1) - 1
    within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= W
    allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed = allowed | torch.eye(S, dtype=torch.bool, device=dev).unsqueeze(0)
    o = F.scaled_dot_product_attention(q0.transpose(1, 2).to(torch.bfloat16), k0.transpose(1, 2).to(torch.bfloat16),
                                       v0.transpose(1, 2).to(torch.bfloat16), attn_mask=allowed.unsqueeze(1), scale=scale).transpose(1, 2)
    return torch.where(m, o.float(), torch.zeros_like(o.float()))


def variant(tag, use_sq, zero_pad, nanfix):
    q = q0.clone().requires_grad_(True); k = k0.clone().requires_grad_(True); v = v0.clone().requires_grad_(True)
    qb, kb, vb = (t.reshape(N * S, H, D).to(torch.bfloat16) for t in (q, k, v))
    if zero_pad:
        mm = valid.reshape(N * S)[:, None, None]
        qb, kb, vb = (torch.where(mm, t, torch.zeros_like(t)) for t in (qb, kb, vb))
    kw = dict(seqused_k=L)
    if use_sq:
        kw["seqused_q"] = L
    o = flash_attn_varlen_func(qb, kb, vb, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S,
                               softmax_scale=scale, window_size=(W, W), **kw)
    o = o[0] if isinstance(o, tuple) else o
    o = o.reshape(N, S, H, D)
    if nanfix:
        o = torch.nan_to_num(o)
    out = torch.where(m, o, torch.zeros_like(o))
    fwd_ok = torch.isfinite(out).all().item()
    ferr = ((out.float() - sdpa_ref()).abs() * m).max().item()
    (out.float().sum()).backward()
    gok = all(torch.isfinite(g).all().item() for g in (q.grad, k.grad, v.grad))
    print(f"{tag:38s} fwd_finite={fwd_ok} fwd_err={ferr:6.3f} grad_finite={gok}")


variant("seqused_q+k (deployed)", True, False, False)
variant("seqused_q+k + nan_to_num", True, False, True)
variant("seqused_q+k + zero_pad_qkv", True, True, False)
variant("seqused_q+k + zero_pad + nan_to_num", True, True, True)
variant("seqused_k only", False, False, False)
print("\nWant: fwd_err~0 AND grad_finite=True")
