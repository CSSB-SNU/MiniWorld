"""Verify the new seqused_k SWA attention == the old varlen (unpad/pack) path.

Both paths use flash_attn_varlen_func. OLD unpads via nonzero+gather, windows over
packed order (== valid-atom rank), scatters back. NEW keeps full [N,S], fixed-stride
cu_seqlens + seqused_k, windows over position. They are IDENTICAL iff valid is
front-packed (position == rank). This test proves:
  (1) front-packed: max|OLD - NEW| ~ 0  (implementation correct)
  (2) gappy:        OLD != NEW           (front-packing is a real precondition)
Also checks both against an SDPA rank-band reference on front-packed input.
"""
import torch
import torch.nn.functional as F
from flash_attn.cute import flash_attn_varlen_func


def old_varlen(q, k, v, valid, scale, w):
    n, s, h, d = q.shape
    seqlens = valid.sum(-1, dtype=torch.int32)
    indices = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
    cu = F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))
    qb, kb, vb = (t.reshape(n * s, h, d).to(torch.bfloat16) for t in (q, k, v))
    qu, ku, vu = qb[indices], kb[indices], vb[indices]
    out = flash_attn_varlen_func(qu, ku, vu, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                 max_seqlen_q=s, max_seqlen_k=s, softmax_scale=scale,
                                 window_size=(w, w))
    if isinstance(out, tuple):
        out = out[0]
    full = out.new_zeros(n * s, h, d)
    full[indices] = out
    return full.view(n, s, h, d).float()


def new_seqused(q, k, v, valid, scale, w):
    n, s, h, d = q.shape
    seqused = valid.sum(-1, dtype=torch.int32)
    cu = torch.arange(0, (n + 1) * s, s, dtype=torch.int32, device=q.device)
    qb, kb, vb = (t.reshape(n * s, h, d).to(torch.bfloat16) for t in (q, k, v))
    out = flash_attn_varlen_func(qb, kb, vb, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                 max_seqlen_q=s, max_seqlen_k=s,
                                 seqused_q=seqused, seqused_k=seqused,
                                 softmax_scale=scale, window_size=(w, w))
    if isinstance(out, tuple):
        out = out[0]
    out = out.reshape(n, s, h, d)
    return (out * valid.unsqueeze(-1).unsqueeze(-1)).float()


def sdpa_band(q, k, v, valid, scale, w):
    n, s, h, d = q.shape
    rank = torch.cumsum(valid.long(), 1) - 1
    within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= w
    allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed = allowed | torch.eye(s, dtype=torch.bool, device=q.device).unsqueeze(0)
    o = F.scaled_dot_product_attention(q.transpose(1, 2).to(torch.bfloat16),
                                       k.transpose(1, 2).to(torch.bfloat16),
                                       v.transpose(1, 2).to(torch.bfloat16),
                                       attn_mask=allowed.unsqueeze(1), scale=scale).transpose(1, 2)
    return (o.float() * valid.unsqueeze(-1).unsqueeze(-1))


def front_packed_valid(n, s, dev, g):
    L = torch.randint(s // 2, s, (n,), generator=g).to(dev)
    return torch.arange(s, device=dev).unsqueeze(0) < L.unsqueeze(1)


def gappy_valid(n, s, dev, g):
    return (torch.rand(n, s, generator=g).to(dev) < 0.7)


def run(tag, valid, dev):
    n, s = valid.shape
    h, d, w = 4, 32, 64
    scale = d ** -0.5
    g = torch.Generator().manual_seed(1)
    q = torch.randn(n, s, h, d, generator=g).to(dev)
    k = torch.randn(n, s, h, d, generator=g).to(dev)
    v = torch.randn(n, s, h, d, generator=g).to(dev)
    o_old = old_varlen(q, k, v, valid, scale, w)
    o_new = new_seqused(q, k, v, valid, scale, w)
    m = valid.unsqueeze(-1).unsqueeze(-1)
    diff = ((o_old - o_new).abs() * m).max().item()
    print(f"[{tag}] N={n} S={s}  max|OLD - NEW| on valid = {diff:.5f}")
    if tag == "front-packed":
        ref = sdpa_band(q, k, v, valid, scale, w)
        print(f"[{tag}] max|NEW - SDPA_band|            = {((o_new - ref).abs() * m).max().item():.5f}")
        print(f"[{tag}] max|OLD - SDPA_band|            = {((o_old - ref).abs() * m).max().item():.5f}")


if __name__ == "__main__":
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    run("front-packed", front_packed_valid(8, 512, dev, g), dev)
    run("gappy", gappy_valid(8, 512, dev, g), dev)
    print("\nExpect: front-packed diff ~0 (bf16), gappy diff LARGE (proves front-packing needed).")
