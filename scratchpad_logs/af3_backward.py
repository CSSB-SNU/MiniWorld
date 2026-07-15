"""Reproduce the training NaN-grad through the real af3 block WITH backward.
Tests several valid-length configs incl. full rows (L=S) and tiny L."""
import torch
from team_gm.modules.blocks.rope_swa_af3_transformer import RoPESWAAttentionAF3
from team_gm.modules.layers.swa_atom_attention import build_3d_rope, build_attention_params

dev = "cuda"
torch.manual_seed(0)
S, d, n_head, num_aug = 3072, 128, 4, 4
B = 1
N = num_aug * B
attn = RoPESWAAttentionAF3(d_single=d, d_cond=d, n_head=n_head, half_window=64,
                           use_qk_norm=True, use_rotation=True, mp_full=True).to(dev)


def run(tag, Lvals):
    L = torch.tensor(Lvals, device=dev)
    valid = torch.arange(S, device=dev)[None, :] < L[:, None]
    ref_pos = torch.randn(B, S, 3, device=dev)
    ref_uid = torch.zeros(B, S, dtype=torch.long, device=dev)
    cos, sin = build_3d_rope(ref_pos, ref_uid, head_dim=d // n_head, n_spatial_per_axis=2, n_uid_pairs=10)
    ap = build_attention_params(cos, sin, valid, num_aug)
    single = torch.randn(N, S, d, device=dev, requires_grad=True)
    cond = torch.randn(N, S, d, device=dev, requires_grad=True)
    for p in attn.parameters():
        if p.grad is not None:
            p.grad = None
    out = attn(single, cond, ap)
    loss = out.float().pow(2).mean()
    loss.backward()
    pg = [p.grad for p in attn.parameters() if p.grad is not None]
    param_ok = all(torch.isfinite(g).all().item() for g in pg)
    in_ok = torch.isfinite(single.grad).all().item() and torch.isfinite(cond.grad).all().item()
    print(f"{tag:28s} out_finite={torch.isfinite(out).all().item()} "
          f"param_grad_finite={param_ok} input_grad_finite={in_ok}")


run("mixed L", [2601, 2264, 1992, 2968])
run("one full row (L=S)", [S, 2264, 1992, 976])
run("all full (L=S)", [S, S, S, S])
run("tiny L", [8, 40, 100, 2968])
run("one L < window", [30, 2264, 1992, 976])
print("\nany param_grad_finite=False -> reproduced the training NaN")
