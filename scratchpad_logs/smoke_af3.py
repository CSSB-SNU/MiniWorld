"""End-to-end smoke of the DEPLOYED af3 SWA attention path (the one mpfull uses):
import + RoPESWAAttentionAF3.forward with the new static attention_params.
"""
import torch
from team_gm.modules.blocks.rope_swa_af3_transformer import RoPESWAAttentionAF3
from team_gm.modules.layers.swa_atom_attention import build_3d_rope, build_attention_params
import miniworld.modules.diffusion_module  # ensure the whole import chain is clean  # noqa: F401

dev = "cuda"
torch.manual_seed(0)
B, S, d, n_head = 1, 3072, 128, 4        # per-rank batch 1, atom bucket, c_atom=128
num_aug = 4
N = num_aug * B

attn = RoPESWAAttentionAF3(d_single=d, d_cond=d, n_head=n_head, half_window=64,
                           use_qk_norm=True, use_rotation=True, mp_full=True).to(dev)

# front-packed valid, flattened to [N, S]
L = torch.randint(S // 2, S, (N,), device=dev)
valid = torch.arange(S, device=dev)[None, :] < L[:, None]
ref_pos = torch.randn(B, S, 3, device=dev)
ref_uid = torch.zeros(B, S, dtype=torch.long, device=dev)
cos, sin = build_3d_rope(ref_pos, ref_uid, head_dim=d // n_head,
                         n_spatial_per_axis=2, n_uid_pairs=10)
attn_params = build_attention_params(cos, sin, valid, num_aug)

single = torch.randn(N, S, d, device=dev)
cond = torch.randn(N, S, d, device=dev)
with torch.no_grad():
    out = attn(single, cond, attn_params)
print("af3 attention output:", tuple(out.shape), out.dtype,
      "finite:", bool(torch.isfinite(out).all()))
# padding rows must be ~0 (flash_window_seqused zeros them)
pad = ~valid
print("max|out| on padding rows:", out[pad].abs().max().item(), "(expect ~0)")
print("PASS" if out.shape == (N, S, d) and torch.isfinite(out).all() else "FAIL")
