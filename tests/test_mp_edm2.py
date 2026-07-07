"""Smoke + numeric tests for EDM2 magnitude-preservation building blocks.

Covers: magnitude_normalize / MPLinear (forced weight norm), mp_sum
(variance preservation), apply_pairwise_rotation (norm preservation),
AdaptiveLayerNorm rotation modulation, and a full DiffusionTransformer forward
with the MP flags on (PyTorch path) + a training step that keeps ||w|| pinned.
"""

import math

import torch

from team_gm.modules.blocks.diffusion_transformer import DiffusionTransformer
from team_gm.modules.layers.adaln import AdaptiveLayerNorm
from team_gm.modules.layers.ops import apply_pairwise_rotation, mp_sum, mp_swish_gate
from team_gm.modules.primitives import (
    Linear,
    MPLinear,
    convert_linears_to_mp,
    magnitude_normalize,
)


def test_magnitude_normalize_rows_to_sqrt_fan_in():
    # EDM2's normalize constrains each output-channel row to norm sqrt(fan_in)
    # (= sqrt(Nj), the "weights on a hypersphere of radius sqrt(Nj)" rule).
    fan_in = 17
    w = torch.randn(32, fan_in) * 5.0
    n = magnitude_normalize(w)
    row_norms = torch.linalg.vector_norm(n, dim=1)
    target = math.sqrt(fan_in)
    assert torch.allclose(row_norms, torch.full_like(row_norms, target), atol=1e-2)


def test_mplinear_forced_weight_norm_pins_norm():
    torch.manual_seed(0)
    fan_in = 16
    lin = MPLinear(fan_in, 24, bias=False, init="normal")
    lin.train()
    opt = torch.optim.Adam(lin.parameters(), lr=1e-1)  # big LR to force drift
    for _ in range(20):
        x = torch.randn(8, fan_in)
        lin(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    # Forced WN pins stored rows to norm sqrt(fan_in) -> no upward drift.
    row_norms = torch.linalg.vector_norm(lin.weight, dim=1)
    target = math.sqrt(fan_in)
    assert torch.allclose(row_norms, torch.full_like(row_norms, target), atol=5e-2), (
        row_norms.min().item(),
        row_norms.max().item(),
    )


def test_mplinear_rejects_zero_init():
    try:
        MPLinear(4, 4, init="zero")
    except ValueError:
        return
    raise AssertionError("MPLinear should reject init='zero'")


def test_mp_sum_preserves_variance():
    a = torch.randn(100_000)
    b = torch.randn(100_000)  # uncorrelated, unit variance
    out = mp_sum(a, b, t=0.3)
    assert abs(out.var().item() - 1.0) < 0.02, out.var().item()


def test_rotation_preserves_norm():
    x = torch.randn(4, 10)
    theta = torch.randn(4, 5)
    y = apply_pairwise_rotation(x, theta)
    assert torch.allclose(
        torch.linalg.vector_norm(x, dim=-1),
        torch.linalg.vector_norm(y, dim=-1),
        atol=1e-5,
    )


def test_adaln_rotation_identity_at_init():
    # to_angle is zero-init -> rotation is identity at init, so output equals
    # the scaled (no-shift) path.
    torch.manual_seed(0)
    ln = AdaptiveLayerNorm(d_hidden=8, d_cond=6, use_rotation=True)
    ln.eval()
    x = torch.randn(3, 8)
    cond = torch.randn(3, 6)
    out = ln(x, cond)
    scaled = torch.sigmoid(ln.to_scale(ln.ln_cond(cond))) * ln.ln_in(x)
    assert torch.allclose(out, scaled, atol=1e-5)


def _cfg(**kw):
    return DiffusionTransformer.Config(
        d_single=16, d_cond=8, d_pair=4, n_head=2, n_block=2, **kw
    )


def test_pair_bias_projection_is_mp_under_flag():
    from team_gm.modules.layers.augmented_attention import AugmentedAttentionPairBias

    plain = AugmentedAttentionPairBias(16, 8, 4, 2, magnitude_preserving=False)
    assert not isinstance(plain.to_bias, MPLinear)  # original zero-init Linear
    mp = AugmentedAttentionPairBias(16, 8, 4, 2, magnitude_preserving=True)
    assert isinstance(mp.to_bias, MPLinear)  # pair-bias projection now forced-WN
    rows = torch.linalg.vector_norm(mp.to_bias.weight, dim=1)
    assert torch.allclose(rows, torch.full_like(rows, math.sqrt(mp.to_bias.in_features)), atol=5e-2)


def test_diffusion_transformer_baseline_and_mp_forward():
    torch.manual_seed(0)
    single = torch.randn(1, 1, 5, 16)
    cond = torch.randn(1, 1, 5, 8)
    pair = torch.randn(1, 5, 5, 4)
    for kw in (
        {},  # backward-compatible default
        {"magnitude_preserving": True, "use_rotation": True, "mp_residual": True},
    ):
        model = _cfg(**kw).build() if hasattr(_cfg(**kw), "build") else DiffusionTransformer(_cfg(**kw))
        model.train()
        out = model(single, cond, pair)
        assert out.shape == single.shape
        assert torch.isfinite(out).all()


def test_diffusion_transformer_mp_training_step_keeps_norm_bounded():
    torch.manual_seed(0)
    model = DiffusionTransformer(
        _cfg(magnitude_preserving=True, use_rotation=True, mp_residual=True)
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    mp_rows_max = []
    for _ in range(15):
        single = torch.randn(1, 1, 5, 16)
        cond = torch.randn(1, 1, 5, 8)
        pair = torch.randn(1, 5, 5, 4)
        model(single, cond, pair).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    # Every MPLinear weight stays pinned to sqrt(fan_in) despite the aggressive
    # LR (forced WN), i.e. ||w|| does not drift up over training.
    for m in model.modules():
        if isinstance(m, MPLinear):
            rn = torch.linalg.vector_norm(m.weight, dim=1)
            target = math.sqrt(m.in_features)
            mp_rows_max.append((rn.max() / target).item())
    assert mp_rows_max, "no MPLinear found under magnitude_preserving=True"
    assert max(mp_rows_max) < 1.05, max(mp_rows_max)


# ---------------- full-MP (mp_full) ----------------

def test_mp_swish_gate_preserves_variance():
    a = torch.randn(200_000)
    b = torch.randn(200_000)
    assert abs(mp_swish_gate(a, b).var().item() - 1.0) < 0.03, mp_swish_gate(a, b).var().item()


def test_mp_full_makes_every_block_linear_mp():
    torch.manual_seed(0)
    cfg = DiffusionTransformer.Config(
        d_single=16, d_cond=8, d_pair=4, n_head=2, n_block=1,
        mp_full=True, use_rotation=True,
    )
    model = DiffusionTransformer(cfg)
    # mp_full implies mp_residual
    assert model.blocks[0].mp_residual is True
    # every Linear under the block must be MPLinear (no plain Linear left)
    plain = [
        n for n, m in model.named_modules()
        if isinstance(m, Linear) and not isinstance(m, MPLinear)
    ]
    assert not plain, f"plain Linears remain under mp_full: {plain}"
    # spot-check the previously zero/gating-init layers are now MP
    blk = model.blocks[0]
    assert isinstance(blk.attention_pair_bias.to_out, MPLinear)
    assert isinstance(blk.attention_pair_bias.to_gate, MPLinear)
    assert isinstance(blk.transition.squeeze, MPLinear)
    assert isinstance(blk.transition.ada_ln_in.to_scale, MPLinear)
    assert isinstance(blk.attention_pair_bias.ada_ln_in.to_angle, MPLinear)


def test_mp_full_forward_and_norm_pinned():
    torch.manual_seed(0)
    cfg = DiffusionTransformer.Config(
        d_single=16, d_cond=8, d_pair=4, n_head=2, n_block=2,
        mp_full=True, use_rotation=True,
    )
    model = DiffusionTransformer(cfg)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    for _ in range(12):
        single = torch.randn(1, 1, 5, 16)
        cond = torch.randn(1, 1, 5, 8)
        pair = torch.randn(1, 5, 5, 4)
        out = model(single, cond, pair)
        assert out.shape == single.shape and torch.isfinite(out).all()
        out.pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    for m in model.modules():
        if isinstance(m, MPLinear):
            rn = torch.linalg.vector_norm(m.weight, dim=1)
            assert (rn.max() / math.sqrt(m.in_features)) < 1.05


def test_convert_linears_to_mp_excludes_final_denoising():
    import torch.nn as nn
    root = nn.Module()
    root.proj = Linear(8, 8, init="default")
    root.final_denoising = nn.Sequential(nn.LayerNorm(8), Linear(8, 3, bias=False, init="zero"))
    n = convert_linears_to_mp(root, exclude_substrings=("final_denoising",))
    assert n == 1  # only root.proj converted
    assert isinstance(root.proj, MPLinear)
    assert isinstance(root.final_denoising[1], Linear)
    assert not isinstance(root.final_denoising[1], MPLinear)  # excluded
