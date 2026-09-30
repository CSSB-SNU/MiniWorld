"""Smoke + numeric tests for EDM2 magnitude-preservation building blocks.

Covers: magnitude_normalize / MPLinear (forced weight norm), mp_sum
(variance preservation), apply_pairwise_rotation (norm preservation),
AdaptiveLayerNorm rotation modulation and the MP pair-bias projection. The
DiffusionTransformer MP/rotation options were removed in team-gm 26eeba9
(2026-08-03), so the block-level MP tests went with them.
"""

import math

import torch

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


def test_pair_bias_projection_is_mp_under_flag():
    from team_gm.modules.layers.augmented_attention import AugmentedAttentionPairBias

    plain = AugmentedAttentionPairBias(16, 8, 4, 2, magnitude_preserving=False)
    assert not isinstance(plain.to_bias, MPLinear)  # original zero-init Linear
    mp = AugmentedAttentionPairBias(16, 8, 4, 2, magnitude_preserving=True)
    assert isinstance(mp.to_bias, MPLinear)  # pair-bias projection now forced-WN
    rows = torch.linalg.vector_norm(mp.to_bias.weight, dim=1)
    assert torch.allclose(rows, torch.full_like(rows, math.sqrt(mp.to_bias.in_features)), atol=5e-2)


def test_mp_swish_gate_preserves_variance():
    a = torch.randn(200_000)
    b = torch.randn(200_000)
    assert abs(mp_swish_gate(a, b).var().item() - 1.0) < 0.03, mp_swish_gate(a, b).var().item()


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
