"""Tests for sliding-window + 3D RoPE + QK-norm atom attention.

Runs on CPU with the ``sdpa`` backend so it needs no GPU / flash-attn.
"""

from __future__ import annotations

import pytest
import torch

from miniworld.configs.models import AtomSWAConfig, SharedConfig
from miniworld.modules.swa_rope_attention import (
    _FLASH_AVAILABLE,
    SlidingWindowAttention,
    SWAAtomTransformer,
    build_3d_rope,
    build_attention_params,
)

ROPE = {
    "n_spatial_per_axis": 2,
    "n_uid_pairs": 10,
    "spatial_base_freq": 20.0,
    "uid_base_freq": 10000.0,
}


def test_qk_norm_is_unit_rms() -> None:
    attn = SlidingWindowAttention(d_single=128, n_head=4, half_window=8, backend="sdpa")
    x = torch.randn(2, 5, 4, 32) * 3.0  # [B, L, H, D]
    y = attn.norm_query(x)
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_3d_rope_shapes_and_padding() -> None:
    B, L, d_hidden = 2, 7, 32
    cos, sin = build_3d_rope(
        torch.randn(B, L, 3), torch.zeros(B, L, dtype=torch.long), d_hidden, **ROPE,
    )
    # 3*2 spatial + 10 uid = 16 = d_hidden / 2 -> exactly filled, no padding.
    assert cos.shape == (B, L, d_hidden // 2)
    assert sin.shape == (B, L, d_hidden // 2)


def test_sliding_window_masks_out_of_window() -> None:
    """A query well outside the window must be unaffected by a far-away spike."""
    torch.manual_seed(0)
    B, L, d, nh = 1, 64, 32, 4
    half_window = 4
    attn = SlidingWindowAttention(d, nh, half_window=half_window, backend="sdpa").eval()
    # to_out is zero-init by convention; make it non-trivial so the attention
    # output (and thus the windowing) is observable.
    torch.nn.init.normal_(attn.to_out.weight)
    ref = torch.randn(B, L, 3)
    uid = torch.zeros(B, L, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    # head_dim = 8 -> d_hidden/2 = 4, so keep active freqs <= 4.
    ap = build_attention_params(
        ref, uid, mask, d // nh,
        n_spatial_per_axis=1, n_uid_pairs=1,
        spatial_base_freq=20.0, uid_base_freq=10000.0,
    )
    x = torch.zeros(B, L, d)
    x[0, 0] = 1.0
    with torch.no_grad():
        out = attn(x, ap)
    x2 = x.clone()
    x2[0, 0] = 5.0
    with torch.no_grad():
        out2 = attn(x2, ap)
    assert torch.allclose(out[0, 40], out2[0, 40], atol=1e-5)  # far: unaffected
    assert not torch.allclose(out[0, 2], out2[0, 2], atol=1e-3)  # near: affected


def test_transformer_forward_backward_and_padding_grad() -> None:
    torch.manual_seed(0)
    B, L, d, nh = 2, 48, 128, 4
    ref = torch.randn(B, L, 3)
    uid = torch.zeros(B, L, dtype=torch.long)
    uid[:, 24:] = 1
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 44:] = False
    ap = build_attention_params(ref, uid, mask, d // nh, **ROPE)
    tr = SWAAtomTransformer(d_atom=d, n_blocks=2, n_heads=nh, swa_window_size=8, backend="sdpa")
    q = torch.randn(B, L, d, requires_grad=True)
    c = torch.randn(B, L, d)
    out = tr(q, c, ap)
    assert out.shape == (B, L, d)
    out.sum().backward()
    assert torch.isfinite(q.grad).all()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _FLASH_AVAILABLE),
    reason="needs CUDA + flash-attn-4 (FA4, Hopper/Blackwell)",
)
def test_flash_matches_sdpa_on_gpu() -> None:
    """FA4 flash backend == sdpa banded backend (same window + per-space)."""
    dev = "cuda"
    torch.manual_seed(0)
    B, L, d, nh = 2, 300, 128, 4
    ref = torch.randn(B, L, 3, device=dev)
    uid = torch.zeros(B, L, dtype=torch.long, device=dev)
    uid[:, 128:] = 1
    uid[:, 200:] = 2
    mask = torch.ones(B, L, dtype=torch.bool, device=dev)
    mask[0, 290:] = False
    ap = build_attention_params(ref, uid, mask, d // nh, **ROPE)
    flash = SWAAtomTransformer(d_atom=d, n_blocks=2, n_heads=nh, swa_window_size=64, backend="flash").to(dev)
    sdpa = SWAAtomTransformer(d_atom=d, n_blocks=2, n_heads=nh, swa_window_size=64, backend="sdpa").to(dev)
    sdpa.load_state_dict(flash.state_dict())
    q = torch.randn(B, L, d, device=dev)
    c = torch.randn(B, L, d, device=dev)
    with torch.no_grad():
        of = flash(q, c, ap)
        os = sdpa(q, c, ap)
    assert (of.float() - os.float()).abs().max().item() < 1e-2


def test_disabled_config_is_default() -> None:
    assert AtomSWAConfig().enabled is False
    shared = SharedConfig()
    d_hidden = shared.d_single_atom // 4
    cfg = AtomSWAConfig()
    active = 3 * cfg.n_spatial_rope_pairs_per_axis + cfg.n_uid_rope_pairs
    assert active <= d_hidden // 2
