"""Experimental Triton F4--F7 training fusions, with the existing save contract.

A = existing F4 / fused F567. B = fused F45 / fused F67.
No production dispatch or shared tuning cache is modified. BF16 rounding of norm,
projection and gate logits matches the split path; saved gate is BF16, while the
forward epilogue consumes the FP32 sigmoid, just like gate_elem_train.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_materialize


@triton.jit
def _f567(XN, X, WP, WG, PROJ, GATE, Y, RES, DS,
          M, L: tl.constexpr, KP: tl.constexpr, KG: tl.constexpr, N: tl.constexpr,
          wp0: tl.constexpr, wp1: tl.constexpr,
          wg0: tl.constexpr, wg1: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          PROJECT: tl.constexpr):
    rm = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    if PROJECT:
        ap = tl.zeros((BM, BN), tl.float32)
        for k0 in range(triton.cdiv(KP, BK)):
            k = k0 * BK + rk
            a = tl.load(XN + rm[:, None] * KP + k[None, :],
                        (rm[:, None] < M) & (k[None, :] < KP), 0)
            w = tl.load(WP + k[:, None] * wp1 + rn[None, :] * wp0,
                        (k[:, None] < KP) & (rn[None, :] < N), 0)
            ap = tl.dot(a, w, ap)
        p = ap.to(PROJ.dtype.element_ty).to(tl.float32)
        tl.store(PROJ + rm[:, None] * N + rn[None, :], p,
                 (rm[:, None] < M) & (rn[None, :] < N))
    else:
        p = tl.load(PROJ + rm[:, None] * N + rn[None, :],
                    (rm[:, None] < M) & (rn[None, :] < N), 0).to(tl.float32)
    ag = tl.zeros((BM, BN), tl.float32)
    for k0 in range(triton.cdiv(KG, BK)):
        k = k0 * BK + rk
        a = tl.load(X + rm[:, None] * KG + k[None, :],
                    (rm[:, None] < M) & (k[None, :] < KG), 0)
        w = tl.load(WG + k[:, None] * wg0 + rn[None, :] * wg1,
                    (k[:, None] < KG) & (rn[None, :] < N), 0)
        ag = tl.dot(a, w, ag)
    g = tl.sigmoid(ag.to(X.dtype.element_ty).to(tl.float32))
    off = rm[:, None] * N + rn[None, :]
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    ds = tl.load(DS + (rm % L)[:, None] * N + rn[None, :], mask, 0).to(tl.float32)
    res = tl.load(RES + off, mask, 0).to(tl.float32)
    tl.store(Y + off, p * g * ds + res, mask)
    tl.store(GATE + off, g, mask)


@triton.jit
def _f45(X, GAMMA, BETA, WP, XN, MEAN, RSTD, PROJ,
         M, K: tl.constexpr, N: tl.constexpr, EPS: tl.constexpr,
         sx0: tl.constexpr, sx1: tl.constexpr,
         wp0: tl.constexpr, wp1: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, KPAD: tl.constexpr):
    rm = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    rk = tl.arange(0, KPAD)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rk[None, :] < K)
    x = tl.load(X + rm[:, None] * sx0 + rk[None, :] * sx1, mask, 0).to(tl.float32)
    mean = tl.sum(x, 1) / K
    xc = tl.where(mask, x - mean[:, None], 0.)
    var = tl.sum(xc * xc, 1) / K
    rstd = 1. / tl.sqrt(var + EPS)
    g = tl.load(GAMMA + rk, rk < K, 0).to(tl.float32)
    b = tl.load(BETA + rk, rk < K, 0).to(tl.float32)
    xn = tl.where(mask, xc * rstd[:, None] * g[None, :] + b[None, :], 0.)
    xn = xn.to(X.dtype.element_ty)
    # N tiles may repeat the LN computation, but only one owns the saved tensors.
    if tl.program_id(1) == 0:
        tl.store(XN + rm[:, None] * K + rk[None, :], xn, mask)
        tl.store(MEAN + rm, mean, rm < M)
        tl.store(RSTD + rm, rstd, rm < M)
    w = tl.load(WP + rk[:, None] * wp1 + rn[None, :] * wp0,
                (rk[:, None] < K) & (rn[None, :] < N), 0)
    p = tl.dot(xn, w)
    tl.store(PROJ + rm[:, None] * N + rn[None, :], p,
             (rm[:, None] < M) & (rn[None, :] < N))


# Explicit per-shape selections populated by the benchmark's exhaustive sweep.
# Keys contain actual M,K,N and input/weight strides (not dynamic shape buckets).
SELECTED = {}
REQUIRE_TUNED = False
SELECTION_LOG = {}


def config_key(kind, view, x, wp, wg):
    return (kind, tuple(view.shape), tuple(view.stride()), tuple(x.shape),
            tuple(wp.stride()), tuple(wg.stride()), str(view.dtype))


def candidates(kind):
    if kind == 'f45':
        return [dict(BM=m, BN=n, num_warps=w, num_stages=s)
                for m in (16, 32, 64, 128) for n in (32, 64, 128)
                for w in (4, 8) for s in (1, 2, 3)
                if not (m == 16 and w == 8)]
    return [dict(BM=m, BN=n, BK=k, num_warps=w, num_stages=s)
            for m in (16, 32, 64, 128) for n in (32, 64, 128)
            for k in (32, 64) for w in (4, 8) for s in (2, 3, 4)
            if not (m == 16 and w == 8)]


def buffers(view, wp):
    m, k = view.shape
    n = wp.shape[0]
    return (view.new_empty((m, n)), view.new_empty((m, n)),
            view.new_empty((m, k)), view.new_empty((m,), dtype=torch.float32),
            view.new_empty((m,), dtype=torch.float32), view.new_empty((m, n)))


def launch(kind, view, x, gamma, beta, wp, wg, residual, ds, eps, length, out, config=None):
    y, proj, norm, mean, rstd, gate = out
    key = config_key(kind, view, x, wp, wg)
    cfg = config or SELECTED.get(key)
    if config is None:
        if REQUIRE_TUNED and cfg is None:
            raise RuntimeError(f'Untuned runtime geometry: {key}')
        SELECTION_LOG[repr(key)] = dict(config=cfg, hit=cfg is not None)
    if cfg is None:
        cfg = dict(BM=32, BN=128, num_warps=4, num_stages=2)
        if kind != 'f45':
            cfg['BK'] = 32
    grid = (triton.cdiv(view.shape[0], cfg['BM']), triton.cdiv(wp.shape[0], cfg['BN']))
    if kind == 'f45':
        return _f45[grid](view, gamma, beta, wp, norm, mean, rstd, proj,
                         view.shape[0], view.shape[1], wp.shape[0], eps,
                         *view.stride(), *wp.stride(),
                         KPAD=triton.next_power_of_2(view.shape[1]), **cfg)
    return _f567[grid](norm, x, wp, wg, proj, gate, y, residual, ds,
                      view.shape[0], length, view.shape[1], x.shape[1], wp.shape[0],
                      *wp.stride(), *wg.stride(), PROJECT=kind == 'f567', **cfg)


def _fake(view, x, gamma, beta, wp, wg, residual, ds, eps, length, variant):
    return buffers(view, wp)


@opaque(fake=_fake, name='experimental_trimul_output_fusion')
def output_forward(view: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor,
                   beta: torch.Tensor, wp: torch.Tensor, wg: torch.Tensor,
                   residual: torch.Tensor, ds: torch.Tensor, eps: float,
                   length: int, variant: str
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                              torch.Tensor, torch.Tensor, torch.Tensor]:
    assert variant in ('A', 'B')
    assert view.dtype == x.dtype == wp.dtype == wg.dtype == torch.bfloat16
    assert x.is_contiguous() and residual.is_contiguous() and ds.is_contiguous()
    out = buffers(view, wp)
    if variant == 'A':
        norm, mean, rstd = _ln_materialize(view, gamma, beta, eps)
        out = (out[0], out[1], norm, mean, rstd, out[5])
        launch('f567', view, x, gamma, beta, wp, wg, residual, ds, eps, length, out)
    else:
        launch('f45', view, x, gamma, beta, wp, wg, residual, ds, eps, length, out)
        launch('f67', view, x, gamma, beta, wp, wg, residual, ds, eps, length, out)
    return out
