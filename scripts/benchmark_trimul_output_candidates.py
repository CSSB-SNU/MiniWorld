"""Same-GPU, same-input output-LN/projection training candidate audit."""

import argparse
import json
from pathlib import Path
import statistics

import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.layernorm_linear.triton.te_style import (
    _te_forward,
    _te_backward,
    _ln_materialize,
)
from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear import (
    layernorm_linear_cute,
)
from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_engine.kernels.layernorm_linear import autograd as ag
from miniworld_engine.kernels.layernorm_linear.cute import dgrad_lnbwd as dg

try:
    from miniworld_engine.kernels.layernorm_linear.cute.dgrad_ln_full import (
        dgrad_ln_full,
    )
except ImportError:
    # Rejected prototype is retained in the experiment's prototypes directory.
    dgrad_ln_full = None
from miniworld_engine.autotune.cute_config import lnbwd_candidates
from miniworld_engine.kernels.layernorm_linear.cute.dgrad_ln_rows import dgrad_ln_rows
from miniworld_engine.autotune.cute_config import plain_sm90_candidates

FULL_CONFIG = None
ROWS_CONFIG = None


def rows_fake(dy, w, xhat, g, rstd, c1, c2):
    return torch.empty_strided(
        xhat.shape, (1, xhat.shape[0]), device=xhat.device, dtype=xhat.dtype
    )


@opaque(fake=rows_fake, name="trimul_output_probe_rows_backward")
def rows_backward(
    dy: torch.Tensor,
    w: torch.Tensor,
    xhat: torch.Tensor,
    g: torch.Tensor,
    rstd: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
) -> torch.Tensor:
    return dgrad_ln_rows(dy, w, xhat, g, rstd, c1, c2, config=ROWS_CONFIG)


def full_fake(dy, w, x, g, mean, rstd):
    cfg = FULL_CONFIG or lnbwd_candidates(w.shape[1])[0]
    partial = x.new_empty(
        (1, (x.shape[0] + cfg.tile_m - 1) // cfg.tile_m, x.shape[1]), dtype=torch.float32
    )
    return (
        torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype),
        partial,
        torch.empty_like(partial),
    )


@opaque(fake=full_fake, name="trimul_output_probe_full_backward")
def full_backward(
    dy: torch.Tensor,
    w: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return dgrad_ln_full(dy, w, x, g, mean, rstd, config=FULL_CONFIG)


def m1_fake(x, g, b, w, eps):
    return (
        x.new_empty((x.shape[0], w.shape[0])),
        x.new_empty((x.shape[0],), dtype=torch.float32),
        x.new_empty((x.shape[0],), dtype=torch.float32),
    )


@opaque(fake=m1_fake, name="trimul_output_probe_m1")
def m1(
    x: torch.Tensor, g: torch.Tensor, b: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return layernorm_linear_cute(x, g, b, w, None, eps, return_stats=True)


@opaque(fake=m1_fake, name="trimul_output_probe_m2")
def m2(
    x: torch.Tensor, g: torch.Tensor, b: torch.Tensor, w: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return layernorm_linear_cute_fused(x, g, b, w, None, eps, return_stats=True)


original_dgrad = dg.dgrad_lnbwd_cute


def dg_fake(dy, w, xhat, g, rstd):
    return xhat.new_empty(xhat.shape)


@opaque(fake=dg_fake, name="trimul_output_probe_dgrad")
def dgrad(
    dy: torch.Tensor,
    w: torch.Tensor,
    xhat: torch.Tensor,
    g: torch.Tensor,
    rstd: torch.Tensor,
) -> torch.Tensor:
    return original_dgrad(dy, w, xhat, g, rstd)


dg.dgrad_lnbwd_cute = dgrad


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def paired(graphs):
    samples = {k: [] for k in graphs}
    for rep in range(10):
        for name in list(graphs)[:: 1 if rep % 2 else -1]:
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(25):
                graphs[name].replay()
            b.record()
            b.synchronize()
            samples[name].append(a.elapsed_time(b) / 25)
    return {k: {"ms": statistics.median(v), "samples_ms": v} for k, v in samples.items()}


def main():
    global FULL_CONFIG, ROWS_CONFIG
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--names", default="te,normalized_rows")
    p.add_argument("--config", type=int, default=0)
    p.add_argument("--rows-config", type=int, default=0)
    args = p.parse_args()
    M = args.length**2
    K, N = 256, 128
    FULL_CONFIG = lnbwd_candidates(K)[args.config]
    ROWS_CONFIG = plain_sm90_candidates()[args.rows_config]
    torch.manual_seed(891)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(K, M, device="cuda", dtype=torch.bfloat16).t()
    g = (1 + torch.randn(K, device="cuda") * 0.15).bfloat16()
    b = (torch.randn(K, device="cuda") * 0.1).bfloat16()
    w = (torch.randn(N, K, device="cuda") * 0.03).bfloat16()
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    eps = 1e-5

    def te():
        y, xn, mean, rstd = _te_forward(x, g, b, w, None, eps)
        grads = _te_backward(dy, xn, x, mean, rstd, g, w, False)
        return (y,) + grads[:4]

    def native_fused():
        y, mean, rstd = m1(x, g, b, w, eps)
        grads = ag._compose_backward_fused(dy, x, mean, rstd, g, b, w, False)
        return (y, grads[0].t().contiguous().t()) + grads[1:4]

    def native_composed():
        y, mean, rstd = m1(x, g, b, w, eps)
        # Preserve the existing stride-aware LN backward, recompute normalized activation.
        xn = ag._recompute_xnormed(x, g, b, mean, rstd)
        grads = _te_backward(dy, xn, x, mean, rstd, g, w, False)
        return (y,) + grads[:4]

    def m2_fused():
        y, mean, rstd = m2(x, g, b, w, eps)
        grads = ag._compose_backward_fused(dy, x, mean, rstd, g, b, w, False)
        return (y, grads[0].t().contiguous().t()) + grads[1:4]

    def m2_composed():
        y, mean, rstd = m2(x, g, b, w, eps)
        xn = ag._recompute_xnormed(x, g, b, mean, rstd)
        grads = _te_backward(dy, xn, x, mean, rstd, g, w, False)
        return (y,) + grads[:4]

    def te_full():
        y, xn, mean, rstd = _te_forward(x, g, b, w, None, eps)
        dx, dg, db = full_backward(dy, w, x, g, mean, rstd)
        dw = dy.t() @ xn
        return y, dx, dg.sum((0, 1)).to(g.dtype), db.sum((0, 1)).to(g.dtype), dw

    def projected_rows(forward):
        wf, gf, bf = w.float(), g.float(), b.float()
        s = (wf * gf[None, :]).sum(1)
        b2 = (wf * bf[None, :]).sum(1)
        if forward is None:
            xhat, mean, rstd = _ln_materialize(
                x, torch.ones_like(g), torch.zeros_like(b), eps
            )
            y = torch.nn.functional.linear(
                xhat, (wf * gf[None, :]).to(w.dtype), b2.to(w.dtype)
            )
        else:
            y, mean, rstd = forward(x, g, b, w, eps)
            xhat = ag._recompute_xhat(x, mean, rstd)
        c1 = (dy.float() * (y.float() - b2[None, :])).sum(1) / K
        c2 = (dy.float() * s[None, :]).sum(1) / K
        dx = rows_backward(dy, w, xhat, g, rstd, c1, c2)
        t = (dy.t() @ xhat).float()
        db = dy.float().sum(0)
        dw = (gf[None, :] * t + db[:, None] * bf[None, :]).to(w.dtype)
        dg = (wf * t).sum(0).to(g.dtype)
        dbeta = (db @ wf).to(b.dtype)
        return y, dx, dg, dbeta, dw

    def m1_rows():
        return projected_rows(m1)

    def m2_rows():
        return projected_rows(m2)

    def normalized_rows():
        return projected_rows(None)

    def production():
        from miniworld_engine.kernels.trimul_inproj.cute import output_training

        y, xhat, mean, rstd = output_training.forward(x, g, b, w, eps)
        return (y,) + output_training.backward(dy, y, xhat, rstd, g, b, w)

    graphs = {}
    keep = []
    result = {"L": args.length, "gpu": torch.cuda.get_device_name(), "candidates": {}}
    baseline = te()
    for name, fn in [
        ("te", te),
        ("te_full", te_full),
        ("normalized_rows", normalized_rows),
        ("production", production),
        ("m1_rows", m1_rows),
        ("m2_rows", m2_rows),
        ("m1_fused", native_fused),
        ("m1_te_backward", native_composed),
        ("m2_fused", m2_fused),
        ("m2_te_backward", m2_composed),
    ]:
        if name not in args.names.split(","):
            continue
        print("BEGIN", name, flush=True)
        try:
            compiled = torch.compile(
                fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
            )
            graph, out = capture(compiled)
            errors = [
                float((v.float() - r.float()).norm() / r.float().norm().clamp_min(1e-10))
                for v, r in zip(out, baseline)
            ]
            assert max(errors) < 0.025, errors
            graphs[name] = graph
            keep.append((compiled, out))
            result["candidates"][name] = {"rel_l2_vs_te": errors}
            print("CAPTURED", name, errors, flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            result["candidates"][name] = {"error": str(e)}
        args.output.write_text(json.dumps(result, indent=2))
    for name, timing in paired(graphs).items():
        result["candidates"][name].update(timing)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
