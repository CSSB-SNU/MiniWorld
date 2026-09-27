import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch
from miniworld_engine import settings

settings.configure(engine_backend="triton", autotune_miss_cap=24)
from miniworld_engine.kernels.layernorm.interface import layernorm_kernel
from miniworld_engine.kernels.layernorm.triton.main import layer_norm_fwd_fused, layer_norm_bwd_dx_fused
from miniworld_engine.kernels.layernorm.triton.persistent import _ln_bwd_persistent
from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm, rmsnorm_fwd_kernel, rmsnorm_bwd_kernel
from miniworld_engine.kernels.layernorm_linear.interface import layernorm_linear_triton
from miniworld_engine.kernels.layernorm_linear.autograd import _compose_backward
from miniworld_engine.kernels.layernorm_linear.triton.te_style import layernorm_linear_te_fn
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels.layernorm_linear.triton.fused import _lnl_fwd_kernel
from miniworld_engine.kernels.norm_cuda import cuda_layernorm, cuda_rmsnorm, cuda_layernorm_linear

ROOT = Path(__file__).parent
CASES = [(8192, 64), (147456, 128), (589824, 128), (147456, 384), (8192, 1024)]
LINEAR = [(147456, 128, 16), (8192, 384, 512), (8192, 64, 64)]
TUNERS = [layer_norm_fwd_fused, layer_norm_bwd_dx_fused, _ln_bwd_persistent,
          rmsnorm_fwd_kernel, rmsnorm_bwd_kernel, _lnl_fwd_kernel]


class PortableShapeAdapter(torch.autograd.Function):
    """Same portable training operations, fixing only 2-D/3-D wrapper incompatibility.

    Public portable forward requires rank>=3 but its backward expects rank 2.
    This experiment preserves the unflattened forward key and flattens saved x/dY.
    No engine file, GPU kernel, or fusion/saving policy is changed.
    """
    @staticmethod
    def forward(ctx, x, gamma, beta, weight, bias):
        y = layernorm_linear_triton(x, gamma, beta, weight, bias, 1e-5)
        xf = x.reshape(-1, x.shape[-1]).float()
        mean = xf.mean(-1)
        inv = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5)
        ctx.save_for_backward(x.reshape(-1, x.shape[-1]), mean, inv, gamma, beta, weight)
        ctx.shape = x.shape
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        x, mean, inv, gamma, beta, weight = ctx.saved_tensors
        dx, dg, db, dw, dbias = _compose_backward(
            dy.reshape(-1, dy.shape[-1]), x, mean, inv, gamma, beta, weight,
            ctx.has_bias, dx_via_quack=False, shape_key=both_key(x.shape[0]))
        return dx.reshape(ctx.shape), dg, db, dw, dbias


def configs():
    result = {}
    for t in TUNERS:
        c = getattr(t, "best_config", None)
        if c is not None:
            result[t.fn.__name__] = {"kwargs": c.kwargs, "warps": c.num_warps, "stages": c.num_stages}
    return result


def measure(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(10):
            fn()
    for _ in range(3):
        graph.replay()
    times = []
    for _ in range(9):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 10)
    return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times)}


def inference(fn):
    def call():
        with torch.no_grad():
            return fn()
    return call


def run(group, idx, profile):
    torch.manual_seed(823)
    if group == 2:
        m, d, n = LINEAR[idx]
    else:
        m, d = CASES[idx]
        n = d
    x = torch.randn(1, m, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(d, device="cuda", requires_grad=True)
    b = torch.randn_like(w, requires_grad=True)
    if group == 2:
        proj = (torch.randn(n, d, device="cuda", dtype=x.dtype) / d**0.5).requires_grad_()
        bias = torch.randn(n, device="cuda", dtype=x.dtype, requires_grad=True)
        args = (x, w, b, proj, bias)
        forward = lambda: layernorm_linear_triton(*args, 1e-5)
        fn = lambda: PortableShapeAdapter.apply(*args)
        native = lambda: cuda_layernorm_linear(*args, 1e-5)
        composed = lambda: torch.nn.functional.linear(layernorm_kernel(x, w, b, 1e-5), proj, bias)
        ref = lambda: torch.nn.functional.linear(torch.nn.functional.layer_norm(x.float(), (d,), w, b, 1e-5).to(x.dtype), proj, bias)
    elif group == 1:
        args = (x, w)
        fn = forward = lambda: triton_rmsnorm(x, w, 1e-5)
        native = lambda: cuda_rmsnorm(x, w, 1e-5)
        ref = lambda: (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5) * w).to(x.dtype)
    else:
        args = (x, w, b)
        fn = forward = lambda: layernorm_kernel(x, w, b, 1e-5)
        native = lambda: cuda_layernorm(x, w, b, 1e-5)
        ref = lambda: torch.nn.functional.layer_norm(x.float(), (d,), w, b, 1e-5).to(x.dtype)
    dy = torch.randn(1, m, n, device="cuda", dtype=x.dtype)
    train = lambda: torch.autograd.grad(fn(), args, dy)
    y = fn()
    actual = torch.autograd.grad(y, args, dy)
    expected = ref()
    grads = torch.autograd.grad(expected, args, dy)
    errors = [float((a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-12)) for a, e in zip((y, *actual), (expected, *grads))]
    del y, actual, expected, grads
    result = {"group": group, "case": idx, "M": m, "D": d, "N": n, "dtype": "bfloat16", "errors": errors}
    if profile:
        for _ in range(3):
            train()
        result["configs"] = configs()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        train()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        (ROOT / f"profile-{group}-{idx}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    result["triton_fwd"] = measure(inference(forward))
    result["triton_train_fwd"] = measure(fn)
    result["triton_train"] = measure(train)
    result["configs"] = configs()
    result["cuda_fwd"] = measure(inference(native))
    result["cuda_train"] = measure(lambda: torch.autograd.grad(native(), args, dy))
    if group == 0:
        for path in ("atomic", "persistent"):
            settings.configure(layernorm_bwd_path=path)
            result[f"triton_{path}_train"] = measure(train)
            result[f"{path}_configs"] = configs()
        settings.configure(layernorm_bwd_path=None)
    if group == 2:
        result["triton_composed_fwd"] = measure(inference(composed))
        result["triton_composed_train"] = measure(lambda: torch.autograd.grad(composed(), args, dy))
        te = lambda: layernorm_linear_te_fn(x.reshape(m, d), w, b, proj, bias, 1e-5).reshape(1, m, n)
        result["triton_te_fwd"] = measure(inference(te))
        result["triton_te_train"] = measure(lambda: torch.autograd.grad(te(), args, dy))
    (ROOT / f"bench-{group}-{idx}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--case", type=int)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    indices = [a.case] if a.case is not None else range(3 if a.group == 2 else 5)
    for i in indices:
        run(a.group, i, a.profile)
        gc.collect()
        torch.cuda.empty_cache()
