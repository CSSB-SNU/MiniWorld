"""Paired H100 bidirectional training audit; no production/cache mutations.

Whole-step graphs, compiled profiler traces, identical-buffer front tile sweep,
and same-operand cuBLAS/Quack primitive comparisons. Run one length per GPU.
"""

import argparse
import collections
import gc
import importlib
import json
import statistics
from pathlib import Path

import torch
import triton
from miniworld_engine.autotune import cache, native
from miniworld_engine.autotune.cute_config import (
    config_to_kwargs,
    gated_sm90_candidates,
    resolve_config,
)
from miniworld_engine.autotune.shape_key import pack, token_key
from miniworld_engine.kernels.trimul_inproj.cute import bidir_training as cute_train
from miniworld_engine.kernels.trimul_inproj.cute.launch import prepack_lr_operand
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
    _bidir_front_kernel,
    _bidir_front_launch,
)
from miniworld_engine.modules import BidirectionalTriangleMultiplication

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--phase", choices=["all", "whole", "components"], default="all")
parser.add_argument("--dynamic", choices=["auto", "true", "false"], default="auto")
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
L, D, H = args.length, 128, 256
M = L * L
result = dict(
    length=L,
    gpu=torch.cuda.get_device_name(),
    torch=torch.__version__,
    dynamic=args.dynamic,
    source_identity=native.source_identity(),
    whole=[],
    components=[],
    cache=[],
)
original_select, original_miss = native.select_config, cache._miss


def logged_select(op, **kwargs):
    selected = original_select(op, **kwargs)
    result["cache"].append(dict(op=op, bucket=kwargs.get("bucket"), selected=selected))
    return selected


def logged_miss(op, gk, what, why, configs):
    result["cache"].append(dict(op=op, miss=why))
    return original_miss(op, gk, what, why, configs)


native.select_config = cache.select_config = logged_select
cache._miss = logged_miss


def save():
    args.output.write_text(json.dumps(result, indent=2, default=str))


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            out = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    for _ in range(3):
        graph.replay()
    return graph, out


def paired(graphs, repeats=12, replays=30):
    samples = {name: [] for name in graphs}
    for repeat in range(repeats):
        order = list(graphs)
        if repeat % 2:
            order.reverse()
        for name in order:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / replays)
    return {
        name: dict(ms=statistics.median(values), samples_ms=values)
        for name, values in samples.items()
    }


def describe(t):
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype))


def compare(name, functions, operands=()):
    captures = {label: capture(fn) for label, fn in functions.items()}
    row = dict(
        name=name,
        operands=[describe(t) for t in operands],
        timings=paired({k: v[0] for k, v in captures.items()}),
    )
    outputs = [v[1] for v in captures.values()]
    if len(outputs) == 2 and all(isinstance(v, torch.Tensor) for v in outputs):
        a, b = outputs
        error = float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-8))
        row["relative_l2_between_variants"] = error
        assert error < 0.025, (name, error)
    result["components"].append(row)
    print(json.dumps(row), flush=True)
    save()
    return outputs[0]


def hybrid_front(x, WL, WLg, WR, WRg, Wg, **kw):
    """Benchmark-only: substitute only the H100 training front with Triton."""
    h = WL.shape[1]
    wl = kw.get("b_lr")
    if wl is None:
        wl = prepack_lr_operand(WL, WLg, WR, WRg)
    left, right, preact = _bidir_front_launch(
        x.reshape(M, D), wl, h, L, True, token_key(L), kw.get("pair_mask")
    )
    return left, right, preact.reshape(1, 4 * h, L, L)


def whole(compiled):
    graphs, keep = {}, []
    original_front = cute_train.trimul_inproj_cute_forward
    for label in ("triton", "cute", "cute_with_triton_front"):
        cute_train.trimul_inproj_cute_forward = (
            hybrid_front if label == "cute_with_triton_front" else original_front
        )
        torch.manual_seed(681)
        model = (
            BidirectionalTriangleMultiplication(
                D,
                implementation="triton" if label == "triton" else "miniworld",
                p_drop=0.25,
            )
            .cuda()
            .bfloat16()
            .train()
        )
        with torch.no_grad():
            for child in model.modules():
                if isinstance(child, torch.nn.Linear):
                    child.weight.normal_(std=0.02)
        x = torch.randn(
            1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        dy = torch.randn_like(x)
        mask = torch.ones(1, L, device="cuda", dtype=torch.bool)
        mask[:, ::3] = False
        compile_kwargs = (
            {} if args.dynamic == "auto" else {"dynamic": args.dynamic == "true"}
        )
        fn = (
            torch.compile(
                model,
                fullgraph=True,
                options={"triton.cudagraphs": False},
                **compile_kwargs,
            )
            if compiled
            else model
        )

        def step(fn=fn, model=model, x=x, mask=mask, dy=dy):
            model.zero_grad(set_to_none=False)
            if x.grad is not None:
                x.grad.zero_()
            y = fn(x, mask)
            y.backward(dy)
            return y

        graph, y = capture(step)
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        previous = y.clone()
        graph.replay()
        assert not torch.equal(previous, y), "dropout RNG froze"
        del previous
        graphs[label] = graph
        keep.append((model, x, dy, mask, fn, y, step))
        # Profile the uncaptured execution of exactly the compiled/eager step;
        # paired CUDA graph timing remains the performance authority.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            step()
            torch.cuda.synchronize()
        trace = args.output.with_name(
            f"{args.output.stem}_{label}_{compiled}_trace.json"
        )
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        kernels = collections.defaultdict(lambda: dict(count=0, us=0.0))
        for e in events:
            if e.get("cat") == "kernel":
                kernels[e["name"]]["count"] += 1
                kernels[e["name"]]["us"] += e["dur"]
        result["whole"].append(
            dict(
                label=label,
                compiled=compiled,
                trace=str(trace),
                kernels=sorted(
                    [dict(name=k, **v) for k, v in kernels.items()],
                    key=lambda v: -v["us"],
                ),
            )
        )
        print("captured", compiled, label, flush=True)
        save()
    cute_train.trimul_inproj_cute_forward = original_front
    timings = paired(graphs, replays=20)
    for row in result["whole"]:
        if row["compiled"] == compiled:
            row.update(timings[row["label"]])
    print("WHOLE", compiled, json.dumps(timings), flush=True)
    save()
    del graphs, keep, graph, y, model, x, dy, fn, step
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def components():
    masked = importlib.import_module(
        "miniworld_engine.kernels.trimul_inproj.cute.masked_front"
    )
    torch.manual_seed(891)
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(D, H, device="cuda", dtype=x.dtype) * 0.08 for _ in range(4)]
    w = prepack_lr_operand(*weights)
    mask1 = torch.ones(1, L, device="cuda", dtype=torch.bool)
    mask1[:, ::3] = False
    mask = (mask1.unsqueeze(-1) & mask1.unsqueeze(-2)).reshape(1, M).float()
    triton_mask = mask.to(x.dtype)
    out, pre = masked._masked_front_fake(x, w, mask, True)
    left, right = out.T[:H].reshape(1, H, L, L), out.T[H:].reshape(1, H, L, L)

    # Both native and Triton write the EXACT SAME output buffers, including preact.
    def front_triton():
        _bidir_front_kernel[lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]),)](
            x,
            w,
            left,
            right,
            pre.T,
            triton_mask,
            M,
            M,
            K=D,
            H2=H,
            shape_key=pack(token_key(L), H2=H, K=D),
            SAVE_PREACT=True,
        )
        return out

    front_triton()
    expected, expected_pre = out.clone(), pre.clone()
    fp32 = x.float() @ w.float()
    reference = torch.sigmoid(fp32[:, ::2]) * fp32[:, 1::2] * mask.T
    err = float((out.float() - reference).norm() / reference.norm().clamp_min(1e-8))
    assert err < 0.025
    result["front_reference_relative_l2"] = err
    del fp32, reference
    configs = gated_sm90_candidates()
    selected = resolve_config(
        "trimul_inproj_masked_sm90_cute",
        configs,
        dtype=str(x.dtype),
        bucket=native.tensor_key(x, w, mask, extra=(True,)),
        device_index=x.device.index,
        run=lambda c: masked._launch(x, w, out, pre, mask, c),
    )
    result["selected_native_config"] = config_to_kwargs(selected)
    variants = {"triton_cached": capture(front_triton)[0]}
    errors = {}
    for i, cfg in enumerate(configs):
        fn = lambda cfg=cfg: masked._launch(x, w, out, pre, mask, cfg)
        fn()
        error = float(
            (out.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-8)
        )
        pe = float(
            (pre.float() - expected_pre.float()).norm()
            / expected_pre.float().norm().clamp_min(1e-8)
        )
        assert error < 0.025 and pe < 0.025
        assert (out[mask.flatten() == 0] == 0).all()
        label = f"cute_{i}"
        errors[label] = dict(
            config=config_to_kwargs(cfg), relative_l2=error, preact_relative_l2=pe
        )
        variants[label] = capture(fn)[0]
    timing = paired(variants, replays=50)
    result["front_sweep"] = {
        k: dict(**v, **errors.get(k, {})) for k, v in timing.items()
    }
    result["triton_front_best_config"] = str(_bidir_front_kernel.best_config)
    print("FRONT_SWEEP", json.dumps(result["front_sweep"]), flush=True)
    save()
    del variants, expected, expected_pre

    from miniworld_engine.kernels._quack_compat import gemm as qgemm
    from miniworld_engine.kernels._quack_compat import gemm_act as qact
    from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
    from miniworld_engine.kernels.layernorm_linear.triton.te_style import (
        _te_backward,
        _te_forward,
    )
    from miniworld_engine.kernels.trimul_inproj.triton.back_fused import _dconcat
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import (
        gate_elem_bwd_ew,
        gate_elem_train,
    )

    lf, rf = left.reshape(H, L, L), right.reshape(H, L, L)
    go, gi = torch.randn_like(lf[:D]), torch.randn_like(lf[:D])
    pairs = {
        "contr_out_fwd": (lf[:D], rf[:D].transpose(1, 2)),
        "contr_in_fwd": (lf[D:].transpose(1, 2), rf[D:]),
        "contr_out_dleft": (go, rf[:D]),
        "contr_out_dright": (go.transpose(1, 2), lf[:D]),
        "contr_in_dleft": (rf[D:], gi.transpose(1, 2)),
        "contr_in_dright": (lf[D:], gi),
    }
    for name, (a, b) in pairs.items():
        compare(
            name,
            {
                "cublas": lambda a=a, b=b: torch.bmm(a, b),
                "quack": lambda a=a, b=b: qgemm(a, b),
            },
            (a, b),
        )
    gamma = torch.ones(H, device="cuda", dtype=torch.float32)
    beta = torch.zeros_like(gamma)
    tri = torch.cat(
        [torch.bmm(*pairs["contr_out_fwd"]), torch.bmm(*pairs["contr_in_fwd"])]
    )
    view = tri.reshape(H, M).T
    wp = torch.randn(D, H, device="cuda", dtype=x.dtype) * 0.02
    te = _te_forward(view, gamma, beta, wp, None, 1e-5)
    proj, xn, mean, rstd = te
    grad = torch.randn_like(proj)
    wg = torch.randn(D, D, device="cuda", dtype=x.dtype) * 0.02
    ds = torch.ones(L, D, device="cuda", dtype=x.dtype)
    y, gate = gate_elem_train(x, proj, wg, x, ds, L)
    dp, dg = gate_elem_bwd_ew(grad, proj, gate, ds, L)
    dl, dr = torch.randn_like(left), torch.randn_like(right)
    dc = _dconcat(
        dl.flatten(), dr.flatten(), pre.T, M, H, token_key(L), mask.flatten().to(x.dtype)
    )
    ws = torch.cat([weights[1].T, weights[0].T, weights[3].T, weights[2].T])
    for name, (a, b) in {"dW_gate": (x.T, dg), "dW_front": (dc, x)}.items():
        compare(
            name,
            {"cublas": lambda a=a, b=b: a @ b, "quack": lambda a=a, b=b: qgemm(a, b)},
            (a, b),
        )

    def dx_cublas():
        dx = dg @ wg.T
        dx.addmm_(dc.T, ws)
        return dx

    def dx_cute():
        front = qgemm(dc.T, ws)
        return qact(dg, wg.T, C=front, activation=None, store_preact=False)[1]

    compare(
        "dx_input_merged", {"cublas": dx_cublas, "cute": dx_cute}, (dc.T, ws, dg, wg.T)
    )
    common = {
        "shared_ln_input": lambda: triton_layernorm(
            x.reshape(1, L, L, D), gamma[:D], beta[:D], 1e-5
        ),
        "shared_ln_output_projection": lambda: _te_forward(
            view, gamma, beta, wp, None, 1e-5
        ),
        "shared_ln_output_projection_bwd": lambda: _te_backward(
            dp, xn, view, mean, rstd, gamma, wp, False
        ),
        "shared_gate_projection_epilogue": lambda: gate_elem_train(
            x, proj, wg, x, ds, L
        ),
        "shared_gate_elementwise_bwd": lambda: gate_elem_bwd_ew(grad, proj, gate, ds, L),
        "shared_front_elementwise_bwd": lambda: _dconcat(
            dl.flatten(), dr.flatten(), pre.T, M, H, token_key(L), triton_mask
        ),
        "shared_concat_forward": lambda: torch.cat([go, gi], dim=0),
    }
    for name, fn in common.items():
        compare(name, {"shared": fn})


if args.phase in ("all", "whole"):
    whole(True)
    whole(False)
if args.phase in ("all", "components"):
    components()
save()
print("COMPLETE", args.output, flush=True)
