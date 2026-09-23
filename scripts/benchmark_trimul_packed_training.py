"""Compare Triton, the previous H100 back-half, and installed packed H100 training.

Explicit compile mode, paired whole-step CUDA graphs, profiler traces, backend
choices, and native cache-hit evidence. Run one length per GPU.
"""

import argparse
import collections
import gc
import importlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
from miniworld_engine.autotune import cache, native
from miniworld_engine.kernels.trimul_inproj.cute import bidir_training as cute_train
from miniworld_engine.kernels.trimul_inproj.cute import contract
from miniworld_engine.modules import BidirectionalTriangleMultiplication

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline-source", type=Path, required=True)
parser.add_argument("--dynamic", choices=["auto", "true", "false"], default="false")
parser.add_argument("--require-cache-op", action="append", default=[])
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


spec = importlib.util.spec_from_file_location(
    "trimul_before_packed", args.baseline_source
)
before = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = before
spec.loader.exec_module(before)
current_back_half = cute_train.BidirBackHalf
backend_calls = []
original_bmm = contract._bmm


def logged_bmm(a, b, out, quack):
    backend_calls.append(
        dict(
            quack=quack,
            shape=list(a.shape),
            a_stride=list(a.stride()),
            b_stride=list(b.stride()),
        )
    )
    return original_bmm(a, b, out, quack)


contract._bmm = logged_bmm


def whole(compiled):
    graphs, keep = {}, []
    for label in ("triton", "previous_h100", "packed_h100"):
        cute_train.BidirBackHalf = (
            before.BidirBackHalf if label == "previous_h100" else current_back_half
        )
        backend_calls.clear()
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
                contraction_calls=list(backend_calls),
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
    cute_train.BidirBackHalf = current_back_half
    timings = paired(graphs, replays=20)
    for row in result["whole"]:
        if row["compiled"] == compiled:
            row.update(timings[row["label"]])
    print("WHOLE", compiled, json.dumps(timings), flush=True)
    save()
    del graphs, keep, graph, y, model, x, dy, fn, step
    gc.collect()
    torch.cuda.empty_cache()


whole(True)
for op in args.require_cache_op:
    selections = [row for row in result["cache"] if row["op"] == op]
    assert selections and all(row.get("selected") is not None for row in selections), (
        "required native cache did not hit", op, selections
    )
print("COMPLETE", args.output, flush=True)
