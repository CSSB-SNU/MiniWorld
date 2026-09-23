"""Read-only runtime audit of TriMul tuning decisions on an allocated H100.

Cache hooks observe real selector return values; they never replace candidates.
Lookup records describe first-use decisions, not per-launch cache hit rates.
No persistent tuning data is written. Timing is secondary to coverage auditing.
"""

import argparse
import collections
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from miniworld_engine import settings
from miniworld_engine.autotune import cache, native
from miniworld_engine.autotune.configs import op_of

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=["outgoing", "incoming", "bidir"], required=True)
parser.add_argument(
    "--implementation", choices=["miniworld", "triton"], default="miniworld"
)
parser.add_argument("--lengths", nargs="+", type=int, default=[128, 384, 768])
parser.add_argument("--output", required=True)
parser.add_argument("--compare-default", action="store_true")
args = parser.parse_args()
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
context = {}
records = {}
misses = []
force_default = False


def record(event):
    event = dict(context, **event)
    key = json.dumps(event, sort_keys=True, default=str)
    if key not in records:
        records[key] = dict(event, calls=0)
    records[key]["calls"] += 1


original_miss = cache._miss
original_subset = cache._cached_subset
original_select = cache.select_config


def logged_miss(op, gk, what, why, configs):
    misses.append(why)
    return original_miss(op, gk, what, why, configs)


def logged_subset(autotuner, configs, nargs, meta):
    op = op_of(getattr(autotuner, "configs", None) or [])
    before = len(misses)
    result = original_subset(autotuner, configs, nargs, meta)
    if op is not None:
        record(
            dict(
                kind="triton",
                op=op,
                dtype=cache.dtype_of_args(nargs),
                bucket=cache.bucket_of_autotuner(autotuner, nargs, meta),
                cache_hit=len(misses) == before and result is not None,
                reason=misses[-1] if len(misses) > before else None,
                declared_configs=len(autotuner.configs),
                prepruned_configs=len(configs),
                selected_configs=len(result) if result is not None else len(configs),
            )
        )
    return result


def logged_select(op, **kwargs):
    if force_default and op == "trimul_inproj_masked_sm90_cute":
        return None  # Benchmark-only: native.choose_config takes its declared default.
    best = original_select(op, **kwargs)
    data = cache._load(op, cache.gpu_key(kwargs.get("device_index")))
    record(
        dict(
            kind="native",
            op=op,
            dtype=kwargs["dtype"],
            bucket=kwargs["bucket"],
            cache_hit=best is not None,
            file_exists=data is not None,
            selected=best,
            candidate_count=len(kwargs.get("candidates") or []),
            fallback=(kwargs.get("candidates") or [None])[0] if best is None else None,
        )
    )
    return best


cache._miss = logged_miss
cache._cached_subset = logged_subset
cache.select_config = native.select_config = logged_select
from miniworld_engine.modules import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)
from miniworld_engine.kernels.trimul_inproj.cute import dispatch

original_calibrate = dispatch._calibrate


def logged_calibrate(name, key, candidates):
    winner = original_calibrate(name, key, candidates)
    record(
        dict(
            kind="backend_calibration",
            op=name,
            key=repr(key),
            candidates=[name for name, _ in candidates],
            winner=candidates[winner][0],
        )
    )
    return winner


dispatch._calibrate = logged_calibrate
results = dict(
    implementation=args.implementation,
    gpu=cache.gpu_key(),
    torch=torch.__version__,
    env_identity=cache.env_identity(),
    family=args.family,
    runtime_settings=repr(settings.current()),
    cases=[],
)


def write():
    results["lookups"] = list(records.values())
    path.write_text(json.dumps(results, indent=2))


def run_case(length, training, compiled):
    global force_default
    context.clear()
    context.update(length=length, training=training, compiled=compiled)
    torch.manual_seed(681)
    cls = (
        BidirectionalTriangleMultiplication
        if args.family == "bidir"
        else TriangleMultiplication
    )
    extra = {} if args.family == "bidir" else dict(outgoing=args.family == "outgoing")
    model = (
        cls(128, implementation=args.implementation, p_drop=0.25, **extra)
        .cuda()
        .bfloat16()
        .train(training)
    )
    with torch.no_grad():
        for child in model.modules():
            if isinstance(child, torch.nn.Linear):
                child.weight.normal_(std=0.02)
    x = torch.randn(
        1,
        length,
        length,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=training,
    )
    dy = torch.randn_like(x)
    mask = torch.ones(1, length, device="cuda", dtype=torch.bool)
    mask[:, ::3] = False
    fn = (
        torch.compile(model, fullgraph=True, options={"triton.cudagraphs": False})
        if compiled
        else model
    )

    def zero():
        model.zero_grad(set_to_none=False)
        if x.grad is not None:
            x.grad.zero_()

    def step():
        if training:
            zero()
        y = fn(x, mask)
        if training:
            y.backward(dy)
        return y

    started = time.monotonic()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            y = step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()
    if training:
        assert torch.isfinite(x.grad).all()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
    warmup_seconds = time.monotonic() - started
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        if training:
            zero()
        with torch.profiler.record_function("TRIMUL_FORWARD"):
            y = fn(x, mask)
        if training:
            with torch.profiler.record_function("TRIMUL_BACKWARD"):
                y.backward(dy)
        torch.cuda.synchronize()
    trace = path.with_name(
        f"{path.stem}_L{length}_{'train' if training else 'infer'}_{'compiled' if compiled else 'eager'}_trace.json"
    )
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    summary = collections.defaultdict(lambda: dict(count=0, us=0.0))
    for e in events:
        if e.get("cat") == "kernel":
            summary[e["name"]]["count"] += 1
            summary[e["name"]]["us"] += e["dur"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = step()
    if training:
        graph.replay()
        previous = y.clone()
        graph.replay()
        assert not torch.equal(previous, y), "dropout RNG froze"
        del previous
    graphs = {"tuned": graph}
    if args.compare_default:
        try:
            force_default = True
            for _ in range(3):
                step()
            baseline = torch.cuda.CUDAGraph()
            with torch.cuda.graph(baseline):
                baseline_output = step()
            graphs["default"] = baseline
        finally:
            force_default = False
    timings = {label: [] for label in graphs}
    for repeat in range(12 if args.compare_default else 5):
        order = list(graphs)
        if repeat % 2:
            order.reverse()
        for label in order:
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
            for _ in range(20):
                graphs[label].replay()
            b.record()
            b.synchronize()
            timings[label].append(a.elapsed_time(b) / 20)
    samples = timings["tuned"]
    row = dict(
        context,
        width=128,
        backend=str(model._backend),
        warmup_seconds=warmup_seconds,
        graph_ms=statistics.median(samples),
        samples_ms=samples,
        trace=str(trace),
        kernels=sorted(
            [dict(name=k, **v) for k, v in summary.items()], key=lambda v: -v["us"]
        ),
    )
    if args.compare_default:
        row["default_ms"] = statistics.median(timings["default"])
        row["default_samples_ms"] = timings["default"]
        row["tuned_speedup"] = row["default_ms"] / row["graph_ms"]
    results["cases"].append(row)
    write()
    print(
        json.dumps({k: v for k, v in row.items() if k not in ["kernels", "samples_ms"]}),
        flush=True,
    )


# Compiled first: its cache lookups are observed before eager's tuner can warm them.
for compiled in (True, False):
    for length in args.lengths:
        for training in (False, True):
            with torch.set_grad_enabled(training):
                run_case(length, training, compiled)
            gc.collect()
            torch.cuda.empty_cache()
write()
