"""Compare forced Triton Transition routes with H100 auto and compiled PyTorch.

Native settings are restored before every warmup, profile and graph capture.
Graph replays execute captured kernels, independent of subsequent settings.
"""

import argparse
import collections
import copy
import gc
import json
import statistics
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--length", type=int, default=768)
parser.add_argument("--layout", choices=["pair", "token"], default="pair")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--memory-only", action="store_true")
parser.add_argument("--output", required=True)
parser.add_argument("--engine-backward", choices=["triton", "cute"], default="triton")
parser.add_argument(
    "--variants",
    nargs="+",
    choices=["pytorch", "engine", "triton_fused", "triton_split"],
    default=["pytorch", "engine", "triton_fused", "triton_split"],
)
args = parser.parse_args()
if args.variants[0] != "pytorch":
    parser.error("--variants must start with pytorch for the shared reference")
from miniworld_engine import settings

settings.configure(transition_large_d_training=args.engine_backward)
from miniworld_engine.modules import Transition
from miniworld_engine import kernels as engine_kernels

NATIVE_DEFAULTS = dict(
    transition_cuda_b2b=True,
    transition_gatebwd_wgmma=True,
    transition_lnbwd_cuda=True,
    transition_dab_lnbwd=False,
    transition_force_split=False,
)
TRITON_ONLY = dict(
    NATIVE_DEFAULTS,
    transition_cuda_b2b=False,
    transition_gatebwd_wgmma=False,
    transition_lnbwd_cuda=False,
)


def activate(item):
    settings.configure(
        **(TRITON_ONLY if item["label"].startswith("triton_") else NATIVE_DEFAULTS)
    )


class FusedTritonTransition(Transition):
    def forward(self, x):
        return engine_kernels.triton_transition_fused(
            x,
            self.ln_in.weight.to(x.dtype),
            self.ln_in.bias.to(x.dtype),
            self.expand_a.weight.to(x.dtype),
            self.expand_b.weight.to(x.dtype),
            self.squeeze.weight.to(x.dtype),
            self.n,
            self.ln_in.eps,
            save_xn=False,
        )


class SplitTritonTransition(Transition):
    def forward(self, x):
        xn = engine_kernels.triton_layernorm(
            x, self.ln_in.weight, self.ln_in.bias, self.ln_in.eps
        )
        return (
            engine_kernels.triton_transition(
                xn,
                self.expand_a.weight.to(x.dtype),
                self.expand_b.weight.to(x.dtype),
                self.squeeze.weight.to(x.dtype),
                self.n,
            )
            + x
        )


path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(681)
shape = (
    (1, args.length, args.length, args.width)
    if args.layout == "pair"
    else (1, args.length, args.width)
)
x0 = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
dy = torch.randn_like(x0)
base = Transition(args.width, implementation="pytorch").cuda().bfloat16().train()
with torch.no_grad():
    for m in base.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.normal_(std=0.02)
state = copy.deepcopy(base.state_dict())
models = []
for label in args.variants:
    model = (
        (
            {
                "triton_fused": FusedTritonTransition,
                "triton_split": SplitTritonTransition,
            }.get(label, Transition)
        )(args.width, implementation="miniworld" if label == "engine" else "pytorch")
        .cuda()
        .bfloat16()
        .train()
    )
    model.load_state_dict(state)
    fn = torch.compile(model, fullgraph=True, options={"triton.cudagraphs": False})
    models.append(dict(label=label, model=model, fn=fn, x=x0.clone().requires_grad_()))
del base, state
result = {
    "width": args.width,
    "engine_backward": settings.current().transition_large_d_training,
    "length": args.length,
    "layout": args.layout,
    "shape": shape,
    "gpu": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "variants": {},
    "timing_rounds": [],
}


def write():
    path.write_text(json.dumps(result, indent=2))


def zero(item):
    item["model"].zero_grad(set_to_none=False)
    if item["x"].grad is not None:
        item["x"].grad.zero_()


def step(item):
    activate(item)
    zero(item)
    y = item["fn"](item["x"])
    y.backward(dy)
    return y


# Warm compilation/backward and compare every gradient before graph capture.
reference = None
for item in models:
    activate(item)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            y = step(item)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    values = {"output": y.detach(), "input": item["x"].grad}
    values.update({n: p.grad for n, p in item["model"].named_parameters()})
    assert all(v is not None and torch.isfinite(v).all() for v in values.values())
    if reference is None:
        reference = {n: v.detach().float().clone() for n, v in values.items()}
    errors = {
        n: float((v.float() - reference[n]).norm() / reference[n].norm().clamp_min(1e-8))
        for n, v in values.items()
    }
    assert max(errors.values()) < 0.025, errors
    result["variants"][item["label"]] = {
        "relative_errors": errors,
        "native_settings": {k: getattr(settings.current(), k) for k in NATIVE_DEFAULTS},
        "parameter_dtypes": {
            n: str(p.dtype) for n, p in item["model"].named_parameters()
        },
    }
    del values, y
    print(item["label"], "validated", max(errors.values()), flush=True)
del reference
gc.collect()
torch.cuda.synchronize()

# Warm the measurement stream too: cuBLAS workspaces are stream-local. Counting
# their lazy first allocation as retained activations would inflate PyTorch memory.
for item in models:
    activate(item)
    y = step(item)
    del y
torch.cuda.synchronize()
gc.collect()

# Memory: incremental allocation above live model/input/gradient buffers; before capture pools exist.
for item in models:
    activate(item)
    zero(item)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    y = item["fn"](item["x"])
    torch.cuda.synchronize()
    forward_live = torch.cuda.memory_allocated() - before
    y.backward(dy)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    result["variants"][item["label"]]["forward_live_mib"] = forward_live / 2**20
    result["variants"][item["label"]]["step_peak_extra_mib"] = peak / 2**20
    del y

if args.memory_only:
    write()
    print(json.dumps(result), flush=True)
    raise SystemExit(0)

# Forward/backward phase estimates with events, outside graphs (reported separately).
for item in models:
    activate(item)
    phases = []
    for _ in range(5):
        zero(item)
        a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        a.record()
        y = item["fn"](item["x"])
        b.record()
        y.backward(dy)
        c.record()
        c.synchronize()
        phases.append([a.elapsed_time(b), b.elapsed_time(c)])
        del y
    result["variants"][item["label"]]["nongraph_phase_ms"] = {
        "forward": statistics.median(p[0] for p in phases),
        "backward": statistics.median(p[1] for p in phases),
    }

# Profile before capture; CPU ranges help attribute each kernel through its launch parent.
if args.profile:
    for item in models:
        activate(item)
        zero(item)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            with torch.profiler.record_function("TRANSITION_FORWARD"):
                y = item["fn"](item["x"])
            with torch.profiler.record_function("TRANSITION_BACKWARD"):
                y.backward(dy)
            torch.cuda.synchronize()
        trace = path.with_name(path.stem + "_" + item["label"] + "_trace.json")
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        kernels = collections.defaultdict(lambda: dict(count=0, us=0.0))
        ranges = [
            e
            for e in events
            if e.get("name") in ("TRANSITION_FORWARD", "TRANSITION_BACKWARD")
        ]
        launches = {
            e.get("args", {}).get("correlation"): e
            for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver")
        }
        phases = collections.defaultdict(float)
        for e in events:
            if e.get("cat") != "kernel":
                continue  # GPU annotations span whole graphs; they are not kernels.
            v = kernels[e["name"]]
            v["count"] += 1
            v["us"] += e["dur"]
            launch = launches.get(e.get("args", {}).get("correlation"))
            phase = "unattributed"
            if launch:
                for span in ranges:
                    if span["ts"] <= launch["ts"] <= span["ts"] + span["dur"]:
                        phase = span["name"]
            phases[phase] += e["dur"] / 1000
        result["variants"][item["label"]]["profile_kernel_ms"] = dict(phases)
        result["variants"][item["label"]]["gpu_kernels"] = sorted(
            [dict(name=n, **v) for n, v in kernels.items()], key=lambda v: -v["us"]
        )
        if item["label"].startswith("triton_"):
            expected_forward = (
                "transition_fwd_kernel"
                if item["label"] == "triton_split"
                else (
                    "_transition_b2b_kernel"
                    if args.width <= 128
                    else "_transition_b2b_ktiled_kernel"
                )
            )
            assert expected_forward in kernels, (expected_forward, list(kernels))
            assert "_transition_expand_gatebwd_kernel" in kernels
            native_markers = (
                "wgmma_kernel",
                "kernel_cutlass",
                "layer_norm_bwd_main_kernel",
                "layer_norm_bwd_reduce_kernel",
            )
            assert not any(
                marker in name for marker in native_markers for name in kernels
            ), "A forced Triton route launched a native engine kernel"
        if (
            item["label"] == "engine"
            and args.width >= 512
            and args.engine_backward == "cute"
        ):
            assert any("GemmDLnGated" in n for n in kernels), (
                "CuTe backward was not observed"
            )
        result["variants"][item["label"]]["trace"] = str(trace)
        del y
write()

for item in models:
    activate(item)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = step(item)
    item["graph"], item["output"] = g, y
    item["samples"] = []
# Alternate order to reduce thermal/clock/order bias. Same inputs, all variants on same GPU.
for repeat in range(12):
    order = list(range(len(models)))
    if repeat % 2:
        order.reverse()
    round_data = {}
    for i in order:
        item = models[i]
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(20):
            item["graph"].replay()
        b.record()
        b.synchronize()
        ms = a.elapsed_time(b) / 20
        item["samples"].append(ms)
        round_data[item["label"]] = ms
    result["timing_rounds"].append(round_data)
for item in models:
    activate(item)
    out = result["variants"][item["label"]]
    out["graph_samples_ms"] = item["samples"]
    out["graph_median_ms"] = statistics.median(item["samples"])
    print(
        item["label"],
        json.dumps(
            {
                k: v
                for k, v in out.items()
                if k not in ["gpu_kernels", "relative_errors", "parameter_dtypes"]
            }
        ),
        flush=True,
    )
write()
