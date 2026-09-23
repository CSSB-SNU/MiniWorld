"""Identical seeds/shapes, CUDA-graph replay timing; run in separate source overlays."""

import argparse
import gc
import hashlib
import time
import json
import statistics
from pathlib import Path
import torch
from miniworld_engine.modules import Transition, TriangleAttention, TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
# triattn    : TriangleAttention(starting, use_self_attention=False) -- the bias-only form the
#              trunk pairformer/MSA module run (`use_self_attention: False` in the model yaml)
# triattn_qk : TriangleAttention(starting, use_self_attention=True) -- the AF3 q/k form
# Both: 4 heads x 32 (AF3 / trunk config), width 128.
FAMILIES = ["outgoing", "bidir", "transition", "triattn", "triattn_qk"]
parser.add_argument("--family", default="all", choices=["all", *FAMILIES])
parser.add_argument("--families", default="", help="comma list; overrides --family")
parser.add_argument(
    "--implementation", choices=["miniworld", "pytorch", "cuequivariance"], default="miniworld",
    help="engine ImplementationType; 'cuequivariance' routes TriMul/TriAttn to cuequivariance_torch",
)
parser.add_argument("--dropout", type=float, default=0.25,
                    help="training-time module dropout (TriMul/TriAttn own it); 0 disables")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--lengths", type=int, nargs="+", default=[256, 384, 512, 768, 1024, 1536, 2048])
parser.add_argument("--mode", choices=["both", "training", "inference"], default="both")
args = parser.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
results = []


def measure(fn, model, x, training, dropout):
    warm_start = time.monotonic()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = fn()
    warm_seconds = time.monotonic() - warm_start
    assert torch.isfinite(y).all(), "nonfinite output"
    if training:
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for name, param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
    dropout_advances = None
    if dropout:
        graph.replay()
        previous = y.clone()
        graph.replay()
        dropout_advances = not torch.equal(previous, y)
        assert dropout_advances, "dropout RNG froze under CUDA graph replay"
        del previous
    graph_count = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    timings = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / 30)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graph_count
    return (
        statistics.median(timings),
        timings,
        warm_seconds,
        dropout_advances,
        graph_count,
    )


def case(family, length, width, training):
    torch.manual_seed(681)
    if family == "transition":
        model = Transition(width, implementation=args.implementation)
        shape = (1, length, length, width)
    elif family.startswith("triattn"):
        model = TriangleAttention(
            width, n_head=4, d_hidden=128, starting=True,
            use_self_attention=family == "triattn_qk",
            implementation=args.implementation, p_drop=args.dropout if training else 0,
        )
        shape = (1, length, length, width)
    else:
        cls = (
            BidirectionalTriangleMultiplication
            if family == "bidir"
            else TriangleMultiplication
        )
        model = cls(
            width, implementation=args.implementation, p_drop=args.dropout if training else 0
        )
        shape = (1, length, length, width)
    model = model.cuda().bfloat16().train(training)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(std=0.02)
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    weights_sha256 = digest.hexdigest()
    if args.compile:
        # dynamic=False as the trainer compiles: sweeping L in one process otherwise promotes L to
        # a SymInt on the second shape, and the engine's L-dependent dispatch (a python threshold)
        # then fails to trace ("SymNodeVariable() is not a constant").
        model.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=training)
    mask = torch.ones(1, length, device="cuda", dtype=torch.bool)
    mask[:, ::3] = False
    dy = torch.randn_like(x)

    def step():
        if training:
            model.zero_grad(set_to_none=False)
            if x.grad is not None:
                x.grad.zero_()
        y = model(x) if family == "transition" else model(x, mask)
        if training:
            y.backward(dy)
        return y

    with torch.set_grad_enabled(training):
        ms, samples, warm_seconds, dropout_advances, graph_count = measure(
            step, model, x, training, training and family != "transition" and args.dropout > 0
        )
    row = dict(
        label=args.label,
        implementation=args.implementation,
        compiled=args.compile,
        cudagraph="manual",
        gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        precision="bf16-parameters-and-inputs",
        allow_tf32=False,
        weights_sha256=weights_sha256,
        input_sample=x.detach().flatten()[:8].float().tolist(),
        dropout_p=args.dropout if training and family != "transition" else 0.0,
        dropout_rng_advances=dropout_advances,
        compiled_graphs=graph_count,
        warmup_seconds=warm_seconds,
        family=family,
        length=length,
        width=width,
        training=training,
        ms=ms,
        samples_ms=samples,
    )
    results.append(row)
    print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2))


selected = args.families.split(",") if args.families else (FAMILIES if args.family == "all" else [args.family])
for family in selected:
    for length in args.lengths:
        for training in (False, True):
            if args.mode != "both" and training != (args.mode == "training"):
                continue
            try:
                case(family, length, 512 if family == "transition" else 128, training)
            except Exception as exc:  # noqa: BLE001  (OOM at the top of the ladder is expected)
                row = dict(label=args.label, implementation=args.implementation, compiled=args.compile,
                           family=family, length=length, training=training, ms=None,
                           error=f"{type(exc).__name__}: {str(exc)[:200]}")
                results.append(row)
                print(json.dumps(row), flush=True)
                Path(args.output).write_text(json.dumps(results, indent=2))
            gc.collect()
            torch.cuda.empty_cache()
