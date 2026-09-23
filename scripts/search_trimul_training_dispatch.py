"""Benchmark-only packed contractions and per-operation H100 backend search."""

import argparse
import gc
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._quack_compat import gemm
from miniworld_engine.kernels.trimul_inproj.cute import bidir_training as production
from miniworld_engine.modules import BidirectionalTriangleMultiplication

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline-source", type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
L, D = args.length, 128
CHOICES = set()
result = {"length": L, "cases": [], "dtype": "bfloat16", "dynamic": False}


def matmul(name, a, b, out):
    if name in CHOICES:
        gemm(a, b, out=out)
    else:
        torch.bmm(a, b, out=out)


def fwd_fake(left, right, h):
    return left.new_empty(left.shape)


@opaque(fake=fwd_fake, name="trimul_search_contract_fwd")
def packed_forward(left: torch.Tensor, right: torch.Tensor, h: int) -> torch.Tensor:
    tri = torch.empty_like(left)
    matmul("out_fwd", left[:h], right[:h].transpose(1, 2), tri[:h])
    matmul("in_fwd", left[h:].transpose(1, 2), right[h:], tri[h:])
    return tri


def bwd_fake(grad, left, right, h):
    return left.new_empty(left.shape), right.new_empty(right.shape)


@opaque(fake=bwd_fake, name="trimul_search_contract_bwd")
def packed_backward(
    grad: torch.Tensor, left: torch.Tensor, right: torch.Tensor, h: int
) -> tuple[torch.Tensor, torch.Tensor]:
    dl, dr = torch.empty_like(left), torch.empty_like(right)
    matmul("out_dl", grad[:h], right[:h], dl[:h])
    matmul("out_dr", grad[:h].transpose(1, 2), left[:h], dr[:h])
    matmul("in_dl", right[h:], grad[h:].transpose(1, 2), dl[h:])
    matmul("in_dr", left[h:], grad[h:], dr[h:])
    return dl, dr


# Keep the original implementation in its own source file, replacing only the
# contractions and their output-buffer scaffolding. No installed source is edited.
baseline_spec = importlib.util.spec_from_file_location(
    "trimul_search_baseline", args.baseline_source
)
baseline_module = importlib.util.module_from_spec(baseline_spec)
sys.modules[baseline_spec.name] = baseline_module
baseline_spec.loader.exec_module(baseline_module)
text = args.baseline_source.read_text()
start = text.index("        o_out = dispatch.bmm(")
end = text.index("        view = tri.reshape", start)
text = text[:start] + "        tri = packed_forward(lf, rf, h)\n" + text[end:]
start = text.index("        d_o_out, d_o_in = d_tri[:h], d_tri[h:]")
end = text.index("        # front bwd:", start)
text = (
    text[:start]
    + """        d_left, d_right = packed_backward(d_tri, lf, rf, h)
        d_left = d_left.reshape(B, H, L, L)
        d_right = d_right.reshape(B, H, L, L)
        del d_tri

"""
    + text[end:]
)
source = args.output.with_name(args.output.stem + "_candidate.py")
source.write_text(text)
spec = importlib.util.spec_from_file_location("trimul_search_candidate", source)
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
candidate.packed_forward, candidate.packed_backward = packed_forward, packed_backward
spec.loader.exec_module(candidate)


def capture(step):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = step()
    for _ in range(3):
        graph.replay()
    return graph, output


def time_pair(baseline, trial):
    samples = [[], []]
    for repeat in range(12):
        for i in [0, 1] if repeat % 2 == 0 else [1, 0]:
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
            for _ in range(20):
                (baseline, trial)[i].replay()
            b.record()
            b.synchronize()
            samples[i].append(a.elapsed_time(b) / 20)
    return {
        "baseline_ms": statistics.median(samples[0]),
        "candidate_ms": statistics.median(samples[1]),
        "samples": samples,
    }


torch.manual_seed(681)
torch.backends.cuda.matmul.allow_tf32 = False
model = (
    BidirectionalTriangleMultiplication(D, implementation="miniworld", p_drop=0.25)
    .cuda()
    .bfloat16()
    .train()
)
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.normal_(std=0.02)
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dy = torch.randn_like(x)
mask = torch.ones(1, L, device="cuda", dtype=torch.bool)
mask[:, ::3] = False
original = production.BidirBackHalf
production.BidirBackHalf = baseline_module.BidirBackHalf


def compile_step():
    fn = torch.compile(
        model, dynamic=False, fullgraph=True, options={"triton.cudagraphs": False}
    )

    def step():
        model.zero_grad(set_to_none=False)
        if x.grad is not None:
            x.grad.zero_()
        output = fn(x, mask)
        output.backward(dy)
        return output

    return step


baseline_step = compile_step()
baseline, baseline_output = capture(baseline_step)
production.BidirBackHalf = candidate.BidirBackHalf
trial_step = compile_step()
names = ["out_fwd", "in_fwd", "out_dl", "out_dr", "in_dl", "in_dr"]
variants = [("packed_cublas", set()), ("packed_all_quack", set(names))]
variants += [("only_" + n, {n}) for n in names]
for label, choices in variants:
    CHOICES.clear()
    CHOICES.update(choices)
    graph, output = capture(trial_step)
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    previous = output.clone()
    graph.replay()
    assert not torch.equal(previous, output), "dropout froze"
    del previous
    row = dict(label=label, quack=sorted(choices), **time_pair(baseline, graph))
    result["cases"].append(row)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(row), flush=True)
    del graph, output
    gc.collect()
    torch.cuda.empty_cache()

# Assemble the individual wins; remeasure the whole combination rather than
# assuming individual improvements are additive.
packed = result["cases"][0]["candidate_ms"] / result["cases"][0]["baseline_ms"]
CHOICES.clear()
for row in result["cases"][2:]:
    if row["candidate_ms"] / row["baseline_ms"] < packed * 0.998:
        CHOICES.update(row["quack"])
graph, output = capture(trial_step)
row = dict(label="combined", quack=sorted(CHOICES), **time_pair(baseline, graph))
result["cases"].append(row)
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(row), flush=True)
production.BidirBackHalf = original
print("COMPLETE", args.output, flush=True)
