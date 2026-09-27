"""Strict complete F+B replay after input/weight/dy/mask changes, with attribution."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.cuda import dq_backward

parser = argparse.ArgumentParser()
parser.add_argument('--artifact', required=True)
parser.add_argument('--length', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--entry', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parent
folder = root.parent / 'dq' / args.artifact
build = json.loads((folder / 'build-ready.json').read_text())
for name, digest in build.items():
    assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
spec = importlib.util.spec_from_file_location((folder / 'module-name.txt').read_text().strip(),
                                            folder / 'build/triattn_dq.so')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
baseline = dq_backward.extension()
if args.entry:
    from candidate import use
    with use() as selected:
        assert dq_backward.extension() is selected
        candidate = selected
    assert dq_backward.extension() is baseline
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
report = dict(artifact=args.artifact, length=args.length, build=build, records=[],
              explicit_entry=args.entry)
L = args.length


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            values = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        values = fn()
    return graph, values


for ending in (False, True):
    torch.manual_seed(92671)
    old = TriangleAttention(128, n_head=4, d_hidden=128, starting=not ending,
                            implementation='triton', p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():
        for name, weight in old.named_parameters():
            if weight.ndim >= 2:
                weight.normal_(std=weight.shape[-1] ** -.5)
            elif name.endswith('weight'):
                weight.normal_(mean=1, std=.05)
            else:
                weight.normal_(std=.05)
    new = copy.deepcopy(old)
    x0 = torch.randn(1, L, L, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_()
    dy = torch.randn_like(x0)
    mask = torch.ones(1, L, device='cuda', dtype=torch.bool)
    mask[:, ::7] = False
    graphs = []
    for model, x, ext in ((old, x0, baseline), (new, x1, candidate)):
        dq_backward._EXT = ext
        def step(model=model, x=x):
            y = model(x, mask)
            return (y, *torch.autograd.grad(y, (x, *model.parameters()), dy))
        graphs.append(capture(step))
    dq_backward._EXT = baseline
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(graphs[0][1], graphs[1][1]))
    previous = None
    for iteration in range(3):
        if iteration:
            with torch.no_grad():
                x0.mul_(.8)
                x1.copy_(x0)
                old.to_query.weight.mul_(.9)
                new.to_query.weight.copy_(old.to_query.weight)
                old.to_value.weight.add_(.003)
                new.to_value.weight.copy_(old.to_value.weight)
                dy.mul_(.75)
                mask[:, ::(5 if iteration == 1 else 3)] = False
        for graph, _ in graphs:
            graph.replay()
        torch.cuda.synchronize()
        ref, got = graphs[0][1], graphs[1][1]
        assert all(torch.isfinite(v).all() for v in got)
        assert all(torch.equal(a, b) for a, b in zip(ref, got))
        assert got[1].float().norm() > 0
        if previous is not None:
            assert not torch.equal(previous, got[1])
        previous = got[1].clone()
        report['records'].append(dict(ending=ending, iteration=iteration, bitwise=True,
                                      compared_tensors=len(got)))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(3):
            graphs[1][0].replay()
        torch.cuda.synchronize()
    names = [event.name for event in profile.events()
             if event.device_type == torch.autograd.DeviceType.CUDA]
    expected = 'dq_tma_' + args.artifact
    assert sum(expected in name for name in names) == 3, names
    assert not any(name.startswith('dq_tma(') for name in names), names
    report['records'].append(dict(ending=ending, kind='actual_graph_kernels', names=names))
    print('CHANGED_GRAPH_PASS', L, ending, flush=True)
    del graphs, previous, ref, got, old, new, x0, x1, dy, mask, step, model, x
    import gc
    gc.collect()
    torch.cuda.empty_cache()
report['complete'] = True
args.output.write_text(json.dumps(report, indent=2) + '\n')
