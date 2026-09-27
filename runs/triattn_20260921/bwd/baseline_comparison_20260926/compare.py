"""Complete training BWD/F+B: PyTorch, installed engine, explicit dQ candidate.

The pristine Anthropic optimized release has no backward. Record its envelope
and the adapter's runtime refusal instead of timing incomplete gradients.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics

import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention.cuda import dq_backward

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent.parent
ENGINE = Path(__import__('miniworld_engine').__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bwd_reuse_candidate', ROOT.parent / 'reuse_20260926/candidate.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
parser = argparse.ArgumentParser()
parser.add_argument('--length', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rounds', type=int, default=24)
parser.add_argument('--replays', type=int, default=10)
args = parser.parse_args()
L = args.length
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_engine():
    files = json.loads((RUN / 'fwd_training/checkpoint18246/snapshot.json').read_text())['sha256']
    for name, expected in files.items():
        assert digest(ENGINE / 'kernels/triangle_attention' / name) == expected, name
    return files


report = dict(length=L, batch=1, width=128, heads=4, head_dim=32, dtype='bfloat16',
              dropout=0, mask='every seventh key masked', rounds=args.rounds,
              replays=args.replays, torch=torch.__version__, cuda=torch.version.cuda,
              device=torch.cuda.get_device_name(), gpu_properties=str(torch.cuda.get_device_properties(0)),
              job=os.environ.get('SLURM_JOB_ID'), engine_checkpoint=18246,
              engine_sha256=verify_engine(), benchmark_sha256=digest(Path(__file__)),
              candidate_build=json.loads((ROOT.parent / 'dq/reuse_qdo/build-ready.json').read_text()),
              scope='complete block; input and all 8 parameter gradients; no optimizer or gradient accumulation',
              timing='CUDA events over warmed CUDA Graph replay; three arms rotated within each round',
              pytorch='existing implementation=pytorch: eager dense einsum/softmax, under CUDA Graph; not torch.compile or SDPA',
              cross_implementation_relative_l2_limit=.03, measurements={})


def save():
    args.output.write_text(json.dumps(report, indent=2) + '\n')


def anthropic_audit():
    release = RUN / 'oc-release/opt_core'
    original = Path('/home/psk6950/ext/uplifting-biomolecular-modeling/common/opt_core/opt_core')
    names = ('attn/pair_fused.py', 'kernels/triattn/TRIATTN_CELLS.json',
             'kernels/triattn/__init__.py', 'kernels/flash_triattn.py')
    hashes = {name: digest(release / name) for name in names}
    for name, expected in hashes.items():
        assert digest(original / name) == expected, name
    rows = json.loads((release / 'kernels/triattn/TRIATTN_CELLS.json').read_text())['rows']
    support = {name: row.get('backward') for name, row in rows.items()}
    assert all(support[name] is False for name in ('k2b', 'k2', 'flash', 'cuda_sm90a', 'triattn_native', 'triattn_exact'))
    model = TriangleAttention(128, n_head=4, d_hidden=128, implementation='anthropic', p_drop=0).cuda().bfloat16().train()
    x = torch.randn(1, 64, 64, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    try:
        model(x)
    except RuntimeError as error:
        message = str(error)
        assert 'inference-only' in message
    else:
        raise AssertionError('Expected the documented inference-only refusal')
    return dict(status='unsupported', release=str(release), release_source_sha256=hashes,
                rows_backward=support, adapter_runtime_error=message,
                note='Backward-capable stock/cuEq/DS4Sci/SDPA rows delegate to external libraries; they are not the previously benchmarked optimized Anthropic kernel.')


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def capture(fn, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            outputs = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    del outputs
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


@torch.no_grad()
def error_metrics(got, reference):
    result = []
    for value, target in zip(got, reference):
        assert torch.isfinite(value).all() and torch.isfinite(target).all()
        v, r = value.float(), target.float()
        result.append(dict(relative_l2=float((v-r).norm() / r.norm().clamp_min(1e-8)),
                           max_abs=float((v-r).abs().max()), reference_norm=float(r.norm()),
                           bitwise=torch.equal(value, target)))
    return result


def time_graph(graph):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(args.replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / args.replays


def profile_graph(graph):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    kernels = [e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return dict(kernels=kernels, profile_replays=3,
                cuda_graph_launches=sum(e.name == 'cudaGraphLaunch' for e in profile.events()))


def bench_cell(ending, regime):
    torch.manual_seed(92681)
    names = ('pytorch', 'engine18246', 'candidate_reuse_qdo')
    models = [TriangleAttention(128, n_head=4, d_hidden=128, starting=not ending,
                               implementation=('pytorch' if name == 'pytorch' else 'triton'),
                               p_drop=0).cuda().bfloat16().train() for name in names]
    with torch.no_grad():
        for name, weight in models[0].named_parameters():
            if weight.ndim >= 2:
                weight.normal_(std=weight.shape[-1] ** -.5)
            elif name.endswith('weight'):
                weight.normal_(mean=1, std=.05)
            else:
                weight.normal_(std=.05)
        for model in models[1:]:
            model.load_state_dict(models[0].state_dict())
    for model in models[1:]:
        assert all(getattr(model, n) for n in ('_fuse_projection_backward', '_fuse_front_backward',
                                              '_fuse_gate_backward', '_fuse_dq_backward', '_fuse_bias_backward'))
    x = torch.randn(1, L, L, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    xs = [x, x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()]
    dy = torch.randn_like(x)
    mask = torch.ones(1, L, device='cuda', dtype=torch.bool)
    mask[:, ::7] = False
    graphs, outputs, contexts, forwards = [], [], [], []
    params_names = ['input', *dict(models[0].named_parameters())]
    assert len(params_names) == 9, params_names
    result = dict(ending=ending, regime=regime, parameter_names=params_names, arms={})
    baseline_extension = dq_backward.extension()
    candidate.extension()
    for index, (name, model, x) in enumerate(zip(names, models, xs)):
        print('CAPTURE', L, ending, regime, name, flush=True)
        torch.cuda.reset_peak_memory_stats()
        with candidate.use() if index == 2 else nullcontext():
            params = (x, *model.parameters())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            if regime == 'backward':
                with torch.cuda.stream(stream):
                    y = model(x, mask)
                torch.cuda.current_stream().wait_stream(stream)
                forward = y.detach().clone()
                def step(y=y, params=params):
                    return torch.autograd.grad(y, params, dy, retain_graph=True)
            else:
                def step(model=model, x=x, params=params):
                    y = model(x, mask)
                    return (y, *torch.autograd.grad(y, params, dy))
            graph, values = capture(step, stream)
            if regime == 'forward_backward':
                forward = values[0]
            graphs.append(graph)
            outputs.append(values)
            forwards.append(forward)
            contexts.append((stream, step))
        result['arms'][name] = dict(capture_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    assert dq_backward.extension() is baseline_extension
    torch.cuda.synchronize()
    assert (forwards[0].detach().float() - xs[0].detach().float()).norm() > 0
    for index, name in enumerate(names):
        errors = error_metrics(outputs[index], outputs[0])
        result['arms'][name]['versus_pytorch'] = errors
        result['arms'][name]['forward_versus_pytorch'] = error_metrics([forwards[index]], [forwards[0]])[0]
        assert max(v['relative_l2'] for v in errors) < .03, (name, errors)
    result['candidate_bitwise_engine'] = all(torch.equal(a, b) for a, b in zip(outputs[1], outputs[2]))
    assert result['candidate_bitwise_engine']
    assert torch.equal(forwards[1], forwards[2])
    result['forward_candidate_bitwise_engine'] = True
    for _ in range(20):
        for graph in graphs:
            graph.replay()
    torch.cuda.synchronize()
    times = {name: [] for name in names}
    # Six permutations balance both order and immediate predecessor.
    orders = ((0, 1, 2), (2, 1, 0), (1, 2, 0), (0, 2, 1), (2, 0, 1), (1, 0, 2))
    for rnd in range(args.rounds):
        for index in orders[rnd % len(orders)]:
            graphs[index].replay()
            times[names[index]].append(time_graph(graphs[index]))
    for index, name in enumerate(names):
        arm = result['arms'][name]
        arm.update(median_ms=statistics.median(times[name]), rounds_ms=times[name])
        arm['profile'] = profile_graph(graphs[index])
        assert arm['profile']['cuda_graph_launches'] >= 1
        kernels = arm['profile']['kernels']
        if index == 2:
            assert any('dq_tma_reuse_qdo' in name for name in kernels), kernels
        elif index == 1:
            assert any('dq_tma' in name for name in kernels), kernels
            assert not any('dq_tma_reuse_qdo' in name for name in kernels)
        else:
            assert not any('dq_tma' in name for name in kernels)
    result['paired_engine_over_candidate'] = [a / b for a, b in zip(times[names[1]], times[names[2]])]
    result['paired_pytorch_over_candidate'] = [a / b for a, b in zip(times[names[0]], times[names[2]])]
    print('MEASURED', L, ending, regime, {n: result['arms'][n]['median_ms'] for n in names}, flush=True)
    return result


report['anthropic'] = anthropic_audit()
save()
cleanup()
for ending in (False, True):
    for regime in ('backward', 'forward_backward'):
        key = f'{regime}_e{int(ending)}'
        report['measurements'][key] = bench_cell(ending, regime)
        save()
        cleanup()
verify_engine()
report['complete'] = True
save()
print('COMPLETE', args.output, flush=True)
