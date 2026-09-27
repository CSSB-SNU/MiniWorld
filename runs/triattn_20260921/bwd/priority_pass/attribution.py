"""Current installed whole-backward CUDA graph kernel attribution.

CUDA events outside the profiler provide total latency. CUPTI measures every
kernel in the same complete backward graph, including all parameter gradients,
mask/layout work, cuBLAS split-K reductions and initialization operations.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics

import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels.triangle_attention import cuda

ap = argparse.ArgumentParser()
ap.add_argument('--length', type=int, required=True)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
L = a.length
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
pkg = Path(cuda.__file__).parent
manifests = {}
for filename in ['manifest.json', 'bias_manifest.json', 'dq_manifest.json', 'ln_manifest.json', 'gate_manifest.json', 'wgrad_manifest.json']:
    data = json.loads((pkg / filename).read_text())
    for f, digest in data.get('files', data.get('sha256', {})).items():
        assert hashlib.sha256((pkg / f).read_bytes()).hexdigest() == digest
    manifests[filename] = data
report = dict(length=L, batch=1, channels=128, heads=4, head_dim=32,
              dtype='bfloat16', dropout=0, device=torch.cuda.get_device_name(),
              torch=str(torch.__version__), manifests=manifests, records=[])


def capture(fn, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            values = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        values = fn()
    return graph, values


def timed(graph, n=15):
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    begin.record()
    for _ in range(n):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / n


for ending in (False, True):
    torch.manual_seed(92301)
    model = TriangleAttention(128, n_head=4, d_hidden=128, starting=not ending,
                              implementation='triton', p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():
        for name, w in model.named_parameters():
            if w.ndim >= 2:
                w.normal_(std=w.shape[-1] ** -.5)
            elif name.endswith('weight'):
                w.fill_(1)
            else:
                w.zero_()
    assert all(getattr(model, f) for f in ['_fuse_projection_backward', '_fuse_bias_backward',
        '_fuse_dq_backward', '_fuse_front_backward', '_fuse_gate_backward'])
    x = torch.randn(1, L, L, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    mask = torch.ones(1, L, device='cuda', dtype=torch.bool)
    mask[:, ::7] = False
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = model(x, mask)
    torch.cuda.current_stream().wait_stream(stream)
    params = (x, *model.parameters())
    fn = lambda: torch.autograd.grad(y, params, dy, retain_graph=True)
    graph, values = capture(fn, stream)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.isfinite(v).all() for v in values)
    rounds = [timed(graph) for _ in range(9)]
    replay_count = 15
    profiler_warmup_replays = 10
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        # CUPTI startup idles the GPU. Warm again after instrumentation attaches;
        # retain the trace but discard those complete replays from attribution.
        timed(graph, profiler_warmup_replays)
        # Event interval contains exactly these complete backward replays.
        profiled_graph_us = timed(graph, replay_count)
    trace_path = a.output.with_name(a.output.stem + f'-e{int(ending)}-trace.json')
    prof.export_chrome_trace(str(trace_path))
    device_events = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                           key=lambda e: e.time_range.start)
    total_replays = profiler_warmup_replays + replay_count
    assert len(device_events) % total_replays == 0
    kernels_per_replay = len(device_events) // total_replays
    device_events = device_events[profiler_warmup_replays * kernels_per_replay:]
    kernels = {}
    sequence = []
    for e in device_events:
        entry = kernels.setdefault(e.name, dict(name=e.name, total_us=0., total_calls=0))
        entry['total_us'] += e.device_time_total
        entry['total_calls'] += 1
        sequence.append(dict(name=e.name, us=e.device_time_total, start_us=e.time_range.start,
                             end_us=e.time_range.end, stream=e.thread))
    for entry in kernels.values():
        entry['us_per_backward'] = entry['total_us'] / replay_count
        entry['calls_per_backward'] = entry['total_calls'] / replay_count
        assert entry['total_calls'] % replay_count == 0, entry
    kernel_sum = sum(v['us_per_backward'] for v in kernels.values())
    assert any('grouped_dkdv' in k for k in kernels)
    assert any('dq_tma' in k for k in kernels)
    assert any('projection_ln_residual_tma' in k for k in kernels)
    assert not any('_attn_bwd_preprocess' in k for k in kernels)
    for entry in kernels.values():
        entry['kernel_time_percent'] = 100 * entry['us_per_backward'] / kernel_sum
    record = dict(ending=ending, graph_us=statistics.median(rounds), event_rounds_us=rounds,
                  profiled_graph_us=profiled_graph_us, profile_replays=replay_count,
                  profiler_warmup_replays_discarded=profiler_warmup_replays,
                  kernel_sum_us=kernel_sum, kernels_per_backward=len(device_events) // replay_count,
                  kernels=sorted(kernels.values(), key=lambda v: -v['us_per_backward']),
                  kernel_sequence=sequence, trace=str(trace_path))
    report['records'].append(record)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print('ATTRIBUTION', L, ending, 'graph_us', record['graph_us'], 'kernel_sum_us', kernel_sum,
          'kernels', record['kernels_per_backward'], flush=True)
    for k in record['kernels']:
        print(round(k['us_per_backward'], 3), round(k['kernel_time_percent'], 2),
              k['calls_per_backward'], k['name'], flush=True)
    del graph, values, fn, params, y, model, x, dy, stream, prof
    gc.collect()
    torch.cuda.empty_cache()
report['complete'] = True
a.output.write_text(json.dumps(report, indent=2) + '\n')
