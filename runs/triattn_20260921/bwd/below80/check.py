"""Compare an isolated CUDA candidate with the current installed kernel.

No serving mutation; distinct module identities, all outputs checked, balanced
AB/BA CUDA graph timings. Full-module qualification remains a separate gate.
"""
import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('--kind', choices=['bias', 'dq', 'ln'], required=True)
ap.add_argument('--artifact', required=True)
ap.add_argument('--length', type=int, required=True)
ap.add_argument('--mask', choices=['none', 'mixed', 'one_key', 'all_masked'], default='mixed')
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--no-bench', action='store_true')
ap.add_argument('--profile', action='store_true')
ap.add_argument('--native-only', action='store_true')
a = ap.parse_args()
L = a.length
torch.manual_seed(95331)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
folder, library = {'bias': ('bias_fusion', 'triattn_bias_fusion'),
                   'dq': ('dq', 'triattn_dq'),
                   'ln': ('ln_residual', 'triattn_ln_residual')}[a.kind]
artifact = ROOT / folder / a.artifact
digests = json.loads((artifact / 'build-ready.json').read_text())
for filename, digest in digests.items():
    assert hashlib.sha256((artifact / filename).read_bytes()).hexdigest() == digest
name = (artifact / 'module-name.txt').read_text().strip()
spec = importlib.util.spec_from_file_location(name, artifact / 'build' / (library + '.so'))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
if a.kind == 'bias':
    from miniworld_engine.kernels.triangle_attention.cuda import bias_backward as installed
elif a.kind == 'dq':
    from miniworld_engine.kernels.triangle_attention.cuda import dq_backward as installed
else:
    from miniworld_engine.kernels.triangle_attention.cuda import ln_backward as installed
baseline = installed._extension() if a.kind == 'bias' else installed.extension()
assert candidate is not baseline and candidate.__file__ != baseline.__file__

if a.kind != 'ln':
    from miniworld_engine.kernels.triangle_attention.triton import main as core
    from miniworld_engine.autotune.shape_key import token_key, pack
    q, k, v, dy = [torch.randn(1, L, L, 128, device='cuda', dtype=torch.bfloat16)
                   .view(1, L, L, 4, 32).permute(0, 3, 1, 2, 4) for _ in range(4)]
    b = torch.randn(1, 4, L, L, device='cuda', dtype=torch.bfloat16) * .5
    if a.mask == 'mixed':
        b[..., ::3] = torch.finfo(b.dtype).min
    elif a.mask in ('one_key', 'all_masked'):
        b.fill_(torch.finfo(b.dtype).min)
        if a.mask == 'one_key':
            b[..., 7] = 0
    out, m = core._tri_attn_fwd(q, k, v, b, token_key(L))
    delta = torch.empty((1, 4, L, L), device='cuda', dtype=torch.float32)
    grid = lambda meta: [triton.cdiv(L, meta['BLOCK_M1']), 4 * L, 1]
    core._attn_bwd_preprocess[grid](out, dy, delta, *out.stride(), *dy.stride(),
        4 * L, 1, L, 32, shape_key=pack(token_key(L), HEAD_DIM=32), HEAD_DIM_PAD=32)
    args = (q, k, v, b, m, delta, dy) + ((4,) if a.kind == 'bias' else ())
else:
    from miniworld_engine.kernels.layernorm import compile_native as ln
    x = torch.randn(L * L, 128, device='cuda', dtype=torch.bfloat16)
    gamma = torch.randn(128, device='cuda', dtype=torch.float32) * .1 + 1
    z, mean, rstd = ln._dispatch_fwd(x.view(1, L, L, 128), gamma, torch.zeros_like(gamma), 1e-5)
    w = [torch.randn(c, 128, device='cuda', dtype=torch.bfloat16) * .08 for c in (128, 128, 128, 128, 4)]
    dy = [torch.randn(L * L, c, device='cuda', dtype=torch.bfloat16) for c in (128, 128, 128, 128, 4)]
    residual = torch.randn_like(x)
    args = (dy, w, x, mean, rstd, gamma, residual, L)


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


report = dict(kind=a.kind, artifact=a.artifact, length=L, mask=a.mask,
              candidate=candidate.__file__, baseline=baseline.__file__,
              source_digests=digests, records=[])
for ending in ((False, True) if a.kind == 'ln' else (False,)):
    params = args + ((ending,) if a.kind == 'ln' else ())
    functions = [lambda ext=ext: ext.backward(*params) for ext in (baseline, candidate)]
    got = functions[1]()
    got = (got,) if isinstance(got, torch.Tensor) else got
    torch.cuda.synchronize()
    assert all(torch.isfinite(t).all() for t in got)
    record = dict(ending=ending)
    if not a.native_only:
        ref = functions[0]()
        ref = (ref,) if isinstance(ref, torch.Tensor) else ref
        errors = [float((g.float() - r.float()).norm() / r.float().norm().clamp_min(1e-8)) for g, r in zip(got, ref)]
        record.update(relative_l2=errors, bitwise=[torch.equal(g, r) for g, r in zip(got, ref)])
        assert max(errors) < .01, record
        if a.mask == 'all_masked' and a.kind != 'ln':
            assert all(torch.count_nonzero(t) == 0 for t in got)
    if a.profile:
        graph, values = capture(functions[1])
        graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    elif not a.no_bench and not a.native_only:
        graphs = [capture(fn) for fn in functions]
        times = [[], []]
        ratios = []
        for rnd in range(12):
            pair = {}
            for i in ((0, 1) if rnd % 2 == 0 else (1, 0)):
                graph, values = graphs[i]
                graph.replay()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                pair[i] = start.elapsed_time(end) / 20
                times[i].append(pair[i])
            ratios.append(pair[0] / pair[1])
        record.update(baseline_ms=statistics.median(times[0]), candidate_ms=statistics.median(times[1]),
                      speedup=statistics.median(ratios), rounds_ms=times, paired_ratios=ratios)
        del graphs
    report['records'].append(record)
    print('RESULT', a.kind, a.artifact, L, ending, record, flush=True)
report['complete'] = True
a.output.write_text(json.dumps(report, indent=2) + '\n')
