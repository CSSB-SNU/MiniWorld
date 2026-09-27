"""O/LSE against installed training FWD and independent sampled FP64 attention."""
import argparse
import json
import math
from pathlib import Path
import statistics

import torch
from native import extension
from miniworld_engine.kernels.triangle_attention.triton import main as core
from miniworld_engine.autotune.shape_key import token_key

ap = argparse.ArgumentParser()
ap.add_argument('--artifact', required=True)
ap.add_argument('--baseline-artifact', help='Frozen CUDA baseline for incremental comparisons')
ap.add_argument('--length', type=int, required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--native-only', action='store_true')
ap.add_argument('--cases', nargs='+', default=['mixed', 'dense', 'one_key', 'all_masked', 'late_live', 'large_logits'])
a = ap.parse_args()
L = a.length
ext = extension(a.artifact)
base_ext = extension(a.baseline_artifact) if a.baseline_artifact else None


def baseline_forward(q, k, v, b):
    if base_ext is not None:
        return base_ext.forward(q, k, v, b)
    return core._tri_attn_fwd(q, k, v, b, token_key(L))

torch.manual_seed(92501)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def rel(x, y):
    return float((x.double() - y.double()).norm() / y.double().norm().clamp_min(1e-15))


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            out = fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out = fn()
    return g, out


report = dict(artifact=a.artifact, length=L, records=[], native_only=a.native_only,
              baseline_artifact=a.baseline_artifact)
if a.baseline_artifact:
    report['baseline_build'] = json.loads((Path(__file__).parent/a.baseline_artifact/'build-ready.json').read_text())
for case in a.cases:
    q, k, v = [torch.randn(1, L, L, 128, device='cuda', dtype=torch.bfloat16)
               .view(1, L, L, 4, 32).permute(0, 3, 1, 2, 4) for _ in range(3)]
    b = torch.randn(1, 4, L, L, device='cuda', dtype=torch.bfloat16) * .5
    valid = torch.ones_like(b, dtype=torch.bool)
    if case == 'mixed':
        valid[..., ::7] = False
    elif case in ('one_key', 'all_masked', 'late_live'):
        valid.zero_()
        if case == 'one_key':
            valid[..., L-3] = True
        elif case == 'late_live':
            valid[..., L-8:] = True
    elif case == 'large_logits':
        q.mul_(4); k.mul_(4); b.mul_(4)
    b.masked_fill_(~valid, torch.finfo(b.dtype).min)
    got, gm = ext.forward(q, k, v, b)
    torch.cuda.synchronize()
    assert torch.isfinite(got).all() and torch.isfinite(gm).all()
    record = dict(case=case)
    if not a.native_only:
        ref, rm = baseline_forward(q, k, v, b)
        assert got.stride() == ref.stride() and gm.stride() == rm.stride()
        record.update(installed_o_rel=rel(got, ref), installed_lse_max=float((gm-rm).abs().max()))
        rows = torch.tensor(sorted(set([0, 1, L//2, L-1])), device='cuda')
        queries = torch.arange(0, L, max(1, L//17), device='cuda')
        qq = q[:, :, rows][:, :, :, queries].double()
        kk = k[:, :, rows].double(); vv = v[:, :, rows].double()
        logits = qq @ kk.transpose(-1, -2) / math.sqrt(32) + b[:, :, None, queries].double()
        live = valid[:, :, None, queries]
        logits = logits.masked_fill(~live, -torch.inf)
        nonempty = live.any(-1)
        prob = logits.softmax(-1).nan_to_num(0.)
        exact = prob @ vv
        exact_m = torch.logsumexp(logits, -1) / math.log(2)
        exact_m = torch.where(nonempty, exact_m, torch.full_like(exact_m, float(torch.tensor(-1e38, dtype=torch.float32))))
        cg = got[:, :, rows][:, :, :, queries]
        cr = ref[:, :, rows][:, :, :, queries]
        cm = gm[:, :, rows][:, :, :, queries]
        record.update(fp64_o_rel=rel(cg, exact), baseline_fp64_o_rel=rel(cr, exact),
                      fp64_lse_max=float((cm-exact_m).abs().max()))
        assert record['installed_o_rel'] < .008, record
        assert record['fp64_o_rel'] < .005, record
        assert record['fp64_lse_max'] < .0003, record
        if case == 'all_masked':
            assert torch.count_nonzero(got) == 0 and torch.equal(gm, rm)
        if case == 'mixed':
            graphs = [capture(lambda: baseline_forward(q,k,v,b)),
                      capture(lambda: ext.forward(q,k,v,b))]
            times = [[], []]; ratios = []
            for rnd in range(12):
                pair = {}
                for i in ((0,1) if rnd%2==0 else (1,0)):
                    g, values = graphs[i]; g.replay()
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    for _ in range(20): g.replay()
                    end.record(); end.synchronize()
                    pair[i] = start.elapsed_time(end)*1000/20; times[i].append(pair[i])
                ratios.append(pair[0]/pair[1])
            record.update(baseline_us=statistics.median(times[0]), candidate_us=statistics.median(times[1]),
                          speedup=statistics.median(ratios), paired_ratios=ratios)
            # Replay must consume changed inputs and masks, rather than cached values.
            q.mul_(.75); b[..., ::5] = torch.finfo(b.dtype).min
            for g, _ in graphs: g.replay()
            torch.cuda.synchronize()
            assert rel(graphs[1][1][0], graphs[0][1][0]) < .008
            assert float((graphs[1][1][1]-graphs[0][1][1]).abs().max()) < .0003
            del graphs
    report['records'].append(record)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('CHECK', L, record, flush=True)
report['complete'] = True
a.output.write_text(json.dumps(report, indent=2)+'\n')
