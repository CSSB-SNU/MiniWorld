"""Inference-only CUDA correctness, graph replay, and native boundary timings."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import torch
from torch.nn import functional as F
from native import extension

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('training_native', ROOT.parent/'fwd_training/native.py')
training_native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_native)


def rel(x, y):
    return float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-15))


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): out = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): out = fn()
    return graph, out, stream


def paired(functions, rounds=16, replays=20):
    graphs = [capture(f) for f in functions]
    times = [[], []]
    ratios = []
    for rnd in range(rounds):
        pair = {}
        for i in ((0, 1) if rnd % 2 == 0 else (1, 0)):
            graph = graphs[i][0]
            graph.replay()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(replays): graph.replay()
            end.record(); end.synchronize()
            pair[i] = start.elapsed_time(end)*1000/replays
            times[i].append(pair[i])
        ratios.append(pair[0]/pair[1])
    return dict(baseline_us=statistics.median(times[0]), candidate_us=statistics.median(times[1]),
                speedup=statistics.median(ratios), paired_ratios=ratios), graphs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--length', required=True, type=int)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--native-only', action='store_true')
    ap.add_argument('--cases', nargs='+', default=['mixed','dense','one_key','all_masked','late_live','large_logits'])
    a = ap.parse_args(); L = a.length
    torch.set_grad_enabled(False)
    torch.manual_seed(926261)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    ext = extension(a.artifact)
    head_last=hasattr(ext,'bias_head_last') and ext.bias_head_last()
    base = None if a.native_only else training_native.extension('cooperative_head2')
    def view(x): return x.view(1,L,L,4,32).permute(0,3,1,2,4)
    def baseline(z, weights, bias):
        q,k,v,g = [F.linear(z,w) for w in weights]
        out = base.forward(view(q),view(k),view(v),bias)[0].permute(0,2,3,1,4).reshape_as(z)
        return (torch.sigmoid(g.float())*out.float()).bfloat16()
    report = dict(artifact=a.artifact,length=L,records=[],native_only=a.native_only,
                  baseline='separate cuBLAS projections + frozen CUDA attention + gate elementwise; native boundary only',
                  build=json.loads((ROOT/a.artifact/'build-ready.json').read_text()))
    for case in a.cases:
        z = torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
        weights = [torch.randn(128,128,device='cuda',dtype=torch.bfloat16)/math.sqrt(128) for _ in range(4)]
        bias = torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
        valid = torch.ones_like(bias,dtype=torch.bool)
        if case == 'mixed': valid[...,::7] = False
        elif case in ('one_key','all_masked','late_live'):
            valid.zero_()
            if case == 'one_key': valid[...,L-3] = True
            elif case == 'late_live': valid[...,L-8:] = True
        elif case == 'large_logits': z.mul_(4); bias.mul_(4)
        bias.masked_fill_(~valid,torch.finfo(bias.dtype).min)
        packed_bias=bias.permute(0,2,3,1).contiguous() if head_last else bias
        got = ext.forward(z,*weights,packed_bias); torch.cuda.synchronize()
        assert got.shape == z.shape and got.is_contiguous() and torch.isfinite(got).all()
        rec = dict(case=case)
        if not a.native_only:
            ref = baseline(z,weights,bias)
            rec['relative'] = rel(got,ref)
            assert rec['relative'] < .008,rec
            rows = torch.tensor(sorted(set([0,1,L//2,L-1])),device='cuda')
            queries = torch.arange(0,L,max(1,L//17),device='cuda')
            projections = [F.linear(z[:,rows].double(),w.double()).bfloat16().double()
                           .view(1,len(rows),L,4,32).permute(0,3,1,2,4) for w in weights]
            q,k,v,g = projections
            logits = q[:,:,:,queries]@k.transpose(-1,-2)/math.sqrt(32)+bias[:,:,None,queries].double()
            logits.masked_fill_(~valid[:,:,None,queries],-torch.inf)
            exact = (logits.softmax(-1).nan_to_num(0.)@v).bfloat16().double()
            exact = (exact*torch.sigmoid(g[:,:,:,queries])).bfloat16().double()
            rec['fp64_relative'] = rel(view(got)[:,:,rows][:,:,:,queries],exact)
            rec['baseline_fp64_relative'] = rel(view(ref)[:,:,rows][:,:,:,queries],exact)
            assert rec['fp64_relative'] < .008,rec
            assert rec['fp64_relative'] <= rec['baseline_fp64_relative']*1.15+.0004,rec
            if case == 'mixed':
                timing, graphs = paired([lambda: baseline(z,weights,bias), lambda: ext.forward(z,*weights,packed_bias)])
                rec.update(timing)
                z.mul_(.75); weights[0].add_(.003); weights[3].sub_(.004)
                bias[...,::5] = torch.finfo(bias.dtype).min
                if head_last: packed_bias.copy_(bias.permute(0,2,3,1))
                graphs[0][0].replay(); graphs[1][0].replay(); torch.cuda.synchronize()
                rec['changed_graph_relative'] = rel(graphs[1][1],graphs[0][1])
                assert rec['changed_graph_relative'] < .008,rec
                del graphs
        if case == 'all_masked': assert torch.count_nonzero(got) == 0
        report['records'].append(rec)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print('INFERENCE_CHECK',L,rec,flush=True)
    report['complete'] = True
    a.output.write_text(json.dumps(report,indent=2)+'\n')
