"""Isolated A/B fusion qualification, config search, and paired CUDA-graph timing.

No dynamic compilation: whole training always uses dynamic=False/fullgraph=True.
Reports forward-only (with training saves) and full training independently.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _te_forward, _te_backward
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_train, gate_elem_bwd_ew
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as bt
from miniworld_engine.modules import BidirectionalTriangleMultiplication
import trimul_output_fusion as fusion
from trimul_fusion_training import BidirFusionA, BidirFusionB


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            out = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    del out  # release warmup autograd graphs before the capture iteration
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fn()
    graph.replay()
    return graph, out


def paired(graphs, repeats=10, replays=30):
    times = {k: [] for k in graphs}
    for i in range(repeats):
        keys = list(graphs)
        keys = keys[i % len(keys):] + keys[:i % len(keys)]
        if i % 2:
            keys.reverse()
        for key in keys:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(replays):
                graphs[key].replay()
            end.record()
            end.synchronize()
            times[key].append(start.elapsed_time(end) / replays)
    return {k: dict(ms=statistics.median(v), samples_ms=v) for k, v in times.items()}


def error(a, b):
    a, b = a.float(), b.float()
    return dict(rel_l2=((a-b).norm()/b.norm().clamp_min(1e-8)).item(),
                max_abs=(a-b).abs().max().item(), finite=bool(torch.isfinite(a).all()))


def inputs(length, k=256, n=128, m=None, zero=False):
    torch.manual_seed(841)
    m = length**2 if m is None else m
    kw = dict(device='cuda', dtype=torch.bfloat16)
    view = torch.randn(k, m, **kw).t()
    x = torch.randn(m, n, **kw)
    gamma = (1 + .1*torch.randn(k, device='cuda')).to(torch.bfloat16)
    beta = (.1*torch.randn(k, device='cuda')).to(torch.bfloat16)
    wp = (torch.randn(n, k, **kw) / k**.5).contiguous()
    wg = (torch.randn(n, n, **kw) / n**.5).contiguous()
    if zero:
        wg.zero_()
        wp.zero_()
    residual = torch.randn(m, n, **kw)
    ds = ((torch.rand(length, n, device='cuda') > .25).float() / .75).to(torch.bfloat16)
    return (view, x, gamma, beta, wp, wg, residual, ds, 1e-5, length)


def baseline(inp):
    view, x, g, b, wp, wg, res, ds, eps, length = inp
    proj, norm, mean, rstd = _te_forward(view, g, b, wp, None, eps)
    y, gate = gate_elem_train(x, proj, wg, res, ds, length)
    return y, proj, norm, mean, rstd, gate


def backward(inp, out, dy):
    view, x, g, b, wp, wg, res, ds, eps, length = inp
    y, proj, norm, mean, rstd, gate = out
    dp, dg = gate_elem_bwd_ew(dy, proj, gate, ds, length)
    dv, dgamma, dbeta, dwp, _ = _te_backward(dp, norm, view, mean, rstd, g, wp, False)
    return dv, dg @ wg.t(), dgamma, dbeta, dwp, x.t() @ dg, dy


def qualify(inp):
    base = baseline(inp)
    dy = torch.randn_like(base[0])
    bg = backward(inp, base, dy)
    # Independent FP32 reference uses exactly the representable BF16 inputs.
    vr, xr, gr, br, wpr, wgr, rr = [v.detach().float().requires_grad_() for v in inp[:7]]
    ds, eps, length = inp[7:]
    nr = F.layer_norm(vr, (vr.shape[1],), gr, br, eps)
    yr = rr + (nr @ wpr.t()) * torch.sigmoid(xr @ wgr) * ds.float()[torch.arange(vr.shape[0], device='cuda') % length]
    rg = torch.autograd.grad(yr, (vr, xr, gr, br, wpr, wgr, rr), dy.float())
    result = {}
    for label in ('baseline', 'A', 'B'):
        out = base if label == 'baseline' else fusion.output_forward(*inp, label)
        grads = bg if label == 'baseline' else backward(inp, out, dy)
        checks = dict(output_fp32=error(out[0], yr),
                      saved_vs_baseline={name: error(a, b) for name, a, b in zip(
                          ('y','proj','norm','mean','rstd','gate'), out, base)},
                      gradients_fp32={name: error(a, b) for name, a, b in zip(
                          ('tri','x_n','gamma','beta','wp','wg','residual'), grads, rg)},
                      gradients_vs_baseline={name: error(a, b) for name, a, b in zip(
                          ('tri','x_n','gamma','beta','wp','wg','residual'), grads, bg)})
        assert checks['output_fp32']['rel_l2'] < .012, (label, checks)
        for name, e in checks['gradients_fp32'].items():
            assert e['finite'] and e['rel_l2'] < .025, (label, name, e)
        for name, e in checks['saved_vs_baseline'].items():
            assert e['finite'] and e['rel_l2'] < .012, (label, name, e)
        for name, e in checks['gradients_vs_baseline'].items():
            assert e['finite'] and e['rel_l2'] < .012, (label, name, e)
        result[label] = checks
    return result


def tune(inp, save):
    base = baseline(inp)
    result = {}
    for kind in ('f45','f67','f567'):
        out = tuple(t.clone() for t in base)
        rows = []
        for i, cfg in enumerate(fusion.candidates(kind)):
            entry = dict(config=cfg)
            try:
                kernel = fusion.launch(kind, *inp, out, config=cfg)
                torch.cuda.synchronize()
                # All configurations are checked over the entire result, not a sample.
                indices = (1, 2, 3, 4) if kind == 'f45' else ((0, 5) if kind == 'f67' else (0, 1, 5))
                entry['errors'] = [error(out[j], base[j]) for j in indices]
                assert all(e['finite'] and e['rel_l2'] < .012 for e in entry['errors']), entry
                graph, _ = capture(lambda: fusion.launch(kind, *inp, out, config=cfg))
                entry.update(paired({'candidate': graph}, repeats=3, replays=12)['candidate'])
                entry['n_regs'] = kernel.n_regs
                entry['n_spills'] = kernel.n_spills
                entry['shared_bytes'] = kernel.metadata.shared
                del graph
            except Exception as exc:
                entry['error'] = str(exc)
            rows.append(entry)
            if i % 20 == 0:
                print('TUNE', inp[-1], kind, i, entry.get('ms', entry.get('error')), flush=True)
        valid = sorted((r for r in rows if 'ms' in r and 'error' not in r), key=lambda r: r['ms'])
        assert valid, (kind, rows)
        # Re-measure the top six together, avoiding a one-shot winner from clock drift.
        graphs = {}
        for i, row in enumerate(valid[:6]):
            cfg = row['config']
            graph, _ = capture(lambda cfg=cfg: fusion.launch(kind, *inp, out, config=cfg))
            graphs[str(i)] = graph
        finalists = paired(graphs, repeats=9, replays=20)
        winner_i = min(finalists, key=lambda k: finalists[k]['ms'])
        winner = valid[int(winner_i)]['config']
        fusion.SELECTED[fusion.config_key(kind, inp[0], inp[1], inp[4], inp[5])] = winner
        result[kind] = dict(winner=winner, finalist_candidates=[r['config'] for r in valid[:6]],
                            finalists=finalists, candidates=rows)
        save(result)
        print('WINNER', inp[-1], kind, winner, finalists[winner_i], flush=True)
        del graphs
    return result


def tail_benchmark(inp):
    forward_graphs, train_graphs, keep = {}, {}, []
    dy = torch.randn_like(inp[6])
    for label in ('baseline', 'A', 'B'):
        def fwd(label=label):
            return baseline(inp) if label == 'baseline' else fusion.output_forward(*inp, label)
        def train(fwd=fwd):
            out = fwd()
            return out, backward(inp, out, dy)
        forward_graphs[label], o1 = capture(fwd)
        train_graphs[label], o2 = capture(train)
        keep.append((o1, o2))
    return dict(forward=paired(forward_graphs), training=paired(train_graphs))


def whole(length, report):
    original = bt._BidirBackHalfTriton
    graphs, keep = {}, []
    qualification = {}
    try:
        for label, cls in [('baseline',original),('A',BidirFusionA),('B',BidirFusionB)]:
            bt._BidirBackHalfTriton = cls
            torch.compiler.reset()
            torch.manual_seed(631)
            model = BidirectionalTriangleMultiplication(128, implementation='triton', p_drop=0).cuda().bfloat16().train()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if 'ln_' not in name:
                        p.normal_(std=128**-.5)
            x = torch.randn(1,length,length,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            dy = torch.randn_like(x)
            mask = torch.ones(1,length,device='cuda',dtype=torch.bool)
            mask[:,::3] = False
            fn = torch.compile(model, dynamic=False, fullgraph=True, options={'triton.cudagraphs':False})
            y = fn(x, mask)
            y.backward(dy)
            measured = [y.detach().clone(), x.grad.clone()] + [p.grad.clone() for p in model.parameters()]
            if label == 'baseline':
                baseline_values = measured
                ref = BidirectionalTriangleMultiplication(128, implementation='pytorch', p_drop=0).cuda().float().train()
                ref.load_state_dict(model.state_dict())
                xr = x.detach().float().requires_grad_()
                yr = ref(xr, mask)
                yr.backward(dy.float())
                reference_values = [yr.detach().clone(), xr.grad.clone()] + [p.grad.clone() for p in ref.parameters()]
                del ref, xr, yr
            names = ['output','input'] + [n for n,_ in model.named_parameters()]
            qualification[label] = dict(
                vs_baseline={n:error(a,b) for n,a,b in zip(names,measured,baseline_values)},
                vs_fp32={n:error(a,b) for n,a,b in zip(names,measured,reference_values)})
            for n,e in qualification[label]['vs_fp32'].items():
                assert e['finite'] and e['rel_l2'] < .025, (label,n,e)
            for n,e in qualification[label]['vs_baseline'].items():
                assert e['finite'] and e['rel_l2'] < .015, (label,n,e)
            report['whole_qualification'] = qualification
            save_report(report)
            del measured, y  # default-stream AccumulateGrad must not survive into capture
            # RNG is tested on actual model-generated dropout, not just a fixed scale.
            model.p_drop = .25
            def step(fn=fn, model=model, x=x, mask=mask, dy=dy):
                model.zero_grad(set_to_none=False)
                if x.grad is not None:
                    x.grad.zero_()
                out = fn(x, mask)
                out.backward(dy)
                return out
            graph, output = capture(step)
            before = output.clone()
            graph.replay()
            torch.cuda.synchronize()
            assert not torch.equal(before,output), 'dropout frozen in graph'
            assert torch.isfinite(x.grad).all()
            graphs[label] = graph
            keep.append((model,x,dy,mask,fn,output,step))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                step()
                torch.cuda.synchronize()
            trace = args.output.with_name(args.output.stem+'_'+label+'_trace.json')
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            kernels = [dict(name=e['name'], us=e['dur']) for e in events if e.get('cat')=='kernel']
            report.setdefault('profiles',{})[label] = kernels
            save_report(report)
            print('CAPTURED whole',length,label,flush=True)
        report['whole_training'] = paired(graphs,repeats=12,replays=30)
        save_report(report)
    finally:
        bt._BidirBackHalfTriton = original
    return report


def save_report(report):
    report['runtime_selections'] = fusion.SELECTION_LOG
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--length',type=int,default=384)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['smoke','full','qualify','tail'],default='full')
    parser.add_argument('--load-tuning',type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(length=args.length, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, dynamic=False,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__),Path(fusion.__file__),
                                           Path(__file__).with_name('trimul_fusion_training.py')]})
    if args.mode == 'smoke':
        report['smoke'] = [qualify(inputs(17,k=256,n=128,m=289)),
                           qualify(inputs(17,k=250,n=125,m=283)),
                           qualify(inputs(17,k=256,n=128,m=289,zero=True))]
    else:
        inp = inputs(args.length)
        if args.load_tuning:
            report['tuning'] = json.loads(args.load_tuning.read_text())['tuning']
            for kind, record in report['tuning'].items():
                fusion.SELECTED[fusion.config_key(kind,inp[0],inp[1],inp[4],inp[5])] = record['winner']
        elif args.mode in ('full','tail'):
            def save_tuning(tuning):
                report['tuning'] = tuning
                save_report(report)
            report['tuning'] = tune(inp,save_tuning)
        fusion.REQUIRE_TUNED = bool(report.get('tuning'))
        report['tail_qualification'] = qualify(inp)
        save_report(report)
        if args.mode in ('full','tail'):
            report['tail_timing'] = tail_benchmark(inp)
        del inp
        gc.collect()
        torch.cuda.empty_cache()
        save_report(report)
        if args.mode in ('full','qualify'):
            whole(args.length,report)
    report['complete'] = True
    save_report(report)
    print('COMPLETE',args.output,flush=True)
