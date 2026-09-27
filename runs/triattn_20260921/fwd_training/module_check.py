"""Experimental forward insertion; keep all installed native backward fusions."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
from native import extension
from miniworld_engine.modules import TriangleAttention
from miniworld_engine.kernels import _compile
from miniworld_engine.kernels.triangle_attention import cuda
from miniworld_engine.kernels.triangle_attention.triton import main as core

ap = argparse.ArgumentParser()
ap.add_argument('--artifact', required=True)
ap.add_argument('--length', type=int, default=384)
ap.add_argument('--mode', choices=['bench', 'qualify'], required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--installed', action='store_true')
ap.add_argument('--baseline-artifact', help='Frozen native baseline, including after installation')
ap.add_argument('--rounds', type=int, default=16)
ap.add_argument('--replays', type=int, default=20)
a = ap.parse_args()
ext = extension(a.artifact)
original = core._tri_attn_fwd
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


@_compile.opaque(fake=core._tri_attn_fwd_fake, name='triangle_attention_fwd_training_experiment')
def candidate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor,
              shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(ext.forward(q,k,v,b))


if a.installed:
    installed_entry=original
    def original(*args):
        os.environ['MINIWORLD_TRIATTN_TRAINING_FWD']='0'
        return installed_entry(*args)
    def candidate(*args):
        os.environ['MINIWORLD_TRIATTN_TRAINING_FWD']='1'
        return installed_entry(*args)

if a.baseline_artifact:
    baseline_ext = extension(a.baseline_artifact)

    @_compile.opaque(fake=core._tri_attn_fwd_fake, name='triangle_attention_fwd_training_frozen_baseline')
    def frozen_baseline(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor,
                        shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(baseline_ext.forward(q,k,v,b))

    original = frozen_baseline


pkg = Path(cuda.__file__).parent
manifests = {}
for p in pkg.glob('*manifest.json'):
    info = json.loads(p.read_text())
    for filename, digest in info.get('files', info.get('sha256', {})).items():
        assert hashlib.sha256((pkg/filename).read_bytes()).hexdigest() == digest
    manifests[p.name] = info
report = dict(artifact=a.artifact, length=a.length, mode=a.mode, records=[], manifests=manifests,
              installed_dispatch=a.installed,
              candidate_build=json.loads((Path(__file__).parent/a.artifact/'build-ready.json').read_text()))
report.update(rounds=a.rounds, replays=a.replays, baseline_artifact=a.baseline_artifact)
if a.baseline_artifact:
    report['baseline_build'] = json.loads((Path(__file__).parent/a.baseline_artifact/'build-ready.json').read_text())


def save():
    a.output.write_text(json.dumps(report, indent=2)+'\n')


def relative(x,y):
    return float((x.detach().double()-y.detach().double()).norm()/y.detach().double().norm().clamp_min(1e-15))


def compare(got, ref):
    assert all(torch.isfinite(v).all() for v in (*got,*ref))
    errors = [relative(g,r) for g,r in zip(got,ref)]
    assert max(errors) < .015, errors
    return errors


def make_model(ending):
    model = TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,
                              implementation='triton',p_drop=0).cuda().bfloat16().train()
    with torch.no_grad():
        for name,w in model.named_parameters():
            if w.ndim>=2: w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'): w.fill_(1)
            else: w.zero_()
    assert all(getattr(model,n) for n in ('_fuse_projection_backward','_fuse_bias_backward',
        '_fuse_dq_backward','_fuse_front_backward','_fuse_gate_backward'))
    return model


def evaluate(model,x,mask,dy,fn):
    core._tri_attn_fwd = fn
    torch.manual_seed(90321)
    y=model(x,mask)
    return [y.detach(), *torch.autograd.grad(y,(x,*model.parameters()),dy)]


def capture(fn,stream=None):
    stream=stream or torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): values=fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream): values=fn()
    return graph,values,stream


if a.mode == 'qualify':
    from miniworld_engine.kernels.triangle_attention.cuda import bias_backward
    from miniworld_engine.autotune.shape_key import token_key
    for L in (64,128):
        for maskname in ('mixed','one_key','all_masked','dense'):
            torch.manual_seed(92522)
            q,k,v,dy=[torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16)
                      .view(1,L,L,4,32).permute(0,3,1,2,4) for _ in range(4)]
            b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
            if maskname=='mixed': b[...,::3]=torch.finfo(b.dtype).min
            if maskname in ('one_key','all_masked'):
                b.fill_(torch.finfo(b.dtype).min)
                if maskname=='one_key': b[...,7]=0
            base_o,base_m=original(q,k,v,b,token_key(L))
            cand_o,cand_m=candidate(q,k,v,b,token_key(L))
            base=bias_backward._backward(q,k,v,b,base_m,base_o,dy,True)
            got=bias_backward._backward(q,k,v,b,cand_m,cand_o,dy,True)
            if maskname=='all_masked':
                assert all(torch.count_nonzero(x)==0 for x in (*got,*base))
                metrics={'all_masked_zero':True}
            else:
                qr,kr,vr,br=[t.double().requires_grad_() for t in (q,k,v,b)]
                y=(qr@kr.transpose(-1,-2)/32**.5+br.unsqueeze(2)).softmax(-1)@vr
                ref=torch.autograd.grad(y,(qr,kr,vr,br),dy.double())
                metrics=[]
                for bv,cv,rv in zip(base,got,ref):
                    assert torch.isfinite(cv).all()
                    if float(rv.norm())<1e-10:
                        be=float((bv.double()-rv).square().mean().sqrt())
                        ce=float((cv.double()-rv).square().mean().sqrt())
                        assert ce<=be*1.15+2e-7,(be,ce)
                        metrics.append(dict(zero_reference=True,baseline_rms=be,candidate_rms=ce))
                    else:
                        be=relative(bv,rv); ce=relative(cv,rv)
                        assert ce<.03 and ce<=be*1.15+.0006,(be,ce)
                        metrics.append(dict(baseline_rel=be,candidate_rel=ce))
            report['records'].append(dict(kind='fp64_core_gradients',length=L,mask=maskname,metrics=metrics))
            print('FP64_GRAD',L,maskname,metrics,flush=True); save()

for ending in (False,True):
    L=a.length; torch.manual_seed(92301)
    base=make_model(ending); cand=copy.deepcopy(base)
    x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    xcand=x.detach().clone().requires_grad_()
    dy=torch.randn_like(x)
    mask=torch.ones(1,L,device='cuda',dtype=torch.bool); mask[:,::7]=False
    if a.mode=='qualify':
        for maskname in ('mixed','one_key','all_masked'):
            mask.fill_(True)
            if maskname=='mixed': mask[:,::7]=False
            else:
                mask.zero_()
                if maskname=='one_key': mask[:,L-3]=True
            for pdrop in (0.,.25):
                base.p_drop=cand.p_drop=pdrop
                ref=evaluate(base,x,mask,dy,original)
                got=evaluate(cand,x,mask,dy,candidate)
                errors=compare(got,ref)
                report['records'].append(dict(kind='full_module',ending=ending,mask=maskname,
                                               dropout=pdrop,errors=errors)); save()
        base.p_drop=cand.p_drop=0.
        mask.fill_(True); mask[:,::7]=False
        before=[w.detach().clone() for w in cand.parameters()]
        for model,fn in ((base,original),(cand,candidate)):
            core._tri_attn_fwd=fn
            optimizer=torch.optim.SGD(model.parameters(),lr=.1)
            optimizer.zero_grad(set_to_none=True)
            ((model(x,mask)*dy).float().mean()*128).backward(); optimizer.step()
        changed=[int((old!=new).sum()) for old,new in zip(before,cand.parameters())]
        assert all(n>0 for n in changed),changed
        errors=compare(evaluate(cand,x,mask,dy,candidate),evaluate(base,x,mask,dy,original))
        report['records'].append(dict(kind='optimizer_update',ending=ending,errors=errors,changed=changed)); save()
        core._tri_attn_fwd=candidate
        ref=evaluate(cand,x,mask,dy,candidate)
        compiled=torch.compile(cand,backend='inductor',fullgraph=True)
        y=compiled(x,mask)
        got=[y.detach(),*torch.autograd.grad(y,(x,*cand.parameters()),dy)]
        errors=compare(got,ref)
        report['records'].append(dict(kind='fullgraph',ending=ending,errors=errors)); save()
        for w in cand.parameters(): w.requires_grad_(False)
        dx=torch.autograd.grad(cand(x,mask),x,dy)[0]
        assert torch.isfinite(dx).all() and dx.float().norm()>0
        report['records'].append(dict(kind='frozen_parameters',ending=ending)); save()
    else:
        for regime in ('forward','backward','forward_backward'):
            graphs=[]; contexts=[]; ref=None
            for model,forward in ((base,original),(cand,candidate)):
                core._tri_attn_fwd=forward
                xarm=x if model is base else xcand
                params=(xarm,*model.parameters())
                stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                if regime=='forward': fn=lambda model=model,x=xarm: (model(x,mask),)
                elif regime=='backward':
                    with torch.cuda.stream(stream): y=model(xarm,mask)
                    torch.cuda.current_stream().wait_stream(stream)
                    fn=lambda y=y,params=params: torch.autograd.grad(y,params,dy,retain_graph=True)
                    contexts.append(y)
                else:
                    fn=lambda model=model,params=params,x=xarm: torch.autograd.grad(model(x,mask),params,dy)
                g,values,stream=capture(fn,stream); g.replay(); torch.cuda.synchronize()
                if ref is None: ref=[v.detach().clone() for v in values]
                errors=compare(values,ref)
                graphs.append((g,values,stream)); contexts.append(fn)
            times=[[],[]]; ratios=[]
            for rnd in range(a.rounds):
                pair={}
                for i in ((0,1) if rnd%2==0 else (1,0)):
                    g=graphs[i][0]; g.replay()
                    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    for _ in range(a.replays): g.replay()
                    end.record(); end.synchronize()
                    pair[i]=start.elapsed_time(end)*1000/a.replays; times[i].append(pair[i])
                ratios.append(pair[0]/pair[1])
            result=dict(kind=regime,ending=ending,baseline_us=statistics.median(times[0]),
                        candidate_us=statistics.median(times[1]),speedup=statistics.median(ratios),
                        paired_ratios=ratios,rounds_us=times,errors=errors)
            if regime=='forward':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    graphs[1][0].replay(); torch.cuda.synchronize()
                names=sorted(set(e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA))
                assert any('training_fwd_stream' in n for n in names),names
                assert not any('_attn_fwd' in n for n in names),names
                result['candidate_kernels']=names
            report['records'].append(result); save()
            print('PAIRED',L,ending,regime,result['baseline_us'],result['candidate_us'],result['speedup'],flush=True)
            del graphs,contexts,ref,g,values,stream,fn
            y=None
            gc.collect(); torch.cuda.empty_cache()
    core._tri_attn_fwd=original
    del base,cand,x,xcand,dy,mask
    gc.collect(); torch.cuda.empty_cache()
report['complete']=True; save()
print('COMPLETE',a.mode,a.artifact,L,flush=True)
