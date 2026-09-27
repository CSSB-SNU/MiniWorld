"""Paired whole-block wide benchmark; fresh autograd leaves per capture regime."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import torch
import candidate

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--width',type=int,required=True);ap.add_argument('--length',type=int,required=True)
ap.add_argument('--output',type=Path,required=True);ap.add_argument('--mode',default='full',choices=('full','fwd','bwd'))
ap.add_argument('--rounds',type=int,default=16);ap.add_argument('--replays',type=int,default=10)
args=ap.parse_args();C,L=args.width,args.length
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
candidate.preload(C//4,args.mode)
report=dict(width=C,length=L,heads=4,head_dim=C//4,mode=args.mode,job=os.getenv('SLURM_JOB_ID'),
            torch=torch.__version__,device=torch.cuda.get_device_name(),measurements={},
            timing='paired alternating CUDA Graph replay, CUDA events; complete block; 9 gradients',
            source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('compare.py','candidate.py')},
            native={kind:json.loads((ROOT/f'native_head{C//4}/{kind}-ready.json').read_text()) for kind in ('fwd','dq','dkdv')})


def save():args.output.write_text(json.dumps(report,indent=2)+'\n')


def capture(step,stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):out=step()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();del out
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=step()
    graph.replay();torch.cuda.synchronize()
    return graph,out


def errors(got,ref,names):
    result={}
    for name,a,b in zip(names,got,ref):
        assert torch.isfinite(a).all() and torch.isfinite(b).all(),name
        norm=float(b.float().norm());diff=a.float()-b.float()
        result[name]=dict(relative_l2=float(diff.norm()/b.float().norm().clamp_min(1e-8)),max_abs=float(diff.abs().max()))
        assert result[name]['relative_l2']<.015 or (norm<1e-7 and result[name]['max_abs']<2e-4),(name,result[name])
    return result


def time_graph(graph):
    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    a.record()
    for _ in range(args.replays):graph.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)/args.replays


def profile_graph(graph):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(3):graph.replay()
        torch.cuda.synchronize()
    kernels={}
    for e in p.events():
        if e.device_type==torch.autograd.DeviceType.CUDA:
            cell=kernels.setdefault(e.name,dict(us=0,calls=0))
            cell['us']+=e.device_time_total/3;cell['calls']+=1
    launches=sum(e.name=='cudaGraphLaunch' for e in p.events())
    assert launches==3,launches
    return dict(graph_launches=launches,kernels=sorted([dict(name=k,**v) for k,v in kernels.items()],key=lambda x:-x['us']))


def cell(ending,regime):
    torch.manual_seed(92701)
    models=[candidate.OriginalTriton(C,n_head=4,d_hidden=C,starting=not ending,implementation='triton',p_drop=0).cuda().bfloat16().train() for _ in range(2)]
    with torch.no_grad():
        for name,w in models[0].named_parameters():
            if w.ndim>=2:w.normal_(std=C**-.5)
            elif name.endswith('weight'):w.normal_(mean=1,std=.05)
            else:w.normal_(std=.05)
        models[1].load_state_dict(models[0].state_dict())
    candidate.attach(models[1],args.mode)
    x=torch.randn(1,L,L,C,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    xs=[x,x.detach().clone().requires_grad_()];dy=torch.randn_like(x)
    mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
    graphs=[];outputs=[];forwards=[];contexts=[]
    names=('original_triton','wide_cuda')
    grad_names=('input',*dict(models[0].named_parameters()))
    for name,model,x in zip(names,models,xs):
        print('CAPTURE',C,L,ending,regime,name,flush=True)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        params=(x,*model.parameters())
        if regime=='backward':
            with torch.cuda.stream(stream):y=model(x,mask)
            torch.cuda.current_stream().wait_stream(stream)
            fwd=y.detach().clone()
            def step(y=y,params=params):return torch.autograd.grad(y,params,dy,retain_graph=True)
        elif regime=='forward':
            def step(model=model,x=x):return (model(x,mask),)
        else:
            def step(model=model,x=x,params=params):
                y=model(x,mask)
                return (y,*torch.autograd.grad(y,params,dy))
        graph,values=capture(step,stream)
        if regime!='backward':fwd=values[0]
        graphs.append(graph);outputs.append(values);forwards.append(fwd);contexts.append((step,stream))
    result=dict(ending=ending,regime=regime,arms={})
    labels=('output',) if regime=='forward' else grad_names if regime=='backward' else ('output',*grad_names)
    result['errors']=errors(outputs[1],outputs[0],labels)
    result['forward_error']=errors((forwards[1],),(forwards[0],),('output',))
    assert (forwards[0].float()-xs[0].float()).norm()>0
    for _ in range(10):
        for g in graphs:g.replay()
    times=[[],[]]
    for r in range(args.rounds):
        for i in ((0,1) if r%2==0 else (1,0)):times[i].append(time_graph(graphs[i]))
    for i,name in enumerate(names):
        result['arms'][name]=dict(ms=statistics.median(times[i]),rounds_ms=times[i],profile=profile_graph(graphs[i]))
    native_names=[r['name'] for r in result['arms']['wide_cuda']['profile']['kernels']]
    if regime!='backward' and args.mode!='bwd':assert any('wide_training_fwd' in n for n in native_names)
    if regime!='forward' and args.mode!='fwd':
        for expected in ('wide_grouped_dkdv','wide_dq_resident_alias','wide_delta'):assert any(expected in n for n in native_names),(expected,native_names)
    result['speedup']=result['arms']['original_triton']['ms']/result['arms']['wide_cuda']['ms']
    print('RESULT',C,L,ending,regime,result['speedup'],{k:v['ms'] for k,v in result['arms'].items()},flush=True)
    return result


for ending in (False,True):
    for regime in ('forward','backward','forward_backward'):
        key=f'{regime}_e{int(ending)}';report['measurements'][key]=cell(ending,regime);save()
        gc.collect();torch.cuda.empty_cache()
report['complete']=True;save()
