"""Whole-module static graph with in-place input/weight/dy/mask changes."""
import argparse
import json
from pathlib import Path
import torch
import candidate
import selected
ap=argparse.ArgumentParser();ap.add_argument('--width',type=int,required=True)
ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();C=args.width;L=128
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
records=[]


def run(ending):
    torch.manual_seed(92731)
    models=[candidate.OriginalTriton(C,n_head=4,d_hidden=C,starting=not ending,implementation='triton',p_drop=0).cuda().bfloat16().train() for _ in range(2)]
    with torch.no_grad():
        for name,w in models[0].named_parameters():
            if w.ndim>=2:w.normal_(std=C**-.5)
            elif name.endswith('weight'):w.normal_(mean=1,std=.1)
            else:w.normal_(std=.1)
        models[1].load_state_dict(models[0].state_dict())
    selected.attach(models[1])
    x=torch.randn(1,L,L,C,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    xs=[x,x.detach().clone().requires_grad_()];dy=torch.randn_like(x)
    mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
    graphs=[];outs=[];contexts=[]
    for model,x in zip(models,xs):
        params=(x,*model.parameters())
        def step(model=model,x=x,params=params):
            y=model(x,mask);return(y,*torch.autograd.grad(y,params,dy))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):warm=step()
        torch.cuda.synchronize();del warm
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):values=step()
        graphs.append(graph);outs.append(values);contexts.append((stream,step))
    old=None;names=('output','input',*dict(models[0].named_parameters()))
    for iteration in range(3):
        if iteration:
            with torch.no_grad():
                xs[0].normal_(mean=.2*iteration,std=.8);xs[1].copy_(xs[0]);dy.normal_()
                mask.logical_not_()
                for a,b in zip(models[0].parameters(),models[1].parameters()):
                    update=torch.randn_like(a)*.002;a.add_(update);b.add_(update)
        for g in graphs:g.replay()
        torch.cuda.synchronize()
        errs={}
        for name,a,b in zip(names,outs[0],outs[1]):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            rel=float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-8))
            assert rel<.015,(name,rel);errs[name]=rel
        if old is not None:assert not torch.equal(old,outs[1][0])
        old=outs[1][0].clone()
        records.append(dict(ending=ending,iteration=iteration,errors=errs))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(3):graphs[1].replay()
        torch.cuda.synchronize()
    assert sum(e.name=='cudaGraphLaunch' for e in p.events())==3
    names=[e.name for e in p.events() if e.device_type==torch.autograd.DeviceType.CUDA]
    for n in ('wide_training_fwd','wide_dq_resident_alias','wide_grouped_dkdv'):
        assert any(n in name for name in names)
    assert any('wide_projection_dgrad_tma' in name for name in names)==(C==256)
    print('SELECTED_GRAPH_PASS',C,ending,flush=True)


for ending in (False,True):run(ending)
args.output.write_text(json.dumps(dict(width=C,length=L,records=records,complete=True),indent=2)+'\n')
