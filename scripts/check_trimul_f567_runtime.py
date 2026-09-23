"""Fresh-process standard-cache, compiled training correctness and forward timing."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import torch
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as bt
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import output_f567_train
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_materialize, _te_forward
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_train
from miniworld_engine.autotune import cache
from miniworld_engine.modules import BidirectionalTriangleMultiplication


def capture(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):out=fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    del out
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=fn()
    return graph,out


def paired(graphs):
    samples={k:[] for k in graphs}
    for i in range(12):
        for k in list(graphs)[::(-1 if i%2 else 1)]:
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
            for _ in range(30):graphs[k].replay()
            b.record();b.synchronize();samples[k].append(a.elapsed_time(b)/30)
    return {k:dict(ms=statistics.median(v),samples_ms=v) for k,v in samples.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split-source',type=Path,required=True)
    a=p.parse_args();L=a.length;report={'length':L,'dynamic':False,'cache':[]}
    original_subset=cache._cached_subset
    def subset(tuner,configs,nargs,meta):
        answer=original_subset(tuner,configs,nargs,meta)
        if tuner.fn.__name__=='_output_f567_kernel':
            assert answer,'F567 did not use the standard runtime cache'
            report['cache'].append({'bucket':cache.bucket_of_autotuner(tuner,nargs,meta),
                                    'candidates':[cache.as_cfg_dict(c) for c in answer]})
        return answer
    cache._cached_subset=subset
    spec=importlib.util.spec_from_file_location('f567_split_reference',a.split_source)
    split=importlib.util.module_from_spec(spec);sys.modules[spec.name]=split;spec.loader.exec_module(split)
    fused=bt._BidirBackHalfTriton
    graphs={};keep=[];references=None
    try:
        for label,cls in [('split',split.SplitReference),('f567',fused)]:
            bt._BidirBackHalfTriton=cls;torch.compiler.reset();torch.manual_seed(614)
            model=BidirectionalTriangleMultiplication(128,implementation='triton',p_drop=0).cuda().bfloat16().train()
            with torch.no_grad():
                for n,w in model.named_parameters():
                    if 'ln_' not in n:w.normal_(std=128**-.5)
            x=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::3]=False
            dy=torch.randn_like(x)
            fn=torch.compile(model,dynamic=False,fullgraph=True,options={'triton.cudagraphs':False})
            y=fn(x,mask);y.backward(dy)
            values=[y.detach(),x.grad]+[w.grad for w in model.parameters()]
            assert all(v is not None and torch.isfinite(v).all() for v in values)
            if references is None:references=[v.clone() for v in values]
            errors=[((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item() for v,r in zip(values,references)]
            assert max(errors)<.015,errors
            report.setdefault('output_and_gradient_errors',{})[label]=errors
            del values,y
            model.p_drop=.25
            graph,out=capture(lambda:fn(x,mask))
            old=out.detach().clone();graph.replay();torch.cuda.synchronize()
            assert not torch.equal(old,out),'dropout froze'
            graphs[label]=graph;keep.append((model,x,mask,fn,out))
        report['whole_forward']=paired(graphs)
    finally:bt._BidirBackHalfTriton=fused
    del graphs,keep
    torch.cuda.empty_cache()
    torch.manual_seed(734);kw=dict(device='cuda',dtype=torch.bfloat16);m=L*L
    view=torch.randn(256,m,**kw).t();x=torch.randn(m,128,**kw)
    wp=torch.randn(128,256,**kw)/16;wg=torch.randn(128,128,**kw)/128**.5
    gamma=torch.ones(256,device='cuda');beta=torch.zeros_like(gamma)
    res=torch.randn(m,128,**kw);ds=(torch.rand(L,128,device='cuda')>.25).to(torch.bfloat16)/.75
    def split_tail():
        proj,norm,mean,rstd=_te_forward(view,gamma,beta,wp,None,1e-5)
        y,gate=gate_elem_train(x,proj,wg,res,ds,L)
        return y,proj,norm,mean,rstd,gate
    def fused_tail():
        norm,mean,rstd=_ln_materialize(view,gamma,beta,1e-5)
        y,proj,gate=output_f567_train(norm,x,wp,wg,res,ds,L)
        return y,proj,norm,mean,rstd,gate
    graphs={};keep=[]
    for label,fn in [('split',split_tail),('f567',fused_tail)]:
        graphs[label],out=capture(fn);keep.append(out)
    report['tail_forward']=paired(graphs)
    assert report['cache']
    report['complete']=True;a.output.write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__':main()
