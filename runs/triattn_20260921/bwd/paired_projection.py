"""Balanced paired full-module backward and forward+backward, same GPU process."""
from pathlib import Path
source=(Path(__file__).parent/'bench_baseline.py').read_text().split('\nq,k,v=')[0]
exec(compile(source,str(Path(__file__).parent/'bench_baseline.py'),'exec'))
import copy,types,os
import projections
from native import extension
extension()
projections.MODE=os.environ.get('PROJECTION_MODE','cuda64')
report['mode']=projections.MODE

def timed(g):
    st,en=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    st.record()
    for _ in range(15):g.replay()
    en.record();en.synchronize()
    return st.elapsed_time(en)*1000/15

for ending in (False,True):
    torch.manual_seed(92301)
    base=TriangleAttention(128,n_head=4,d_hidden=128,starting=not ending,implementation='triton',p_drop=0).cuda().bfloat16().train()
    base._fuse_projection_backward=False
    with torch.no_grad():
        for name,w in base.named_parameters():
            if w.ndim>=2:w.normal_(std=w.shape[-1]**-.5)
            elif name.endswith('weight'):w.fill_(1)
            else:w.zero_()
    cand=copy.deepcopy(base);cand._attention=types.MethodType(projections.attention,cand)
    x0=torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    x1=x0.detach().clone().requires_grad_()
    mask=torch.ones(1,L,device='cuda',dtype=torch.bool);mask[:,::7]=False
    dy=torch.randn_like(x0)
    for regime in ('backward','forward_backward'):
        graphs=[];outs=[];contexts=[];ref=None
        for model,x in ((base,x0),(cand,x1)):
            s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
            params=(x,*model.parameters())
            with torch.cuda.stream(s):y=model(x,mask)
            torch.cuda.current_stream().wait_stream(s)
            if regime=='backward':fn=lambda y=y,params=params:torch.autograd.grad(y,params,dy,retain_graph=True)
            else:fn=lambda model=model,x=x,params=params:torch.autograd.grad(model(x,mask),params,dy)
            g,out=capture(fn,s);g.replay();torch.cuda.synchronize()
            assert all(torch.isfinite(v).all() for v in out)
            if ref is None:ref=[v.detach().clone() for v in out]
            errs=[float((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)) for v,r in zip(out,ref)]
            assert max(errs)<.015,errs
            graphs.append(g);outs.append(out);contexts.append((s,y,params,fn))
        times=[[],[]];ratios=[]
        for rnd in range(12):
            pair={}
            for i in ((0,1) if rnd%2==0 else (1,0)):
                graphs[i].replay();pair[i]=timed(graphs[i]);times[i].append(pair[i])
            ratios.append(pair[0]/pair[1])
        result=dict(baseline_us=statistics.median(times[0]),candidate_us=statistics.median(times[1]),speedup=statistics.median(ratios),paired_ratios=ratios,rounds_us=times,gradient_relative_l2=errs)
        report['measurements']['%s_e%d'%(regime,ending)]=result;save()
        print('PAIRED',regime,ending,result['baseline_us'],result['candidate_us'],result['speedup'],flush=True)
        del graphs,outs,contexts,ref,g,out,s,y,params,fn;gc.collect();torch.cuda.empty_cache()
    del base,cand,x0,x1,dy,model,x;gc.collect();torch.cuda.empty_cache()
report['complete']=True;save()
