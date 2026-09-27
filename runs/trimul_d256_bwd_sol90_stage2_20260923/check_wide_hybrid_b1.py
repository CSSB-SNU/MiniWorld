from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_hybrid_b1 import WideB1
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-hybrid-D{D}-L{N}-{record["job"]}.json'
if os.environ.get('COMPARE_TRITON')=='1':path=path.with_name(path.name.replace('wide-hybrid-','wide-hybrid-vs-triton-'))
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn),packed=f.w)
 if D==384:old_b1=B1(p)
 else:
  def old_b1():W.launch(p.ks['b1'],p.params,p.grid,D=D)
 candidate=WideB1(p,leaves)
 def tail():
  ab=p.front.ab;h=2*D
  torch.bmm(p.dt[:D],ab[h:h+D],out=p.dl[:D]);torch.bmm(p.dt[:D].transpose(-1,-2),ab[:D],out=p.dr[:D])
  torch.bmm(ab[h+D:],p.dt[D:].transpose(-1,-2),out=p.dl[D:]);torch.bmm(ab[D:h],p.dt[D:],out=p.dr[D:])
  W.launch(p.ks['b7'],p.params7,p.grid7,D=D,gp=p.gp_native)
  return p.outputs
 def old():old_b1();return tail()
 def new():candidate();return tail()
 def oldfull():return f(),old()
 def newfull():return f(),new()
 expected=[t.clone() for t in old()];dt=p.dt.clone();norm=p.tensors[6].clone();dp=p.tensors[7].clone();dg=p.dg.clone()
 p.dt.fill_(float('nan'));p.tensors[6].fill_(float('nan'));p.tensors[7].fill_(float('nan'));p.dg.fill_(float('nan'))
 actual=[t.clone() for t in new()]
 record['errors']={n:error(a,b) for n,a,b in zip(names[1:],actual,expected)}
 record['intermediate_errors']={n:error(a,b) for n,a,b in [('dt',p.dt,dt),('norm',p.tensors[6],norm),('dp',p.tensors[7],dp),('dg',p.dg,dg)]}
 record['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 record['cubins']=[str(candidate.norm_cubin),str(candidate.ln_cubin)]
 if candidate.exact_dwp is not None:record['cubins'].append(str(candidate.exact_dwp.cubin))
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  functions=dict(old_b1=old_b1,new_b1=candidate,old_bwd=old,new_bwd=new,old_full=oldfull,new_full=newfull)
  if os.environ.get('COMPARE_TRITON')=='1':
   from miniworld_engine import settings
   settings.configure(engine_backend='triton',trimul_sm90_kernels=(),autotune_miss_cap=24)
   tls=tuple(t.detach().clone().requires_grad_(True) for t in leaves)
   compiled=torch.compile(triton,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
   def baseline():
    with torch.enable_grad():
     y=compiled(*tls,mask,ds);return y,torch.autograd.grad(y,tls,dy)
   bs=torch.cuda.Stream();bs.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(bs):ty,tg=baseline()
   torch.cuda.current_stream().wait_stream(bs)
   frozen=[ty.clone(),*[t.clone() for t in tg]];cy,cg=newfull()
   record['triton_errors']={n:error(a,b) for n,a,b in zip(names,[cy,*cg],[ty,*tg])}
   assert all(v<(.005 if n=='y' else .01) for n,v in record['triton_errors'].items()),record['triton_errors']
   import validate_engine as benchmark
   original_capture=benchmark.capture
   def capture_baseline(fn):
    if fn is not baseline:return original_capture(fn)
    bs.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(bs):
     for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(bs)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=bs):out=fn()
    graph.replay();torch.cuda.synchronize()
    ge=[error(a,b) for a,b in zip([out[0],*out[1]],frozen)]
    assert max(ge)<5e-6,ge
    record['triton_graph_errors']=ge
    return graph,out
   benchmark.capture=capture_baseline
   functions['triton_full']=baseline
   record['baseline']='Existing static-compiled Triton heuristic-24; full CUDA graph replay.'
  record['times']=paired(functions)
  if 'triton_full' in functions:record['speedup_vs_triton']=record['times']['triton_full']['median_us']/record['times']['new_full']['median_us']
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
