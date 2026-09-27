"""Native LN/residual fusion vs installed projection + stock LN + residual."""
import argparse,json,os,statistics
from pathlib import Path
import torch
from native import extension,ARTIFACT_ROOT
from miniworld_engine.kernels.triangle_attention import cuda as projection
from miniworld_engine.kernels.layernorm import compile_native as ln
from miniworld_engine.autotune.shape_key import both_key
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=64);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--native-only',action='store_true');a=ap.parse_args()
L=a.length;M=L*L;torch.manual_seed(9334)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
ext=extension()
x=torch.randn(M,128,device='cuda',dtype=torch.bfloat16)
gamma=torch.randn(128,device='cuda',dtype=torch.float32)*.1+1
beta=torch.randn_like(gamma)*.1
z,mean,rstd=ln._dispatch_fwd(x.view(1,L,L,128),gamma,beta,1e-5)
w=[torch.randn(c,128,device='cuda',dtype=torch.bfloat16)*.08 for c in (128,128,128,128,4)]
dy=[torch.randn(M,c,device='cuda',dtype=torch.bfloat16) for c in (128,128,128,128,4)]
residual=torch.randn_like(x)
result=dict(length=L,records=[],native_build=json.loads((ARTIFACT_ROOT/'build-ready.json').read_text()))
for ending in (False,True):
 def baseline():
  dz=projection._dgrad(dy,w)
  dx,dg,db=ln._dispatch_bwd(dz.view(1,L,L,128),x.view(1,L,L,128),gamma,mean,rstd)
  dx=dx.view(M,128)
  if ending:dx=dx.view(1,L,L,128).transpose(1,2).contiguous().view(M,128)
  return dx+residual,dg,db
 def candidate():return ext.backward(dy,w,x,mean,rstd,gamma,residual,L,ending)
 got=candidate();torch.cuda.synchronize()
 if a.native_only:continue
 ref=baseline();errors=[float((aa.float()-bb.float()).norm()/aa.float().norm().clamp_min(1e-8)) for aa,bb in zip(ref,got)]
 assert all(torch.isfinite(v).all() for v in got)
 assert max(errors)<.01,errors
 # Independent FP64 algebra with the explicitly retained BF16 boundaries.
 if L<=64:
  dz=sum(g.double()@ww.double() for g,ww in zip(dy,w)).bfloat16().double()
  xx=x.double();mm=xx.mean(1,keepdim=True);rr=torch.rsqrt(((xx-mm)**2).mean(1,keepdim=True)+1e-5);xh=(xx-mm)*rr
  ww=dz*gamma.double();dxx=(ww-(ww*xh).mean(1,keepdim=True)*xh-ww.mean(1,keepdim=True))*rr
  if ending:dxx=dxx.view(1,L,L,128).transpose(1,2).contiguous().view(M,128)
  exact=[(dxx.bfloat16().double()+residual.double()).bfloat16(),(dz*xh).sum(0),dz.sum(0)]
  fp64err=[float((aa.double()-bb.double()).norm()/aa.double().norm().clamp_min(1e-8)) for aa,bb in zip(exact,got)]
  assert max(fp64err)<.01,fp64err
 else:fp64err=None
 def capture(fn):
  st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(st):
   for _ in range(3):out=fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=st):out=fn()
  return g,out
 graphs=[capture(fn) for fn in (baseline,candidate)];times=[[],[]];ratios=[]
 for rnd in range(12):
  pair={}
  for j in ((0,1) if rnd%2==0 else (1,0)):
   g,out=graphs[j];g.replay();begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)];begin.record()
   for _ in range(30):g.replay()
   end.record();end.synchronize();pair[j]=begin.elapsed_time(end)/30;times[j].append(pair[j])
  ratios.append(pair[0]/pair[1])
 result['records'].append(dict(ending=ending,relative_l2=errors,fp64_relative_l2=fp64err,baseline_ms=statistics.median(times[0]),candidate_ms=statistics.median(times[1]),speedup=statistics.median(ratios),rounds_ms=times))
 a.output.write_text(json.dumps(result,indent=2)+'\n');print('FUSION',ending,result['records'][-1],flush=True)
 del graphs
if a.native_only:a.output.write_text(json.dumps(dict(length=L,native_only=True,complete=True))+'\n')
