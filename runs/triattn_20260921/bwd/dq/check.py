"""Independent dQ checks and balanced graph timings."""
from pathlib import Path
import argparse,json,statistics
import torch,triton
from native import extension,ARTIFACT_ROOT
from miniworld_engine.kernels.triangle_attention.triton import main as core
from miniworld_engine.autotune.shape_key import token_key,pack
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--mask',choices=['none','mixed','one_key','all_masked'],default='mixed');ap.add_argument('--no-bench',action='store_true');ap.add_argument('--native-only',action='store_true');a=ap.parse_args()
L=a.length;torch.manual_seed(95331);ext=extension()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
q,k,v,dy=[torch.randn(1,L,L,128,device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4) for _ in range(4)]
b=torch.randn(1,4,L,L,device='cuda',dtype=torch.bfloat16)*.5
if a.mask=='mixed':b[...,::3]=torch.finfo(b.dtype).min
if a.mask in ('one_key','all_masked'):
 b.fill_(torch.finfo(b.dtype).min)
 if a.mask=='one_key':b[...,7]=0
out,m=core._tri_attn_fwd(q,k,v,b,token_key(L))
delta=torch.empty((1,4,L,L),device='cuda',dtype=torch.float32)
grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),4*L,1]
core._attn_bwd_preprocess[grid](out,dy,delta,*out.stride(),*dy.stride(),4*L,1,L,32,shape_key=pack(token_key(L),HEAD_DIM=32),HEAD_DIM_PAD=32)
def baseline():
 dq=torch.empty((1,L,L,128),device='cuda',dtype=torch.bfloat16).view(1,L,L,4,32).permute(0,3,1,2,4)
 grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),1,4*L]
 core._attn_bwd_dq[grid](q,k,v,b,32**-.5,dy,dq,m,delta,*q.stride(),*dq.stride(),*dy.stride(),*b.stride(),L,4*L,32,HEAD_DIM_PAD=32,shape_key=pack(token_key(L),HEAD_DIM=32))
 return dq
def candidate():return ext.backward(q,k,v,b,m,delta,dy)
got=candidate();torch.cuda.synchronize();assert torch.isfinite(got).all();assert torch.equal(got,candidate())
result=dict(length=L,mask=a.mask,native_build=json.loads((ARTIFACT_ROOT/'build-ready.json').read_text()))
if a.native_only:
 result['native_only']=True;a.output.write_text(json.dumps(result)+'\n');raise SystemExit()
ref=baseline();err=float((got.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-8));result['baseline_relative_l2']=err
if a.mask=='all_masked':assert torch.count_nonzero(got)==0 and torch.count_nonzero(ref)==0
else:assert err<.015,err
if L<=128 and a.mask!='all_masked':
 qq=q.double().requires_grad_();kr,vr=k.double(),v.double();probs=torch.softmax(qq@kr.transpose(-1,-2)*(32**-.5)+b.double().unsqueeze(2),dim=-1)
 exact=torch.autograd.grad(probs@vr,qq,dy.double())[0]
 be=float((ref.double()-exact).square().mean().sqrt());ce=float((got.double()-exact).square().mean().sqrt())
 if float(exact.norm())<1e-10:assert ce<=be*1.10+2e-7,(be,ce)
 else:
  br=float((ref.double()-exact).norm()/exact.norm());cr=float((got.double()-exact).norm()/exact.norm());assert cr<=br*1.15+.0006 and cr<.03,(br,cr)
 result['fp64_rms']=dict(baseline=be,candidate=ce)
print('CORRECT',result,flush=True)
if not a.no_bench:
 def capture(fn):
  st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(st):
   for _ in range(3):y=fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=st):y=fn()
  return g,y
 graphs=[capture(fn) for fn in (baseline,candidate)];times=[[],[]];ratios=[]
 for rnd in range(12):
  pair={}
  for i in ((0,1) if rnd%2==0 else (1,0)):
   g,y=graphs[i];g.replay();st,en=[torch.cuda.Event(enable_timing=True) for _ in range(2)];st.record()
   for _ in range(30):g.replay()
   en.record();en.synchronize();pair[i]=st.elapsed_time(en)/30;times[i].append(pair[i])
  ratios.append(pair[0]/pair[1])
 result.update(baseline_ms=statistics.median(times[0]),candidate_ms=statistics.median(times[1]),speedup=statistics.median(ratios),rounds_ms=times)
 print('TIMED',result['baseline_ms'],result['candidate_ms'],result['speedup'],flush=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
