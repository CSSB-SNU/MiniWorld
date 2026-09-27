"""Independent gate/delta checks and balanced graph timings."""
from pathlib import Path
import argparse,json,statistics
import torch,triton
from native import extension,ARTIFACT_ROOT
from miniworld_engine.kernels.triangle_attention.triton import main as core
from miniworld_engine.autotune.shape_key import token_key,pack
ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--mask',choices=['none','mixed','one_key','all_masked'],default='mixed');ap.add_argument('--no-bench',action='store_true');ap.add_argument('--native-only',action='store_true');a=ap.parse_args()
L=a.length;torch.manual_seed(95331);ext=extension()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _dgrad_epilogue
M=L*L
dy,gate,out=[torch.randn(M,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
w=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.08

def baseline():
 dr,dg,aa=_dgrad_epilogue(dy,w,gate,out,shape_key=token_key(L))
 o=out.view(1,L,L,4,32).permute(0,3,1,2,4)
 d=dr.view(1,L,L,4,32).permute(0,3,1,2,4)
 delta=torch.empty((1,4,L,L),device='cuda',dtype=torch.float32)
 grid=lambda META:[triton.cdiv(L,META['BLOCK_M1']),4*L,1]
 core._attn_bwd_preprocess[grid](o,d,delta,*o.stride(),*d.stride(),4*L,1,L,32,shape_key=pack(token_key(L),HEAD_DIM=32),HEAD_DIM_PAD=32)
 return dr,dg,aa,delta.reshape(4,M)
def candidate():return tuple(ext.backward(dy,w,gate,out))
got=candidate();torch.cuda.synchronize()
assert all(torch.isfinite(x).all() for x in got)
assert all(torch.equal(x,y) for x,y in zip(got,candidate()))
result=dict(length=L,native_build=json.loads((ARTIFACT_ROOT/'build-ready.json').read_text()))
if a.native_only:
 result['native_only']=True;a.output.write_text(json.dumps(result)+'\n');raise SystemExit()
ref=baseline();err=[float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-8)) for x,y in zip(got,ref)]
assert max(err)<.01,err
# Delta must use the returned rounded BF16 gradient, never the FP32 accumulator.
exact_delta=(got[0].float().reshape(M,4,32)*out.float().reshape(M,4,32)).sum(-1).T
assert torch.allclose(got[3],exact_delta,rtol=2e-5,atol=1e-5),float((got[3]-exact_delta).abs().max())
result['relative_l2']=err
if L<=128:
 da=dy.double()@w.double();sig=gate.double().sigmoid()
 exact=(sig*da,da*out.double()*sig*(1-sig),sig*out.double())
 errors=[]
 for x,y,z in zip(got[:3],ref[:3],exact):
  be=float((y.double()-z).norm()/z.norm());ce=float((x.double()-z).norm()/z.norm())
  assert ce<be*1.1+.0001,(be,ce)
  errors.append([be,ce])
 result['fp64_errors']=errors
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
