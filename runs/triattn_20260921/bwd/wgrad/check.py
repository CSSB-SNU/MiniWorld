"""Paired shared-Z four-wide wgrad against four cuBLAS GEMMs."""
import argparse, gc, json, statistics
from pathlib import Path
import torch
from native import extension, ARTIFACT
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--native-only',action='store_true');p.add_argument('--split',type=int);p.add_argument('--splits',type=int,nargs='+');a=p.parse_args()
torch.manual_seed(95332)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
ext=extension();L=a.length
z=torch.randn(L*L,128,device='cuda',dtype=torch.bfloat16)
dy=[torch.randn_like(z) for _ in range(4)]
report=dict(length=L,artifact=str(ARTIFACT),build=json.loads((ARTIFACT/'build-ready.json').read_text()),records=[])
def capture(fn):
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):out=fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=s):out=fn()
 return g,out
ref=[d.T@z for d in dy] if not a.native_only else None
for split in ([a.split] if a.split else (a.splits or [512,1024,2048,4096,8192])):
 got=ext.backward(dy,z,split);got=list(got);torch.cuda.synchronize()
 assert all(torch.isfinite(t).all() for t in got)
 rec=dict(split=split)
 if not a.native_only:
  err=[float((g.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)) for g,r in zip(got,ref)]
  assert max(err)<.005,err
  rec['relative_l2']=err
  if L<=64:
   fp64=[d.double().T@z.double() for d in dy]
   errs=[float((g.double()-r).norm()/r.norm().clamp_min(1e-8)) for g,r in zip(got,fp64)]
   assert max(errs)<.005,errs
   rec['fp64_relative_l2']=errs
   # Exact cancellation with equal-and-opposite adjacent token contributions.
   cz=torch.randn_like(z);cz[1::2]=cz[::2]
   cd=[torch.randn_like(z) for _ in range(4)]
   for d in cd:d[1::2]=-d[::2]
   zero=ext.backward(cd,cz,split)
   assert all(torch.count_nonzero(t)==0 for t in zero)
   rec['exact_cancellation']=True
  graphs=[capture(lambda:[d.T@z for d in dy]),capture(lambda:ext.backward(dy,z,split))]
  times=[[],[]];ratios=[]
  for rnd in range(10):
   pair={}
   for i in ((0,1) if rnd%2==0 else (1,0)):
    g,out=graphs[i];g.replay();st,en=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    st.record()
    for _ in range(20):g.replay()
    en.record();en.synchronize();pair[i]=st.elapsed_time(en)/20;times[i].append(pair[i])
   ratios.append(pair[0]/pair[1])
  rec.update(baseline_ms=statistics.median(times[0]),candidate_ms=statistics.median(times[1]),speedup=statistics.median(ratios),rounds_ms=times)
  del graphs,g,out;gc.collect();torch.cuda.empty_cache()
 report['records'].append(rec);a.output.write_text(json.dumps(report,indent=2)+'\n');print('WGRAD',L,rec,flush=True)
report['complete']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')
