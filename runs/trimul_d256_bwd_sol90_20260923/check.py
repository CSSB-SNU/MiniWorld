from pathlib import Path
import sys,os,json,argparse
R=Path(__file__).resolve().parent
ROOT=R.parent.parent
sys.path.insert(0,str(ROOT/'.engine-release-2.0.0/src'))
sys.path.insert(0,str(R.parent/'trimul_cuda_widths_opt_20260923'))
sys.path.insert(0,str(R.parent/'trimul_forward_wide_20260923'))
sys.path.insert(0,str(R.parent/'trimul_backward_wide_20260923'))
from fixture import setup
from validate_engine import error,paired
from plan import B1
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from b7 import B7

ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,default=384);ap.add_argument('--splits',type=int,default=8);ap.add_argument('--sanitize',action='store_true');ap.add_argument('--profile',action='store_true');ap.add_argument('--fused-b1',action='store_true');ap.add_argument('--wide-finish',action='store_true');ap.add_argument('--ring',action='store_true');ap.add_argument('--blas-b1',action='store_true');a=ap.parse_args()
leaves,dy,mask,ds,ref,triton,names=setup(256,a.length)
record=dict(L=a.length,D=256,splits=a.splits,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=R/f'result-L{a.length}-s{a.splits}{"-b1" if a.fused_b1 else ""}{"-wide" if a.wide_finish else ""}{"-ring" if a.ring else ""}{"-blas" if a.blas_b1 else ""}.json'
if os.environ.get('B7_LOOP')=='1':path=path.with_stem(path.stem+'-loop')
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f()
 p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn))
 b1=B1(p);b7=B7(p,a.splits)
 if a.ring:
  from ring import RingB7
  b7=RingB7(p)
 if a.fused_b1:
  from b1 import FusedB1
  new1=FusedB1(p)
 else:new1=b1
 if a.blas_b1:
  from blas_b1 import BlasB1
  new1=BlasB1(p,b1);new1.wp=leaves[6]
 if a.wide_finish:
  from wide_finish import Finish
  finish1=Finish(p,'b1',32);b7.wide_finish=Finish(p,'b7',a.splits)
  def new1():
   b1.prepare.launch((b1.sms,1,1),(128*b1.groups,1,1),[b1.params],b1.smem)
   finish1()
 p.backward()
 def old7():return W.launch(p.ks['b7'],p.params7,p.grid7,D=256,gp=p.gp_native)
 def contract():
  ab=p.front.ab;d=256;h=512
  torch.bmm(p.dt[:d],ab[h:h+d],out=p.dl[:d]);torch.bmm(p.dt[:d].transpose(-1,-2),ab[:d],out=p.dr[:d])
  torch.bmm(ab[h+d:],p.dt[d:].transpose(-1,-2),out=p.dl[d:]);torch.bmm(ab[d:h],p.dt[d:],out=p.dr[d:])
 def old():b1();contract();old7();return p.outputs
 def new():new1();contract();b7();return p.outputs
 def oldfull():f();return old()
 def newfull():f();return new()
 if a.sanitize:
  newfull();torch.cuda.synchronize();print('SANITIZER_DONE',flush=True);raise SystemExit
 if a.profile:
  for _ in range(3):newfull()
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();new();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop();raise SystemExit
 previous=[t.clone() for t in old()];result=[t.clone() for t in new()]
 record['previous']={n:error(x,y) for n,x,y in zip(names[1:],result,previous)}
 record['strict_previous_pass']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['previous'].items())
 print('PREVIOUS',record['previous'],flush=True);path.write_text(json.dumps(record,indent=2))
 assert max(record['previous'].values())<.005,record
 if not a.fused_b1:assert record['strict_previous_pass'],record
reference=torch.compile(ref);y=reference(*leaves,mask,ds);grad=torch.autograd.grad(y,leaves,dy)
record['pytorch']={n:error(x,y) for n,x,y in zip(names[1:],result,grad)}
print('PYTORCH',record['pytorch'],flush=True);assert max(record['pytorch'].values())<.01
del result,previous,y,grad
with torch.no_grad(),T.native_context(leaves[0].device):
 record['times']=paired(dict(old_b1=b1,new_b1=new1,old_b7=old7,new_b7=b7,old_backward=old,new_backward=new,old_full=oldfull,new_full=newfull))
record['complete']=True;path.write_text(json.dumps(record,indent=2));print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
