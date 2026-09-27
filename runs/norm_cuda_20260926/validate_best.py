import json,re
from pathlib import Path
import torch
from miniworld_engine.kernels.norm_cuda import cuda_layernorm,cuda_rmsnorm
root=Path(__file__).parent;torch.set_num_threads(4);torch.manual_seed(737)
results=[]
for path in (root/'bench-0.json',root/'bench-1.json'):
 for row in json.loads(path.read_text()):
  candidates=[(v['train_us'],k) for k,v in row['times'].items() if k.startswith('cuda') and 'train_us' in v]
  _,key=min(candidates);threads,rows=map(int,re.findall(r'\d+',key));m,d=row['M'],row['D'];rms=row['rms'];dt=getattr(torch,row['dtype'].split('.')[-1])
  x=torch.randn(1,m,d,device='cuda',dtype=dt,requires_grad=True);w=torch.randn(d,device='cuda',requires_grad=True);b=None if rms else torch.randn_like(w,requires_grad=True)
  out=cuda_rmsnorm(x,w,threads=threads,rows=rows) if rms else cuda_layernorm(x,w,b,threads=threads,rows=rows)
  xf=x.float();ref=(xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-5)*w).to(dt) if rms else torch.nn.functional.layer_norm(xf,(d,),w,b,1e-5).to(dt)
  tol=.017 if dt==torch.bfloat16 else 4e-5
  torch.testing.assert_close(out,ref,atol=tol,rtol=tol)
  dy=torch.randn_like(out);params=[v for v in (x,w,b) if v is not None]
  ga=torch.autograd.grad(out,params,dy);gb=torch.autograd.grad(ref,params,dy)
  errors=[]
  for g,r in zip(ga,gb):
   err=float((g.float()-r.float()).norm()/(r.float().norm()+1e-10));errors.append(err)
   assert err < (.002 if g.dtype==torch.bfloat16 else .0003), (row,key,errors)
  results.append({k:row[k] for k in ('M','D','rms','dtype')}|{'config':key,'relative_l2':errors,'ok':True})
  del out,ref,ga,gb,x,w,b,dy,params,xf
(root/'selected-config-correctness.json').write_text(json.dumps(results,indent=2));print('SELECTED CONFIG PASS',len(results),flush=True)
