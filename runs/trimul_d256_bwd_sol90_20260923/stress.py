"""Compare selected B7 to the existing schedule under changed live inputs."""
from pathlib import Path
import sys,json,os
R=Path(__file__).resolve().parent
exec((R/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
from validate_engine import capture
N=int(os.environ.get('LENGTH','384'))
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
results=[]
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f()
 p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn));b1=B1(p);b7=B7(p,8 if N==384 else 16)
 def start():
  T.pack_into(p.w1,*p.weights);f();b1()
  ab=p.front.ab;d=256;h=512
  torch.bmm(p.dt[:d],ab[h:h+d],out=p.dl[:d]);torch.bmm(p.dt[:d].transpose(-1,-2),ab[:d],out=p.dr[:d])
  torch.bmm(ab[h+d:],p.dt[d:].transpose(-1,-2),out=p.dl[d:]);torch.bmm(ab[d:h],p.dt[d:],out=p.dr[d:])
 def old():start();W.launch(p.ks['b7'],p.params7,p.grid7,D=256,gp=p.gp_native);return p.outputs
 def new():start();return b7()
 graph,_=capture(new)
 for case in ('normal','changed','zero_gamma','mask_zero','dropout_zero'):
  if case=='changed':
   leaves[0].mul_(.7);dy.normal_();leaves[1].mul_(1.1);leaves[2].mul_(.9)
   mask.copy_((torch.rand_like(mask.float())>.3).bfloat16());ds.copy_((torch.rand_like(ds.float())>.4).bfloat16()*(1/.6))
  if case=='zero_gamma':leaves[7][::3]=0;leaves[9][::3]=0
  if case=='mask_zero':mask.zero_()
  if case=='dropout_zero':mask.fill_(1);ds.zero_()
  f.mask.copy_(mask.reshape_as(f.mask));p.mask.copy_(mask.reshape_as(p.mask));b7.mask.copy_(mask.reshape_as(b7.mask))
  expected=[x.clone() for x in old()];actual=[x.clone() for x in new()]
  errors={n:error(x,y) for n,x,y in zip(names[1:],actual,expected)}
  assert all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in errors.items()),(case,errors)
  graph.replay();torch.cuda.synchronize()
  graph_errors={n:error(x,y) for n,x,y in zip(names[1:],p.outputs,actual)}
  assert max(graph_errors.values())<5e-6,(case,graph_errors)
  results.append(dict(case=case,previous=errors,graph=graph_errors));print(case,max(errors.values()),flush=True)
(R/f'stress-L{N}.json').write_text(json.dumps(results,indent=2))
