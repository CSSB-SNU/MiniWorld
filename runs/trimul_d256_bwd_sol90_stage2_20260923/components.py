from pathlib import Path
import sys,os,json
R=Path(__file__).resolve().parent;PRE=R.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
from blas_b1 import BlasB1
from validate_engine import paired
N=int(os.environ.get('LENGTH','384'))
leaves,dy,mask,ds,*_=setup(256,N)
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f()
 p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn))
 b1=B1(p);b1();q=BlasB1(p,b1);q.wp=leaves[6]
 def prep():b1.prepare.launch((b1.sms,1,1),(128*b1.groups,1,1),[b1.params],b1.smem)
 def dn():torch.mm(p.tensors[7],q.wp,out=p.tensors[9])
 def dwp():torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp)
 def dwg():torch.mm(p.dg.t(),p.xn.reshape(p.M,256),out=p.dwg)
 def ln():W.launch(q.k,p.params,132,D=256)
 times=paired(dict(prepare=prep,dn=dn,dwp=dwp,dwg=dwg,ln=ln))
 print({k:v['median_us'] for k,v in times.items()},flush=True)
 (Path(__file__).resolve().parent/f'components-L{N}.json').write_text(json.dumps(times,indent=2))
