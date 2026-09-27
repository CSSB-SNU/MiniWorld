from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
from blas_b1 import BlasB1
sys.path.insert(0,str(THIS));from ln import LN
N=int(os.environ.get('LENGTH','384'))
leaves,dy,mask,ds,*_=setup(256,N)
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn))
 b1=B1(p);b1();old=BlasB1(p,b1);fast=LN(p)
 def baseline():W.launch(old.k,p.params,132,D=256)
 baseline();expected=[x.clone() for x in (p.dt,p.floats[10],p.floats[11])];fast()
 errors=[error(x,y) for x,y in zip((p.dt,p.floats[10],p.floats[11]),expected)]
 print('ERROR',errors,flush=True);assert errors[0]<2e-5 and max(errors[1:])<5e-6
 times=paired(dict(old=baseline,new=fast));print({k:v['median_us'] for k,v in times.items()},flush=True)
 (THIS/f'ln-L{N}.json').write_text(json.dumps(dict(errors=errors,times=times),indent=2))
