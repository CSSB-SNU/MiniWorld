from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923';sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
from blas_b1 import BlasB1
sys.path.insert(0,str(THIS));from dn_ln import DNLN
from ln import LN
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,*_=setup(256,N)
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn));b1=B1(p);b1();base=LN(p)
 if os.environ.get('DN_SLIM')=='1':
  from dn_slim import DNSlim
  fast=DNSlim(p)
 else:fast=DNLN(p)
 def old():torch.mm(p.tensors[7],leaves[6],out=p.tensors[9]);base()
 old();expected=[x.clone() for x in (p.dt,p.floats[10],p.floats[11])];fast()
 errors=[error(x,y) for x,y in zip((p.dt,p.floats[10],p.floats[11]),expected)];print('ERROR',errors,flush=True);assert max(errors)<5e-6
 times=paired(dict(old=old,new=fast));print({k:v['median_us'] for k,v in times.items()},flush=True)
 (THIS/f'dn-ln-L{N}-{os.environ.get("SLURM_JOB_ID","local")}.json').write_text(json.dumps(dict(errors=errors,times=times,config={k:v for k,v in os.environ.items() if k.startswith(('DN_','LN_'))}),indent=2))
