from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923';sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS));from cute_prepare import CutePrepare
from blas_prepare import BlasPrepare
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,*_=setup(256,N)
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn));old=BlasPrepare(p,leaves[6],leaves[5]);new=CutePrepare(p,leaves[6],leaves[5])
 old();expected=[x.clone() for x in (p.tensors[7],p.dg)];new()
 errors=[error(x,y) for x,y in zip((p.tensors[7],p.dg),expected)];print('ERROR',errors,flush=True)
 times=paired(dict(old=old,new=new,projection=new.projection));print({k:v['median_us'] for k,v in times.items()},flush=True)
 (THIS/f'cute-prepare-{os.environ.get("SLURM_JOB_ID")}.json').write_text(json.dumps(dict(errors=errors,times=times),indent=2))
