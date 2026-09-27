from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923';sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS));from pipe3 import Pipe3
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,*_=setup(256,N)
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn));b1=B1(p);fast=Pipe3(b1)
 def old():b1.prepare.launch((b1.sms,1,1),(128*b1.groups,1,1),[b1.params],b1.smem)
 old();expected=[x.clone() for x in (p.tensors[6],p.tensors[7],p.dg,p.floats[5],p.floats[6])];fast()
 errors=[error(x,y) for x,y in zip((p.tensors[6],p.tensors[7],p.dg,p.floats[5],p.floats[6]),expected)]
 print('ERROR',errors,flush=True);assert max(errors)==0
 times=paired(dict(old=old,new=fast));print({k:v['median_us'] for k,v in times.items()},flush=True)
 (THIS/f'pipe3-L{N}.json').write_text(json.dumps(dict(errors=errors,times=times),indent=2))
