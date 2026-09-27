from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923';sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS));from cute_prepare import CutePrepare
from blas_prepare import BlasPrepare
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,*_=setup(256,N);records=[]
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn));old=BlasPrepare(p,leaves[6],leaves[5]);old();expected=[x.clone() for x in (p.tensors[7],p.dg)]
 for tm,tn,ping,cm in [(128,128,0,1),(64,256,0,1),(128,256,1,1),(128,256,0,1),(128,128,1,2)]:
  config=dict(CUTE_TM=str(tm),CUTE_TN=str(tn),CUTE_PING=str(ping),CUTE_CM=str(cm));os.environ.update(config)
  try:
   new=CutePrepare(p,leaves[6],leaves[5]);new()
   errors=[error(x,y) for x,y in zip((p.tensors[7],p.dg),expected)];times=paired(dict(old=old,new=new,projection=new.projection))
   record=dict(config=config,errors=errors,times=times)
   print(config,errors,{k:v['median_us'] for k,v in times.items()},flush=True)
  except Exception as e:
   record=dict(config=config,exception=repr(e));print(record,flush=True)
  records.append(record);(THIS/f'tune-cute-L{N}.json').write_text(json.dumps(records,indent=2))
