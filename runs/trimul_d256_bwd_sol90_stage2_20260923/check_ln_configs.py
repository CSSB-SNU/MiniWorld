from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from ln import LN
from hybrid_b1 import HybridB1
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,*_=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),candidates={})
configs={'base':(256,2,0,0),'agg':(256,2,1,0),
         'cache128x3':(128,3,1,1),'cache128x2':(128,2,1,1),
         'agg128x3':(128,3,1,0)}
meta=os.environ.get('LN_SWEEP')=='metadata'
if meta:configs={f's{s}r{r}':(128,3,1,0,s,r) for s,r in ((0,0),(1,0),(0,1),(1,1))}
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f()
 p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn))
 prep=HybridB1(p,leaves,use_cute=True);prep()
 expected=[x.clone() for x in (p.dt,p.floats[10],p.floats[11])]
 fns={}
 for name,values in configs.items():
  for k,v in zip(('LN_THREADS','LN_MINBLOCKS','LN_AGG','LN_CACHE','LN_STATS_TMA','LN_STORE_READ'),values):os.environ[k]=str(v)
  ln=LN(p);ln()
  es=[error(x,y) for x,y in zip((p.dt,p.floats[10],p.floats[11]),expected)]
  assert es[0]==0 and max(es)<5e-6,(name,es)
  meta=dict(errors=es,grid=ln.grid,cubin=str(ln.cubin),ptxas=ln.cubin.with_suffix('.ptxas.log').read_text())
  record['candidates'][name]=meta;fns[name]=ln;print('CANDIDATE',name,meta,flush=True)
 record['times']=paired(fns)
 print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True
(THIS/f'ln-configs-L{N}-{record["job"]}.json').write_text(json.dumps(record,indent=2))
