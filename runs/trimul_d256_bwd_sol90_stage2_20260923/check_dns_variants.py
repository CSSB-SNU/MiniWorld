from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from dn_slim import DNSlim
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates={})
with torch.no_grad(),T.native_context(leaves[0].device):
 os.environ['DN_SLIM']='0';plan=Training(leaves,mask,ds,dy);plan();p=plan.p
 expected=[x.clone() for x in (p.dt,p.floats[10],p.floats[11])]
 def old():torch.mm(p.tensors[7],leaves[6],out=p.tensors[9]);plan.b1.ln()
 fns={'base':old}
 configs=((32,0,0,0),(32,1,0,0),(16,0,0,0),(16,1,0,0))
 if os.environ.get('DNS_SWEEP')=='metadata':configs=((32,1,0,0),(32,1,1,0),(32,1,0,1),(32,1,1,1))
 for rows,read,gamma,stats in configs:
  os.environ.update(DNS_LN_ROWS=str(rows),DNS_STORE_READ=str(read),DNS_CACHE_GAMMA=str(gamma),DNS_STATS_TMA=str(stats));name=f'r{rows}-read{read}-g{gamma}-s{stats}'
  cand=DNSlim(p);cand()
  es=[error(x,y) for x,y in zip((p.dt,p.floats[10],p.floats[11]),expected)]
  assert es[0]==0 and max(es)<5e-6,(name,es)
  record['candidates'][name]=dict(errors=es,grid=cand.grid,cubin=str(cand.cubin),ptxas=cand.cubin.with_suffix('.ptxas.log').read_text())
  print('CANDIDATE',name,record['candidates'][name],flush=True);fns[name]=cand
 record['times']=paired(fns);print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True
(THIS/f'result-dns-variants-L{N}-{record["job"]}.json').write_text(json.dumps(record,indent=2))
