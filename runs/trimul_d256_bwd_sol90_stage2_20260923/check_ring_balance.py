"""Check smaller ring working sets after NCU showed substantial HBM traffic."""
from pathlib import Path
import os,sys,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from ring3 import Ring3
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates={})
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);plan()
 expected=[x.clone() for x in plan.p.outputs];base=plan.b7
 def make(cand,full):
  def run():
   if full:plan.forward()
   prior=plan.b7;plan.b7=cand
   try:return plan.backward()
   finally:plan.b7=prior
  return run
 fns={'base_bwd':make(base,False),'base_full':make(base,True)}
 for name,cohorts,consumers,slots in (
     ('c4n16s16',4,16,16),('c4n24s16',4,24,16),
     ('c6n12s16',6,12,16),('c6n12s8',6,12,8)):
  os.environ.update(RCOHORTS=str(cohorts),RCONSUMERS=str(consumers),RSLOTS=str(slots))
  cand=Ring3(plan.p,leaves);bwd=make(cand,False);out=bwd()
  es={n:error(x,y) for n,x,y in zip(names[1:],out,expected)}
  good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'][name]=dict(errors=es,strict=good,ctas=cand.ctas,ring_bytes=cand.ring.numel()*cand.ring.element_size())
  print('CANDIDATE',name,record['candidates'][name],flush=True)
  if good:fns[name+'_bwd']=bwd;fns[name+'_full']=make(cand,True)
 record['times']=paired(fns)
 print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True
record['config']={k:v for k,v in os.environ.items() if k.startswith(('GP_','DX_','LN_','SAVE_'))}
(THIS/f'result-ring-balance-L{N}-{record["job"]}.json').write_text(json.dumps(record,indent=2))
