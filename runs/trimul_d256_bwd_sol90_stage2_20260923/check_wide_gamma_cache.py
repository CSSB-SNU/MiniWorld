from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint5 import Training
from wide_cached_input import CachedInput
from wide_cached_ln import CachedLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-wide-gamma-cache-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in g];oldy=y.clone()
 old_input=plan.dx.reduce_only;old_ln=plan.ln
 for kind in ('input','output','both'):
  ci=CachedInput(p) if kind in ('input','both') else old_input
  co=CachedLN(p,32 if D==384 else 16,True,False,2) if kind in ('output','both') else old_ln
  def old():plan.dx.reduce_only=old_input;plan.ln=old_ln;return plan()
  def new():plan.dx.reduce_only=ci;plan.ln=co;return plan()
  yn,gn=new();es={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  row=dict(kind=kind,errors=es,strict=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items()))
  record['candidates'].append(row);print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
  if row['strict']:
   row['times']=paired(dict(old_input=old_input,new_input=ci,old_ln=old_ln,new_ln=co,old_full=old,new_full=new))
   print('TIMES',kind,{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
