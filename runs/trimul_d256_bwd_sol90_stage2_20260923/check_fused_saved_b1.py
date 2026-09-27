from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from shared_candidate import attach
from fused_saved_b1 import enable
N=int(os.environ.get('LENGTH',('384','768')[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]))
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),complete=False)
path=THIS/f'result-fused-saved-b1-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);os.environ['SHARED_CANDIDATE']='saved_products';record['control']=attach(plan)
 y,_=plan();yold=y.clone();expected=[x.clone() for x in plan.p.outputs];old=plan.b1
 dp=plan.p.tensors[7].clone();dg=plan.p.dg.clone();dt=plan.p.dt.clone()
 record['candidate']=enable(plan);new=plan.b1;print('BUILT',record['candidate'],flush=True)
 plan.p.tensors[7].fill_(float('nan'));plan.p.dg.fill_(float('nan'));plan.p.dt.fill_(float('nan'))
 y,_=plan();torch.cuda.synchronize()
 es={n:error(x,y) for n,x,y in zip(names[1:],plan.p.outputs,expected)}
 good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items()) and error(y,yold)==0
 record.update(errors=es,forward_error=error(y,yold),strict=good,dp=error(plan.p.tensors[7],dp),dg=error(plan.p.dg,dg),dt=error(plan.p.dt,dt),poison_coverage=True)
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 print(new.fused.cubin.with_suffix('.ptxas.log').read_text(),flush=True)
 if good:
  def make(b1,full):
   def run():
    prior=plan.b1;plan.b1=b1
    try:return plan() if full else plan.backward()
    finally:plan.b1=prior
   return run
  record['times']=paired(dict(old_b1=old,new_b1=new,old_bwd=make(old,False),new_bwd=make(new,False),old_full=make(old,True),new_full=make(new,True)))
  print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
