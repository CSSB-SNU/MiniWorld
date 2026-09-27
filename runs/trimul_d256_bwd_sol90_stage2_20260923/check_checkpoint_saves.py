"""Same-buffer comparison against the validated pre-resumption checkpoint."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from shared_candidate import attach
N=int(os.environ.get('LENGTH',('384','768')[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]))
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),complete=False)
path=THIS/f'result-checkpoint-saves-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy)
 def state():return plan.f.front,plan.f.output,plan.b1.prepare,plan.b7.wide_finish
 def select(s):plan.f.front,plan.f.output,plan.b1.prepare,plan.b7.wide_finish=s
 old=state();y,_=plan();yold=y.clone();expected=[x.clone() for x in plan.p.outputs]
 os.environ['SHARED_CANDIDATE']='saved_products';record['candidate']=attach(plan);new=state()
 y,_=plan();torch.cuda.synchronize()
 es={n:error(x,y) for n,x,y in zip(names[1:],plan.p.outputs,expected)}
 good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items()) and error(y,yold)==0
 record.update(errors=es,forward_error=error(y,yold),strict=good)
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2));assert good
 def make(s,mode):
  def run():
   prior=state();select(s)
   try:return plan() if mode=='full' else plan.backward() if mode=='bwd' else plan.forward()
   finally:select(prior)
  return run
 fns={}
 for mode in ('fwd','bwd','full'):
  fns['old_'+mode]=make(old,mode);fns['new_'+mode]=make(new,mode)
 record['times']=paired(fns)
 print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
 m=N*N;d=256;flops=66*m*d*d+8*d*N**3;payload=20*m*d+44*d*d+48*d+2*m+2*N*d
 bound_us=max(flops/(989.5e12),payload/(3.35e12))*1e6
 record['sol_model']=dict(flops=flops,ideal_bytes=payload,bound_us=bound_us,
   old_pct=100*bound_us/record['times']['old_bwd']['median_us'],new_pct=100*bound_us/record['times']['new_bwd']['median_us'],
   note='Unchanged recomputing-policy comparison model; saved-products removes 6*M*D^2 executed backward FLOPs. This ratio is not measured hardware utilization.')
 print('SOL_MODEL',record['sol_model'],flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
