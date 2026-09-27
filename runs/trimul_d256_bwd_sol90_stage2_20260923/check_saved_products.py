from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from saved_input_stats import enable as input_enable
from saved_products import enable
N=int(os.environ.get('LENGTH',('384','768')[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]))
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),complete=False)
path=THIS/f'result-saved-products-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);record['control']=input_enable(plan,gamma_cache=True)
 y,_=plan();yold=y.clone();expected=[x.clone() for x in plan.p.outputs];old_output,old_prepare=plan.f.output,plan.b1.prepare
 dp=plan.p.tensors[7].clone();dg=plan.p.dg.clone();gate=old_prepare.gate.clone()
 record['candidate']=enable(plan);new_output,new_prepare=plan.f.output,plan.b1.prepare
 y,_=plan();torch.cuda.synchronize()
 es={n:error(x,y) for n,x,y in zip(names[1:],plan.p.outputs,expected)}
 good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items()) and error(y,yold)==0
 record.update(errors=es,forward_error=error(y,yold),strict=good,dp=error(plan.p.tensors[7],dp),dg=error(plan.p.dg,dg),gate=error(old_prepare.gate,gate))
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if good:
  def make(output,prepare,mode):
   def run():
    prior=plan.f.output,plan.b1.prepare;plan.f.output,plan.b1.prepare=output,prepare
    try:return plan() if mode=='full' else plan.backward() if mode=='bwd' else plan.forward()
    finally:plan.f.output,plan.b1.prepare=prior
   return run
  fns=dict(old_prepare=old_prepare,new_prepare=new_prepare)
  for mode in ('fwd','bwd','full'):
   fns['old_'+mode]=make(old_output,old_prepare,mode);fns['new_'+mode]=make(new_output,new_prepare,mode)
  record['times']=paired(fns);print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
