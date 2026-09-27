"""Paired full-workload test against the validated saved-products checkpoint."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from shared_candidate import attach
from ln import LN
N=int(os.environ.get('LENGTH',('384','768')[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]))
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),complete=False,candidates={})
path=THIS/f'result-ln-gamma-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);os.environ['SHARED_CANDIDATE']='saved_products';record['control']=attach(plan)
 plan();old=plan.b1.ln;expected=[x.clone() for x in plan.p.outputs]
 kernels={'old':old}
 for slots in (1,):
  name='gamma';os.environ['LN_GAMMA_SMEM']='1';new=LN(plan.p);os.environ['LN_GAMMA_SMEM']='0'
  dt_expected=plan.p.dt.clone();plan.p.dt.fill_(float('nan'))
  plan.b1.ln=new;plan.backward();torch.cuda.synchronize();plan.b1.ln=old
  es={n:error(x,y) for n,x,y in zip(names[1:],plan.p.outputs,expected)}
  coverage=bool(torch.isfinite(plan.p.dt).all());dt_error=error(plan.p.dt,dt_expected)
  good=coverage and dt_error==0 and all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'][name]=dict(cubin=str(new.cubin),grid=new.grid,smem=new.smem,errors=es,dt_error=dt_error,coverage=coverage,strict=good)
  print('CHECK',name,record['candidates'][name],flush=True);path.write_text(json.dumps(record,indent=2));assert good
  print(new.cubin.with_suffix('.ptxas.log').read_text(),flush=True);kernels[name]=new
 def make(dx,mode):
  def run():
   prior=plan.b1.ln;plan.b1.ln=dx
   try:return plan() if mode=='full' else plan.backward()
   finally:plan.b1.ln=prior
  return run
 fns={}
 for name,k in kernels.items():
  fns[name+'_ln']=k;fns[name+'_bwd']=make(k,'bwd');fns[name+'_full']=make(k,'full')

 record['times']=paired(fns);print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
