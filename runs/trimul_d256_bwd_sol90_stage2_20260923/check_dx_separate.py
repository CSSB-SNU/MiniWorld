"""Paired full-workload test against the validated saved-products checkpoint."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from shared_candidate import attach
from dx_separate import DxSeparate
N=int(os.environ.get('LENGTH',('384','768')[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]))
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),complete=False,candidates={})
path=THIS/f'result-dx-separate-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);os.environ['SHARED_CANDIDATE']='saved_products';record['control']=attach(plan)
 plan();old=plan.b7.wide_finish;expected=[x.clone() for x in plan.p.outputs]
 kernels={'old':old}
 for slots in (2,3):
  name=f'slots{slots}';new=DxSeparate(plan.p,plan.b7.splits,slots)
  plan.p.tensors[10].fill_(float('nan'));plan.p.dx.fill_(float('nan'))
  new();torch.cuda.synchronize()
  es={n:error(x,y) for n,x,y in zip(names[1:],plan.p.outputs,expected)}
  coverage=bool(torch.isfinite(plan.p.tensors[10]).all()) and bool(torch.isfinite(plan.p.dx).all())
  good=coverage and all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'][name]=dict(**new.metadata,errors=es,coverage=coverage,strict=good)
  print('CHECK',name,record['candidates'][name],flush=True);path.write_text(json.dumps(record,indent=2));assert good
  print(new.cubin.with_suffix('.ptxas.log').read_text(),flush=True);kernels[name]=new
 def make(dx,mode):
  def run():
   prior=plan.b7.wide_finish;plan.b7.wide_finish=dx
   try:return plan() if mode=='full' else plan.backward()
   finally:plan.b7.wide_finish=prior
  return run
 fns={}
 for name,k in kernels.items():
  fns[name+'_dx']=k;fns[name+'_bwd']=make(k,'bwd');fns[name+'_full']=make(k,'full')
  if name!='old':fns[name+'_gemm']=k.gemm;fns[name+'_ln']=k.normalize
 record['times']=paired(fns);print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
