from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from shared_candidate import attach
from saved_products import SavedProductsPrepare
N=int(os.environ.get('LENGTH',('384','768')[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]))
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),complete=False,candidates={})
path=THIS/f'result-saved-epi-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);os.environ['SHARED_CANDIDATE']='saved_products';record['control']=attach(plan);plan()
 old=plan.b1.prepare;expected=[x.clone() for x in plan.p.outputs];dp=plan.p.tensors[7].clone();dg=plan.p.dg.clone()
 # Confirm the fixed-stride kernel is invalid at 128 threads; unchanged
 # fixture data previously hid its unwritten half behind stale gradients.
 plan.p.tensors[7].fill_(float('nan'));plan.p.dg.fill_(float('nan'))
 old.original.epi.launch((1056,1,1),(128,1,1),[old.original.ep],0)
 record['legacy128_missing']=[int(torch.isnan(t).sum()) for t in (plan.p.tensors[7],plan.p.dg)]
 assert record['legacy128_missing']==[plan.p.M*256//2]*2
 print('LEGACY128_COVERAGE_FAILURE',record['legacy128_missing'],flush=True);old()
 def make(prep,full):
  def run():
   prior=plan.b1.prepare;plan.b1.prepare=prep
   try:return plan() if full else plan.backward()
   finally:plan.b1.prepare=prior
  return run
 fns=dict(base_epi=old,base_bwd=make(old,False),base_full=make(old,True));keep=[]
 for threads,grid in ((256,264),(256,1056),(128,1056),(128,2112)):
  name=f't{threads}g{grid}';new=SavedProductsPrepare(old.original,grid,threads);keep.append(new)
  plan.p.tensors[7].fill_(float('nan'));plan.p.dg.fill_(float('nan'))
  make(new,True)();torch.cuda.synchronize()
  es={n:error(x,y) for n,x,y in zip(names[1:],plan.p.outputs,expected)}
  good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items()) and error(plan.p.tensors[7],dp)==error(plan.p.dg,dg)==0
  record['candidates'][name]=dict(errors=es,strict=good,poison_coverage=True);print(name,record['candidates'][name],flush=True);path.write_text(json.dumps(record,indent=2));assert good
  fns[name+'_epi']=new;fns[name+'_bwd']=make(new,False);fns[name+'_full']=make(new,True)
 record['times']=paired(fns);print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
