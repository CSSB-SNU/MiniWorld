from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_gate_checkpoint import Training
from d256_warp_contract import WarpContract
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-warp-contract-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dl=p.dl.clone();dr=p.dr.clone()
 op=WarpContract(plan);original=plan.schedule.run
 def oldrun():
  for name in ('bc0','bc1','bc2','bc3'):original(name)
 def newrun(name):
  if name=='bc0':op()
  elif name not in ('bc1','bc2','bc3'):original(name)
 def old():plan.schedule.run=original;return plan()
 def new():plan.schedule.run=newrun;return plan()
 yn,gn=new();torch.cuda.synchronize()
 record.update(cubin=str(op.cubin),dl_error=error(p.dl,dl),dr_error=error(p.dr,dr))
 record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
 record['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in record['errors'].items())
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_contract=oldrun,new_contract=op,old_full=old,new_full=new))
  print('TIMES',{k:t['median_us'] for k,t in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
