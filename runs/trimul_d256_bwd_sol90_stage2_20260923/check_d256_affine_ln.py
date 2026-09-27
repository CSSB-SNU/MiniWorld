from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_gate_checkpoint import Training,configure
from wide_affine_output_ln import AffineOutputLN
from wide_mixed_affine_ln import MixedAffineLN
configure()
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-d256-affine-ln-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]];dt=p.dt.clone();oldln=plan.b1.ln
 for cls in (AffineOutputLN,MixedAffineLN):
  op=cls(p)
  def old():plan.b1.ln=oldln;return plan()
  def new():plan.b1.ln=op;return plan()
  yn,gn=new();torch.cuda.synchronize()
  v=dict(kind=cls.__name__,occupancy=op.occupancy,cubin=str(op.cubin),dt_error=error(p.dt,dt))
  v['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
  v['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in v['errors'].items())
  print('CHECK',v,flush=True)
  if v['strict']:
   v['times']=paired(dict(old_ln=oldln,new_ln=op,old_full=old,new_full=new))
   print('TIMES',{k:t['median_us'] for k,t in v['times'].items()},flush=True)
  record['candidates'].append(v);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
