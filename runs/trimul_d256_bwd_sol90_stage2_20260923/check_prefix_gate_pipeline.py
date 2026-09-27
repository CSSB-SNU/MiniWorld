from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint14 import Training as WideTraining
from d256_gate_checkpoint import Training as D256Training
from prefix_gate_pipeline import PipelinedGateEpi
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-prefix-gate-pipeline-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=(D256Training if D==256 else WideTraining)(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dp=p.tensors[7].clone();dg=plan.prefix_gate.prefix.clone();original=plan.prefix_gate
 def install(op):
  if D==256:plan.b1.prepare=op
  else:plan.b1.epi=op
 for slots in (1,2):
  op=PipelinedGateEpi(plan,slots)
  def old():install(original);return plan()
  def new():install(op);return plan()
  p.tensors[7].fill_(float('nan'));plan.prefix_gate.prefix.fill_(float('nan'))
  yn,gn=new();torch.cuda.synchronize()
  v=dict(slots=slots,occupancy=op.occupancy,cubin=str(op.cubin),dp_error=error(p.tensors[7],dp),dg_error=error(plan.prefix_gate.prefix,dg))
  v['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
  v['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in v['errors'].items())
  print('CHECK',v,flush=True)
  if v['strict']:
   v['times']=paired(dict(old_epi=original,new_epi=op,old_full=old,new_full=new))
   print('TIMES',{k:t['median_us'] for k,t in v['times'].items()},flush=True)
  record['candidates'].append(v);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
