from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint9 import Training
from wide_joint_input_reduce import JointInputReduce
from lt_contract import LtBmm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,choices=[])
path=THIS/f'result-joint-input-dw-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]]
 splits=plan.input_splits;step=p.M//splits
 a=plan.dx.input.as_strided((splits,9*D,step),(step,p.M,1))
 b=p.xn.as_strided((splits,step,D),(step*D,D,1))
 partial=p.floats[7].reshape(-1)[2*D*D:].as_strided((splits,9*D,D),(11*D*D,D,1))
 workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
 op=LtBmm(a,b,partial,workspace);reduce=JointInputReduce(p,16,128,4,splits=splits)
 functions={}
 for idx in op.indices:
  op.index=idx;op();reduce();torch.cuda.synchronize()
  es={n:error(a,b) for n,a,b in zip(names,[y,*p.outputs],expected)}
  strict=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['choices'].append(dict(index=idx,algo=list(op.heuristics[idx].algo.data),errors=es,strict=strict))
  print('CHOICE',idx,strict,es,flush=True)
  if strict:
   def run(idx=idx):op.index=idx;op()
   functions[f'lt{idx}']=run
 if functions:
  times=paired(functions);best=min((v for v in record['choices'] if v['strict']),key=lambda v:times[f'lt{v["index"]}']['median_us'])
  op.index=best['index'];record['selected']=best
  old_dw=plan.input_dw;old_reduce=plan.dx.reduce_only;oldrun=plan.schedule.run
  def newrun(name):
   if name!='dwg':oldrun(name)
  def old():
   plan.input_dw=old_dw;plan.dx.reduce_only=old_reduce;plan.schedule.run=oldrun;return plan()
  def new():
   plan.input_dw=op;plan.dx.reduce_only=reduce;plan.schedule.run=newrun;return plan()
  yn,gn=new();record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
  record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
  if record['strict']:
   def separate():old_dw();plan.prefix_gate_dw()
   record['times']=paired(dict(old_dw=separate,joint_dw=op,old_full=old,new_full=new))
   print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
