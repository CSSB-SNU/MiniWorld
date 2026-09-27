from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint13 import Training
from wide_variable_joint_input_reduce import VariableJointInputReduce
from lt_contract import LtBmm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-joint-dw-splits-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]]
 old_dw=plan.input_dw;old_reduce=plan.dx.reduce_only;oldrun=plan.schedule.run
 workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
 for splits in (32,64):
  step=p.M//splits
  buffer=torch.empty((splits,11*D*D),device=p.x.device,dtype=torch.float32)
  a=plan.dx.input.as_strided((splits,9*D,step),(step,p.M,1))
  b=p.xn.as_strided((splits,step,D),(step*D,D,1))
  partial=buffer.reshape(-1)[2*D*D:].as_strided((splits,9*D,D),(11*D*D,D,1))
  op=LtBmm(a,b,partial,workspace);reduce=VariableJointInputReduce(p,16,128,4,splits=splits,partial=buffer)
  v=dict(splits=splits,choices=[]);functions={}
  for idx in op.indices:
   op.index=idx;op();reduce();torch.cuda.synchronize()
   errors={names[i]:error(p.outputs[i-1],expected[i]) for i in range(2,7)}
   strict=max(errors.values())<5e-4
   choice=dict(index=idx,algo=list(op.heuristics[idx].algo.data),errors=errors,strict=strict);v['choices'].append(choice)
   if strict:
    def run(idx=idx):op.index=idx;op()
    functions[f'lt{idx}']=run
  if functions:
   times=paired(functions);best=min((c for c in v['choices'] if c['strict']),key=lambda c:times[f'lt{c["index"]}']['median_us'])
   op.index=best['index'];v['selected']=best;v['algorithm_medians_us']={k:x['median_us'] for k,x in times.items()}
   def newrun(name):
    if name!='dwg':oldrun(name)
   def old():plan.input_dw=old_dw;plan.dx.reduce_only=old_reduce;plan.schedule.run=oldrun;return plan()
   def new():plan.input_dw=op;plan.dx.reduce_only=reduce;plan.schedule.run=newrun;return plan()
   yn,gn=new();torch.cuda.synchronize()
   v['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
   v['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in v['errors'].items())
   if v['strict']:
    v['times']=paired(dict(old_full=old,new_full=new))
   print('SELECT',splits,best['index'],v['strict'],v['errors'],{k:x['median_us'] for k,x in v.get('times',{}).items()},flush=True)
   old()
  record['candidates'].append(v);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
