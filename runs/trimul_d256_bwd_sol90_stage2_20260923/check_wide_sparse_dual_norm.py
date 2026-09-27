from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint9 import Training
from wide_sparse_dual_norm import SparseDualNorm,PatchBackwardNorm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
D=512;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-wide-sparse-dual-norm-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]];norm=p.tensors[6].clone();stats=[p.floats[i].clone() for i in (5,6)]
 original=plan.product_output;oldnorm=plan.b1.norm
 for cap in (0,None):
  op=SparseDualNorm(p,original,capacity=cap);back=PatchBackwardNorm(op)
  def old():plan.product_output=original;plan.b1.norm=oldnorm;return plan()
  def new():plan.product_output=op;plan.b1.norm=back;return plan()
  yn,gn=new();torch.cuda.synchronize()
  v=dict(capacity=op.capacity,patches=int(op.count.item()),norm_error=error(p.tensors[6],norm),stats_errors=[error(p.floats[i],x) for i,x in zip((5,6),stats)],cubin=str(op.cubin),fallback_cubin=str(op.fallback_cubin))
  v['fallback']=v['patches']>v['capacity']
  if cap==0:assert v['fallback'],'Forced overflow was not exercised'
  v['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
  v['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in v['errors'].items())
  print('CHECK',v,flush=True)
  if v['strict'] and cap is None:
   v['times']=paired(dict(old_full=old,new_full=new))
   print('TIMES',{k:t['median_us'] for k,t in v['times'].items()},flush=True)
  record['candidates'].append(v);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
