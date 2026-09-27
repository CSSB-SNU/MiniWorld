from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint10 import Training as Wide
from d256_gate_checkpoint import Training as D256,configure
from wide_cached_forward_norm import CachedForwardNorm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384)[index//2];N=(384,768)[index%2]
if D==256:configure()
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-cached-forward-norm-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=(D256 if D==256 else Wide)(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]]
 owner=plan.f if D==256 else plan;key='output' if D==256 else 'product_output';original=getattr(owner,key)
 for rows in (16,32):
  op=CachedForwardNorm(p,original,rows)
  def old():setattr(owner,key,original);return plan()
  def new():setattr(owner,key,op);return plan()
  yn,gn=new();torch.cuda.synchronize()
  v=dict(rows=rows,cubin=str(op.cubin))
  v['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
  v['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in v['errors'].items())
  print('CHECK',v,flush=True)
  if v['strict']:
   v['times']=paired(dict(old_norm=lambda:original.norm.launch((original.grid,1,1),(original.threads,1,1),[original.np],original.smem),new_norm=lambda:op.norm.launch((op.grid,1,1),(op.threads,1,1),[op.np],op.smem),old_full=old,new_full=new))
   print('TIMES',{k:t['median_us'] for k,t in v['times'].items()},flush=True)
  record['candidates'].append(v);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
