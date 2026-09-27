from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint3 import Training
from wide_dense_output import DenseOutput
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-wide-dense-output-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;b=plan.b1;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();original=plan.product_output
 for threads in (128,256):
  out=DenseOutput(plan.f,p,(b.proj,b.gate),threads)
  def old():plan.product_output=original;return plan()
  def new():plan.product_output=out;return plan()
  yn,gn=new();torch.cuda.synchronize()
  es={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  row=dict(threads=threads,cubin=str(out.cubin),grid=out.grid,errors=es)
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'].append(row);print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
  if row['strict']:
   row['times']=paired(dict(old_output=original,new_output=out,old_full=old,new_full=new))
   print('TIMES',threads,{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
