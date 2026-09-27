from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint6 import Training
from wide_dual_norm import DualNorm,DualNormBackward
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
D=512;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-dual-norm-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();oldy=y.clone();expected=[t.clone() for t in g];norm=p.tensors[6].clone();stats=[p.floats[i].clone() for i in (5,6)]
 original=plan.product_output;original_bwd=plan.backward
 candidate=DualNorm(p,original);backward=DualNormBackward(plan,candidate)
 def old():plan.product_output=original;plan.backward=original_bwd;return plan()
 def new():plan.product_output=candidate;plan.backward=backward;return plan()
 candidate.legacy.fill_(float('nan'));yn,gn=new();torch.cuda.synchronize()
 record['norm_error']=error(candidate.legacy,norm);record['stats_errors']=[error(p.floats[i],v) for i,v in zip((5,6),stats)]
 record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
 record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 record['cubin']=str(candidate.cubin)
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_full=old,new_full=new))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
