from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint7 import Training
from wide_four_group_source import FourGroupSource
from wide_cached_input import CachedInput
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-four-group-d512-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();gp=p.gp_all.clone()
 original=plan.b7.source_only;old_front=plan.f.front;old_input=plan.dx.reduce_only
 source=FourGroupSource(p,plan.b7,async_dw=False)
 reduce=CachedInput(p,16,128,4,splits=32) if D==512 else old_input
 def old():
  plan.b7.source_only=original;plan.f.front=old_front;plan.dx.reduce_only=old_input
  return plan()
 def new():
  plan.b7.source_only=source;plan.f.front=plan.baseline_front;plan.dx.reduce_only=reduce
  return plan()
 record['cubin']=str(source.cubin);print('START',record,flush=True)
 p.gp_all.fill_(float('nan'));yn,gn=new();torch.cuda.synchronize()
 record['gp_error']=error(p.gp_all,gp);assert record['gp_error']==0,record
 record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
 record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_source=original,new_source=source,old_full=old,new_full=new))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
