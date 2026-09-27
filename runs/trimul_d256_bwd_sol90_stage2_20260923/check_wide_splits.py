"""Change only the input-weight split count; keep GP and dX ordering fixed."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint4 import Training
from wide_split_pipe import SplitPipeSource
from wide_split_input import SplitTmaInput
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-wide-splits-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();gp=p.gp_all.clone()
 old_source,old_reduce=plan.b7.source_only,plan.dx.reduce_only
 for splits in (8,16):
  source=SplitPipeSource(p,plan.b7,use_offsets=True,prefetch=True,n128=False,splits=splits)
  reduce=SplitTmaInput(p,16,128,4,splits)
  def old():plan.b7.source_only=old_source;plan.dx.reduce_only=old_reduce;return plan()
  def new():plan.b7.source_only=source;plan.dx.reduce_only=reduce;return plan()
  p.gp_all.fill_(float('nan'));p.floats[7][:,3*D*D:].fill_(float('nan'))
  yn,gn=new();torch.cuda.synchronize()
  es={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  row=dict(splits=splits,source_cubin=str(source.cubin),reduce_cubin=str(reduce.cubin),gp_error=error(p.gp_all,gp),errors=es)
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'].append(row);print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
  if row['strict']:
   row['times']=paired(dict(old_source=old_source,new_source=source,old_reduce=old_reduce,new_reduce=reduce,old_full=old,new_full=new))
   print('TIMES',splits,{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
