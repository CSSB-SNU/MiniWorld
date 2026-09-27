from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint5 import Training
from wide_saved_front_tma import SavedFrontTma
from wide_saved_pipe import SavedPipe
from wide_split_input import SplitTmaInput
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-wide-saved-pipe-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();gp=p.gp_all.clone()
 original_front=plan.f.front;original_source=plan.b7.source_only;original_reduce=plan.dx.reduce_only
 pre=p.x.new_empty((D//8,p.M,64));front=SavedFrontTma(original_front,pre)
 for splits in (32,8 if N==384 else 16):
  source=SavedPipe(p,plan.b7,pre,splits);reduce=SplitTmaInput(p,16,128,4,splits=splits)
  row=dict(splits=splits,source_cubin=str(source.cubin),front_cubin=str(front.cubin))
  def old():plan.f.front=original_front;plan.b7.source_only=original_source;plan.dx.reduce_only=original_reduce;return plan()
  def new():plan.f.front=front;plan.b7.source_only=source;plan.dx.reduce_only=reduce;return plan()
  pre.fill_(float('nan'));p.gp_all.fill_(float('nan'));p.floats[7].fill_(float('nan'))
  yn,gn=new();torch.cuda.synchronize()
  row['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  row['gp_error']=error(p.gp_all,gp)
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in row['errors'].items())
  record['candidates'].append(row);print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
  if row['strict']:
   row['times']=paired(dict(old_front=original_front,new_front=front,old_source=original_source,new_source=source,old_full=old,new_full=new))
   print('TIMES',splits,{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
