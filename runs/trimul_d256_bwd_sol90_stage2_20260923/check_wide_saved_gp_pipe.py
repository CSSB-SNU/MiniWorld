from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint6 import Training
from wide_saved_gp_pipe import SavedGpPipe
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
D=512;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-wide-saved-gp-pipe-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();gp=p.gp_all.clone();original=plan.saved_source
 for pipe,major,gf in ((False,False,4),(True,False,4),(True,True,4),(True,True,8)):
  source=SavedGpPipe(original,pipe,major,gf)
  row=dict(pipe=pipe,rank_major=major,grid_factor=gf,cubin=str(source.cubin));print('START',row,flush=True)
  p.gp_all.fill_(float('nan'));source.derivatives();torch.cuda.synchronize()
  row['gp_error']=error(p.gp_all,gp);assert row['gp_error']==0,row
  def old():plan.b7.source_only=original;return plan()
  def new():plan.b7.source_only=source;return plan()
  yn,gn=new();row['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in row['errors'].items())
  record['candidates'].append(row);print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
  if row['strict']:
   row['times']=paired(dict(old_gp=original.derivatives,new_gp=source.derivatives,old_full=old,new_full=new))
   print('TIMES',{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
