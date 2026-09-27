from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_dense_checkpoint import Training
from d256_mask_source import MaskSource
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=256;N=(384,768)[index]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-d256-mask-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();gp=[t.clone() for t in p.gp];part=p.floats[7].clone();original=plan.b7
 def original_source():original.source.launch((32*original.splits,1,1),(original.source_threads,1,1),[original.params],114816)
 for mode in ('prefetch','bulk'):
  source=MaskSource(original,mode);row=dict(mode=mode,cubin=str(source.cubin))
  print('START',row,flush=True)
  for t in p.gp:t.fill_(float('nan'))
  p.floats[7][:original.splits,3*D*D:].fill_(float('nan'))
  source.source_only();torch.cuda.synchronize()
  row['gp_error']=max(error(a,b) for a,b in zip(p.gp,gp))
  row['partial_error']=error(p.floats[7][:original.splits,3*D*D:],part[:original.splits,3*D*D:])
  assert row['gp_error']==0 and row['partial_error']==0,row
  def old():plan.b7=original;return plan()
  def new():plan.b7=source;return plan()
  yn,gn=new();row['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in row['errors'].items())
  record['candidates'].append(row);path.write_text(json.dumps(record,indent=2));print('CHECK',row,flush=True)
  if row['strict']:
   row['times']=paired(dict(old_source=original_source,new_source=source.source_only,old_full=old,new_full=new))
   print('TIMES',mode,{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
