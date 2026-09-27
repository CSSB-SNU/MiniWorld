from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint5 import Training
from wide_tile_back_norm import TileBackNorm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=512;N=(384,768)[index]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-wide-tile-back-norm-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;b=plan.b1;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();norm=p.tensors[6].clone();stats=[p.floats[k].clone() for k in (5,6)]
 keys=('norm','grid','threads','smem','np');original={k:getattr(b,k) for k in keys}
 def original_norm():original['norm'].launch((original['grid'],1,1),(original['threads'],1,1),[original['np']],original['smem'])
 for rows,threads in ((16,128),(16,256),(32,128),(32,256)):
  candidate=TileBackNorm(p,rows,threads);row=dict(rows=rows,threads=threads,cubin=str(candidate.cubin))
  p.tensors[6].fill_(float('nan'));candidate();torch.cuda.synchronize()
  row['norm_error']=error(p.tensors[6],norm);row['stat_errors']=[error(p.floats[k],s) for k,s in zip((5,6),stats)]
  assert row['norm_error']==0 and max(row['stat_errors'])==0,row
  def old():
   for k,v in original.items():setattr(b,k,v)
   return plan()
  def new():
   for k in keys:setattr(b,k,getattr(candidate,k))
   return plan()
  yn,gn=new();row['errors']={n:error(a,z) for n,a,z in zip(names,[yn,*gn],[oldy,*expected])}
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in row['errors'].items())
  record['candidates'].append(row);path.write_text(json.dumps(record,indent=2));print('CHECK',row,flush=True)
  if row['strict']:
   row['times']=paired(dict(old_norm=original_norm,new_norm=candidate,old_full=old,new_full=new))
   print('TIMES',rows,threads,{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
