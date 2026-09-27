from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint7 import Training
from wide_compact_ln import CompactLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-compact-ln-D{D}-L{N}-{record["job"]}.json'
record['candidates']=[]
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 oldy=y.clone();expected=[t.clone() for t in g];dt=p.dt.clone();original=plan.ln
 for rows,mb in ((16,3),(16,4),(8,3),(8,4)):
  ln=CompactLN(p,rows,True,False,mb)
  row=dict(rows=rows,minblocks=mb,grid=ln.grid,cubin=str(ln.cubin));ln();torch.cuda.synchronize()
  row['dt_error']=error(p.dt,dt);assert row['dt_error']==0,row
  def old():plan.ln=original;return plan()
  def new():plan.ln=ln;return plan()
  yn,gn=new();row['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in row['errors'].items())
  record['candidates'].append(row);print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
  if row['strict']:
   row['times']=paired(dict(old_ln=original,new_ln=ln,old_full=old,new_full=new))
   print('TIMES',{k:v['median_us'] for k,v in row['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
