from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint11 import Training
from wide_ln_stats import LNStats
from lt_contract import LtBmm
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,algorithms=[])
path=THIS/f'result-wide-ln-stats-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]];dt=p.dt.clone();dn=p.tensors[9].clone();oldln=plan.ln;oldrun=plan.schedule.run
 transposed=p.x.new_empty((2*D,p.M));workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
 mm=LtBmm(p.tensors[7].unsqueeze(0),plan.b1.wp.unsqueeze(0),transposed.t().unsqueeze(0),workspace)
 best=None
 for idx in mm.indices[:12]:
  mm.index=idx;mm();torch.cuda.synchronize();es=error(transposed.t(),dn)
  v=dict(index=idx,error=es,algo=list(mm.heuristics[idx].algo.data))
  if es==0:
   v['times']=paired(dict(transposed_dn=mm));us=v['times']['transposed_dn']['median_us']
   if best is None or us<best[0]:best=(us,idx)
  record['algorithms'].append(v);path.write_text(json.dumps(record,indent=2))
  print('ALGO',idx,es,None if 'times' not in v else v['times']['transposed_dn']['median_us'],flush=True)
 assert best is not None,'No bitwise transposed dNorm algorithm'
 mm.index=best[1];record['selected']=best[1];op=LNStats(p,transposed,32 if D==384 else 16)
 def newrun(name):
  if name=='dn':mm()
  else:oldrun(name)
 def old():plan.schedule.run=oldrun;plan.ln=oldln;return plan()
 def new():plan.schedule.run=newrun;plan.ln=op;return plan()
 yn,gn=new();torch.cuda.synchronize()
 record['dt_error']=error(p.dt,dt);record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
 record['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in record['errors'].items())
 print('CHECK',record['dt_error'],record['strict'],record['errors'],flush=True)
 if record['strict']:
  record['times']=paired(dict(old_dn=lambda:oldrun('dn'),new_dn=mm,old_ln=oldln,new_stats=op.statistics,new_ln=op,old_full=old,new_full=new))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
