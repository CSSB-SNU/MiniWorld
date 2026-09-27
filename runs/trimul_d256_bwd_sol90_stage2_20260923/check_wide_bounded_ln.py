from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_combined import WideTraining
from wide_bounded_ln import BoundedLN,B1WithLN
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),loop=os.environ.get('BOUNDED_LN_LOOP')=='1',complete=False,candidates=[])
path=THIS/f'result-wide-bounded-ln-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=WideTraining(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();wanted=p.dt.clone()
 for threads in (128,256,512):
  ln=BoundedLN(p,plan.b1,threads);b1=B1WithLN(plan.b1,ln)
  def newbwd():b1();plan.contract();plan.b7.source_only();plan.dx();return p.outputs
  def newfull():return plan.forward(),newbwd()
  p.dt.fill_(float('nan'));yn,gn=newfull();torch.cuda.synchronize()
  es={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  row=dict(threads=threads,cubin=str(ln.cubin),grid=ln.grid,errors=es,dt_error=error(p.dt,wanted))
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'].append(row);print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
  if row['strict']:
   row['times']=paired(dict(old_bwd=plan.backward,new_bwd=newbwd,old_full=plan,new_full=newfull,ln=ln))
   print('TIMES',threads,{k:v['median_us'] for k,v in row['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
