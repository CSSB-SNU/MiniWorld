from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_lt_checkpoint import Training
from ring3 import Ring3
from ring_stream import RingStream
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-ring-stream-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();oldy=y.clone();expected=[t.clone() for t in g]
 def baseline_finish():plan.b7.source_only();plan.dx()
 def full(finish):
  y=plan.forward();sch=plan.schedule
  plan.b1.prepare();sch.run('dn');sch.run('dwp');sch.run('dwg');plan.b1.ln()
  for name in ('bc0','bc1','bc2','bc3'):sch.run(name)
  finish();return y,p.outputs
 for slots in (12,24):
  os.environ.update(RCOHORTS='4',RCONSUMERS='32',RSLOTS=str(slots),RWINDOWS='2' if N==384 else '4')
  old=Ring3(p,leaves);new=RingStream(p,leaves)
  def oldfull():return full(old)
  def newfull():return full(new)
  yn,gn=newfull();torch.cuda.synchronize()
  row=dict(slots=slots,errors={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])})
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in row['errors'].items())
  record['candidates'].append(row);print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
  if row['strict']:
   row['times']=paired(dict(baseline_finish=baseline_finish,old_ring=old,new_ring=new,baseline_full=plan,old_full=oldfull,new_full=newfull))
   print('TIMES',{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
