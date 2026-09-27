from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_prefix_checkpoint import Training
from d256_lt_schedule import LtSchedule
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=256;N=(384,768)[index]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,operations={})
path=THIS/f'result-d256-lt-schedule-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in g];oldy=y.clone()
 schedule=LtSchedule(plan)
 for name,kernel in schedule.kernels.items():
  plan();schedule.original(name);wanted=kernel.out.clone();rows=[];functions={'old':lambda name=name:schedule.original(name)}
  for index in kernel.indices:
   kernel.index=index;kernel.out.fill_(float('nan'));kernel();torch.cuda.synchronize()
   err=error(kernel.out,wanted);row=dict(index=index,error=err,algo=list(kernel.heuristics[index].algo.data))
   rows.append(row)
   if err==0:
    def run(index=index,kernel=kernel):kernel.index=index;kernel()
    functions[f'lt{index}']=run
  times=paired(functions);valid=[]
  for row in rows:
   key=f'lt{row["index"]}'
   if key in times:row['time_us']=times[key]['median_us'];valid.append(row)
  best=min(valid,key=lambda row:row['time_us']) if valid else None
  chosen=best if best and best['time_us']<times['old']['median_us']*.99 else None
  if chosen:kernel.index=chosen['index'];schedule.selected[name]=chosen['index']
  record['operations'][name]=dict(candidates=rows,old_us=times['old']['median_us'],selected=chosen)
  print('SELECT',name,record['operations'][name]['old_us'],chosen,flush=True);path.write_text(json.dumps(record,indent=2))
  del wanted
 yn,gn=schedule();es={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
 record['errors']=es;record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
 print('CHECK',record['strict'],es,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_forward=plan.forward,new_forward=schedule.forward,old_backward=plan.backward,new_backward=schedule.backward,old_full=plan,new_full=schedule))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
