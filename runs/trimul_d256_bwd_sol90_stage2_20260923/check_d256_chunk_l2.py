from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_lt_checkpoint import Training
from d256_chunk_l2 import ChunkL2
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-d256-chunk-l2-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();oldy=y.clone();expected=[t.clone() for t in g];dxn=p.tensors[10].clone()
 def baseline_finish():plan.b7.source_only();plan.dx()
 def full(finish):
  y=plan.forward();sch=plan.schedule
  plan.b1.prepare();sch.run('dn');sch.run('dwp');sch.run('dwg');plan.b1.ln()
  for name in ('bc0','bc1','bc2','bc3'):sch.run(name)
  finish();return y,p.outputs
 for rows,splits in ((4096,4),(8192,4),(8192,8),(16384,8)):
  op=ChunkL2(plan,rows,splits);op.stage(0);kernel=op.ops[0]
  choices=[];functions={}
  for index in kernel.indices:
   kernel.index=index;kernel();torch.cuda.synchronize();err=error(kernel.out,dxn[:rows])
   choice=dict(index=index,error=err,algo=list(kernel.heuristics[index].algo.data));choices.append(choice)
   if err==0:
    def run(index=index):kernel.index=index;kernel()
    functions[f'lt{index}']=run
  row=dict(rows=rows,splits=splits,lt=choices,source_cubin=str(op.source_cubin))
  record['candidates'].append(row)
  if not functions:
   row['strict']=False;print('NO_EXACT_LT',row,flush=True);path.write_text(json.dumps(record,indent=2));continue
  times=paired(functions)
  best=min((c for c in choices if c['error']==0),key=lambda c:times[f'lt{c["index"]}']['median_us'])
  row['selected']=best;op.select(best['index'])
  def newfull():return full(op)
  p.tensors[10].fill_(float('nan'));yn,gn=newfull();torch.cuda.synchronize()
  row['dxn_error']=error(p.tensors[10],dxn)
  row['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  row['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in row['errors'].items())
  print('CHECK',row,flush=True);path.write_text(json.dumps(record,indent=2))
  if row['strict']:
   row['times']=paired(dict(baseline_finish=baseline_finish,new_finish=op,baseline_full=plan,new_full=newfull))
   print('TIMES',{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
