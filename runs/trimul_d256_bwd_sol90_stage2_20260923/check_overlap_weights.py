from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from overlap_weights import OverlapWeights
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(256,384,512)[index//2];N=(384,768)[index%2]
if D==256:from d256_prefix_checkpoint import Training
else:from wide_checkpoint6 import Training
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-overlap-weights-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in g]
 modes=((True,False),(False,True),(True,True)) if D==512 else ((True,False),)
 for output,inp in modes:
  op=OverlapWeights(plan,output,inp);actual=op();torch.cuda.synchronize()
  row=dict(output=output,input=inp,errors={n:error(a,b) for n,a,b in zip(names[1:],actual,expected)})
  row['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in row['errors'].items())
  def new():return plan.forward(),op()
  if row['strict']:
   graph,out=capture(new);graph.replay();torch.cuda.synchronize()
   row['graph_errors']={n:error(a,b) for n,a,b in zip(names[1:],out[1],expected)}
   assert max(row['graph_errors'].values())<5e-6,row
   del graph,out
   row['times']=paired(dict(old_bwd=plan.backward,new_bwd=op,old_full=plan,new_full=new))
   print('TIMES',output,inp,{k:v['median_us'] for k,v in row['times'].items()},flush=True)
  record['candidates'].append(row);print('CHECK',row['errors'],flush=True);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
