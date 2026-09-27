from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_combined import WideTraining
from wide_split_source import SplitInputSource
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,algorithms=[])
path=THIS/f'result-wide-split-source-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=WideTraining(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();wanted=p.gp_all.clone()
 source=SplitInputSource(p,plan.b7);partial=source.partial.clone()
 p.gp_all.fill_(float('nan'));source.derivatives();torch.cuda.synchronize()
 record['derivative_error']=error(p.gp_all,wanted);assert record['derivative_error']==0
 for i in source.matmul.indices:
  source.matmul.index=i;source.partial.fill_(float('nan'));source.matmul();plan.dx.reduce_only();torch.cuda.synchronize()
  es={n:error(a,b) for n,a,b in zip(names[2:6],p.outputs[1:5],expected[1:5])}
  row=dict(index=i,weight_errors=es,partial_error=error(source.partial,partial),algo=list(source.matmul.heuristics[i].algo.data))
  if max(es.values())<5e-4:row['time_us']=paired(dict(candidate=source.matmul))['candidate']['median_us']
  record['algorithms'].append(row);print('ALGORITHM',row,flush=True);path.write_text(json.dumps(record,indent=2))
 valid=[r for r in record['algorithms'] if 'time_us' in r]
 if valid:
  winner=min(valid,key=lambda r:r['time_us']);source.matmul.index=winner['index'];record['selected']=winner
  def newbwd():plan.b1();plan.contract();source();plan.dx();return p.outputs
  def newfull():return plan.forward(),newbwd()
  yn,gn=newfull();record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
  if record['strict']:
   record['times']=paired(dict(old_source=plan.b7.source_only,new_source=source,derivatives=source.derivatives,old_bwd=plan.backward,new_bwd=newbwd,old_full=plan,new_full=newfull))
   print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
