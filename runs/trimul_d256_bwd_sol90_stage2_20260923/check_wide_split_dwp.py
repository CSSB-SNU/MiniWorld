from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_combined import WideTraining
from wide_split_dwp import SplitProjectionWeight
from wide_hybrid_b1 import ExactProjectionWeight
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
D=512;N=768
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,algorithms=[])
path=THIS/f'result-wide-split-dwp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=WideTraining(leaves,mask,ds,dy);p=plan.p;native=plan.b1.exact_dwp
 y,g=plan();expected=[t.clone() for t in g];oldy=y.clone();wanted=p.dwp.clone()
 candidate=SplitProjectionWeight(p,native);native();partial=candidate.partial.clone()
 for i in candidate.matmul.indices:
  candidate.matmul.index=i;p.dwp.fill_(float('nan'));candidate.partial.fill_(float('nan'))
  candidate();torch.cuda.synchronize()
  row=dict(index=i,error=error(p.dwp,wanted),partial_error=error(candidate.partial,partial),algo=list(candidate.matmul.heuristics[i].algo.data))
  if row['error']<5e-4:row['time_us']=paired(dict(candidate=candidate))['candidate']['median_us']
  record['algorithms'].append(row);print('ALGORITHM',row,flush=True);path.write_text(json.dumps(record,indent=2))
 valid=[r for r in record['algorithms'] if 'time_us' in r]
 if valid:
  winner=min(valid,key=lambda r:r['time_us']);candidate.matmul.index=winner['index'];record['selected']=winner
  def old():plan.b1.exact_dwp=native;return plan()
  def new():plan.b1.exact_dwp=candidate;return plan()
  yn,gn=new();record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
  record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
  if record['strict']:
   record['times']=paired(dict(old_dwp=native,new_dwp=candidate,old_full=old,new_full=new))
   print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
