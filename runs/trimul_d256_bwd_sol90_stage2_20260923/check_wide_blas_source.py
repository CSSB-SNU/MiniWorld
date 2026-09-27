from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_combined import WideTraining
from wide_blas_source import BlasWideSource
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-blas-source-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=WideTraining(leaves,mask,ds,dy);p=plan.p;source=BlasWideSource(p,plan.f.mask)
 def newbwd():
  plan.b1();plan.contract();source.derivatives();plan.dx();source.weights();return p.outputs
 def newfull():return plan.forward(),newbwd()
 y,g=plan();y=y.clone();expected=[t.clone() for t in g];gp=p.gp_all.clone()
 p.gp_all.fill_(float('nan'));p.floats[7].fill_(float('nan'));p.tensors[10].fill_(float('nan'))
 yn,gn=newfull();torch.cuda.synchronize()
 record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[y,*expected])}
 record['derivative_error']=error(p.gp_all,gp)
 record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 record['cubin']=str(source.cubin)
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_bwd=plan.backward,new_bwd=newbwd,old_full=plan,new_full=newfull,old_source=plan.b7.source_only,derivatives=source.derivatives,weights=source.weights))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
