from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_lt_checkpoint import Training
from d256_rowmajor_gp import RowMajorGP
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-rowmajor-gp-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();oldy=y.clone();expected=[t.clone() for t in g]
 dxn=p.tensors[10].clone();gp=p.gp_all.clone();op=RowMajorGP(plan);op.prepare();torch.cuda.synchronize()
 record['gp_error']=error(op.input[:,D:].reshape(p.M,4,2*D).permute(1,2,0),gp)
 record['prefix_error']=error(op.input[:,:D],p.dg)
 record['cubin']=str(op.cubin);print('CHECK_INPUT',record,flush=True)
 assert record['gp_error']==0 and record['prefix_error']==0,record
 choices=[];functions={};kernel=op.op
 for index in kernel.indices:
  kernel.index=index;kernel();torch.cuda.synchronize();err=error(kernel.out,dxn)
  choice=dict(index=index,error=err,algo=list(kernel.heuristics[index].algo.data));choices.append(choice)
  if err==0:
   def run(index=index):kernel.index=index;kernel()
   functions[f'lt{index}']=run
 record['lt']=choices;assert functions,record
 times=paired(functions)
 best=min((c for c in choices if c['error']==0),key=lambda c:times[f'lt{c["index"]}']['median_us'])
 record['selected']=best;kernel.index=best['index']
 def baseline_finish():plan.b7.source_only();plan.dx()
 def newfull():
  y=plan.forward();sch=plan.schedule
  plan.b1.prepare();sch.run('dn');sch.run('dwp');sch.run('dwg');plan.b1.ln()
  for name in ('bc0','bc1','bc2','bc3'):sch.run(name)
  op();return y,p.outputs
 yn,gn=newfull();record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
 record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(baseline_source=plan.b7.source_only,new_source=op.source_only,baseline_gemm=plan.dx.gemm_only,new_gemm=op.op,baseline_finish=baseline_finish,new_finish=op,baseline_full=plan,new_full=newfull))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
