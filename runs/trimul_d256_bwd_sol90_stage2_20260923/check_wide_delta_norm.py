from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint9 import Training
from wide_delta_norm import DeltaNorm,PatchDeltaNorm
from wide_flagged_projection import FlaggedProjection
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
D=512;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-wide-delta-norm-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]];norm=p.tensors[6].clone();proj=plan.b1.proj.clone();stats=[p.floats[i].clone() for i in (5,6)]
 original=plan.product_output;oldnorm=plan.b1.norm;oldback=plan.backward
 changed=torch.ones(p.M//64,device=p.x.device,dtype=torch.int32)
 check_proj=FlaggedProjection(plan,changed);check_proj();torch.cuda.synchronize()
 record['dense_projection_error']=error(plan.b1.proj,proj)
 record['dense_projection_max']=(plan.b1.proj.float()-proj.float()).abs().max().item()
 print('PROJECTION',record,flush=True);path.write_text(json.dumps(record,indent=2))
 for cap in (0,None):
  op=DeltaNorm(p,original,capacity=cap);back=PatchDeltaNorm(op);project=FlaggedProjection(plan,op.changed)
  def sparse_backward():
   b=plan.b1;sch=plan.schedule
   back.launch();project()
   b.epi.launch((1056,1,1),(256,1,1),[b.ep],0);sch.run('dn')
   if plan.split_dwp is not None:plan.split_dwp()
   else:sch.run('dwp')
   sch.run('dwg');plan.ln()
   for name in ('bc0','bc1','bc2','bc3'):sch.run(name)
   plan.b7.source_only();plan.dx.copy_prefix();plan.dx.pack_weights();sch.run('dx');plan.dx.reduce_only()
   return p.outputs
  def old():plan.product_output=original;plan.b1.norm=oldnorm;plan.backward=oldback;return plan()
  def norm_only():plan.product_output=op;plan.b1.norm=back;plan.backward=oldback;return plan()
  def new():plan.product_output=op;plan.b1.norm=back;plan.backward=sparse_backward;return plan()
  for label,fn in (('norm_only',norm_only),('sparse_projection',new)):
   yn,gn=fn();torch.cuda.synchronize()
   v=dict(mode=label,capacity=op.capacity,patches=int(op.count.item()),changed_tiles=int(op.changed.sum().item()),total_tiles=op.changed.numel(),norm_error=error(p.tensors[6],norm),proj_error=error(plan.b1.proj,proj),stats_errors=[error(p.floats[i],x) for i,x in zip((5,6),stats)],cubin=str(op.cubin),fallback_cubin=str(op.fallback_cubin),projection_cubin=str(project.cubin))
   v['fallback']=v['patches']>v['capacity']
   if cap==0:assert v['fallback'],'Forced overflow was not exercised'
   v['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
   v['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in v['errors'].items())
   print('CHECK',v,flush=True)
   if v['strict'] and cap is None:
    v['times']=paired(dict(old_full=old,new_full=fn))
    print('TIMES',{k:t['median_us'] for k,t in v['times'].items()},flush=True)
   record['candidates'].append(v);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
