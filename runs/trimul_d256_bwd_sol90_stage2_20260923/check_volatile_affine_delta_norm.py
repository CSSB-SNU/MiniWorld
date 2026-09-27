from pathlib import Path
import sys,os,json,subprocess
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint13 import Training
from wide_volatile_affine_delta_norm import VolatileAffineDeltaNorm as TileDeltaNorm,PatchDeltaNorm
from wide_compact_projection import CompactProjection
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
D=512;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-volatile-affine-delta-norm-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[t.clone() for t in [y,*g]];norm=p.tensors[6].clone()
 original=(plan.product_output,plan.delta_output,plan.delta_backward,plan.delta_projection)
 for rows,mb in ((16,2),(16,4),(32,3)):
  threads=128
  op=TileDeltaNorm(p,original[0],rows=rows,threads=threads,minblocks=mb);back=PatchDeltaNorm(op);proj=CompactProjection(plan,op)
  print(subprocess.run(['cuobjdump','--dump-resource-usage',str(op.cubin)],capture_output=True,text=True).stdout,flush=True)
  def old():plan.product_output,plan.delta_output,plan.delta_backward,plan.delta_projection=original;return plan()
  def new():plan.product_output,plan.delta_output,plan.delta_backward,plan.delta_projection=op,op,back,proj;return plan()
  yn,gn=new();torch.cuda.synchronize()
  v=dict(minblocks=mb,rows=rows,threads=threads,grid=op.grid,cubin=str(op.cubin),patches=int(op.count.item()),norm_error=error(p.tensors[6],norm))
  v['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
  v['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in v['errors'].items())
  print('CHECK',v,flush=True)
  if v['strict']:
   prior=original[0]
   v['times']=paired(dict(old_norm=lambda:prior.norm.launch((prior.grid,1,1),(prior.threads,1,1),[prior.np],prior.smem),new_norm=lambda:op.norm.launch((op.grid,1,1),(op.threads,1,1),[op.np],op.smem),old_full=old,new_full=new))
   print('TIMES',{k:t['median_us'] for k,t in v['times'].items()},flush=True)
  record['candidates'].append(v);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
