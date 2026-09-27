from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_checkpoint14 import Training
from wide_plain_cached_ln import PlainCachedLN
from wide_plain_affine_output_ln import PlainAffineOutputLN
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma')
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','1'));D=(384,512)[index//2];N=768
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-post-dn-overlap-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan();expected=[x.clone() for x in [y,*g]];dt=p.dt.clone()
 side=torch.cuda.Stream(device=p.x.device)
 op=PlainAffineOutputLN(p) if D==512 else PlainCachedLN(p,32,True,False,2)
 fullgrid=op.grid
 def weights():
  if plan.split_dwp is not None:plan.split_dwp()
  else:plan.schedule.run('dwp')
  plan.schedule.run('dwg')
 for overlap,scale in ((False,1),(True,1),(True,.5)):
  op.grid=int(fullgrid*scale)
  def new():
   y=plan.forward();s=plan.schedule;b=plan.b1;main=torch.cuda.current_stream()
   if D==512:plan.delta_backward.launch();plan.delta_projection()
   b.epi.launch((1056,1,1),(256,1,1),[b.ep],0);s.run('dn')
   if overlap:
    side.wait_stream(main)
    with torch.cuda.stream(side):op()
    weights();main.wait_stream(side)
   else:weights();op()
   for name in ('bc0','bc1','bc2','bc3'):s.run(name)
   plan.b7.source_only();plan.dx.copy_prefix();plan.dx.pack_weights();s.run('dx');plan.dx.reduce_only()
   return y,p.outputs
  yn,gn=new();torch.cuda.synchronize()
  v=dict(overlap=overlap,grid_scale=scale,grid=op.grid,cubin=str(op.cubin),dt_error=error(p.dt,dt))
  v['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],expected)}
  v['strict']=all(e<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,e in v['errors'].items())
  print('CHECK',v,flush=True)
  if v['strict']:
   wanted=[a.clone() for a in [yn,*gn]];graph,out=capture(new);graph.replay();torch.cuda.synchronize()
   v['graph_errors']={n:error(a,b) for n,a,b in zip(names,[out[0],*out[1]],wanted)}
   assert max(v['graph_errors'].values())<5e-6,v
   del graph,out
   v['times']=paired(dict(old_full=plan,new_full=new))
   print('TIMES',{k:t['median_us'] for k,t in v['times'].items()},flush=True)
  record['candidates'].append(v);path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
