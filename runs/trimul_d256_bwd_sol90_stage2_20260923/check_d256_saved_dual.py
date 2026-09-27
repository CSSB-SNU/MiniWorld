from pathlib import Path
from types import SimpleNamespace
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from d256_mask_checkpoint import Training
from d256_saved_front_tma import D256SavedFrontTma
from wide_saved_dual import SavedDual
D=256;N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-d256-saved-dual-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);p=plan.p;y,g=plan()
 expected=[t.clone() for t in g];oldy=y.clone();gp=[t.clone() for t in p.gp]
 original_front=plan.f.front;original_b7=plan.b7
 pre=p.x.new_empty((D//8,p.M,64));front=D256SavedFrontTma(original_front,pre)
 L=T._launch_module();tm=lambda t:L.tensor_map(t,[64,32],dims=[p.M,2*D],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
 params=L.Struct([p.maps[0],W.tm(p.w1),tm(p.dl),tm(p.dr),*[tm(t) for t in p.gp],plan.f.mask,p.floats[7],p.M])
 source=SavedDual(p,SimpleNamespace(params=params),pre,original_b7.splits)
 def new_b7():source();original_b7.wide_finish();return p.outputs
 def old():plan.f.front=original_front;plan.b7=original_b7;return plan()
 def new():plan.f.front=front;plan.b7=new_b7;return plan()
 pre.fill_(float('nan'));p.input_stats.fill_(float('nan'))
 yn,gn=new();torch.cuda.synchronize()
 record['errors']={n:error(a,b) for n,a,b in zip(names,[yn,*gn],[oldy,*expected])}
 record['gp_error']=max(error(a,b) for a,b in zip(p.gp,gp))
 record['strict']=all(v<(1e-30 if n=='y' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 record['artifacts']=[str(front.cubin),str(source.cubin)]
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_front=original_front,new_front=front,old_source=original_b7.source_only,new_source=source,old_full=old,new_full=new))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
