from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from shared_candidate import attach
from saved_front_products import enable
from validate_engine import capture
N=int(('384','768')[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))])
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),complete=False)
path=THIS/f'result-saved-front-products-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);record['control']=attach(plan)
 y,_=plan();yold=y.clone();expected=[x.clone() for x in plan.p.outputs];gp=plan.p.gp_all.clone()
 oldfront,oldb7=plan.f.front,plan.b7
 record['candidate']=enable(plan);newfront,newb7=plan.f.front,plan.b7
 plan.saved[-1].fill_(float('nan'));plan.p.gp_all.fill_(float('nan'))
 y,_=plan();torch.cuda.synchronize()
 es={n:error(x,e) for n,x,e in zip(names[1:],plan.p.outputs,expected)}
 good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items()) and error(y,yold)==0
 finite=all(bool(t.isfinite().all()) for t in (*plan.p.outputs,plan.saved[-1],plan.p.gp_all))
 record.update(errors=es,forward_error=error(y,yold),strict=good and finite,gp_error=error(plan.p.gp_all,gp),finite=finite)
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if good and finite:
  eager=[t.clone() for t in plan.p.outputs];graph,out=capture(plan);graph.replay();torch.cuda.synchronize()
  record['graph_errors']=[error(t,e) for t,e in zip(out[1],eager)]
  assert max(record['graph_errors'])<5e-6,record['graph_errors']
  def make(front,b7,mode):
   def run():
    prior=plan.f.front,plan.b7;plan.f.front,plan.b7=front,b7
    try:return plan() if mode=='full' else plan.backward() if mode=='bwd' else plan.forward()
    finally:plan.f.front,plan.b7=prior
   return run
  def oldsource():oldb7.source.launch((32*oldb7.splits,1,1),(oldb7.source_threads,1,1),[oldb7.params],114816)
  fns=dict(old_source=oldsource,new_source=newb7.source_only)
  for mode in ('fwd','bwd','full'):
   fns['old_'+mode]=make(oldfront,oldb7,mode);fns['new_'+mode]=make(newfront,newb7,mode)
  record['times']=paired(fns);print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
