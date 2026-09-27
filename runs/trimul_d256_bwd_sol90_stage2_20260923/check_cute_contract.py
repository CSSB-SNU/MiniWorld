from pathlib import Path
import sys,os,json,traceback
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from shared_candidate import attach
from cute_contract import Contractions
N=int(('384','768')[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))])
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-cute-contract-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);record['control']=attach(plan)
 plan();expected=[x.clone() for x in plan.p.outputs];deriv=[p.clone() for p in (plan.p.dl,plan.p.dr)]
 def old():
  p=plan.p;ab=p.front.ab;d=256;h=512
  torch.bmm(p.dt[:d],ab[h:h+d],out=p.dl[:d]);torch.bmm(p.dt[:d].transpose(-1,-2),ab[:d],out=p.dr[:d])
  torch.bmm(ab[h+d:],p.dt[d:].transpose(-1,-2),out=p.dl[d:]);torch.bmm(ab[d:h],p.dt[d:],out=p.dr[d:])
 for tm,tn,ping,cm,cn in [(128,128,True,1,1),(128,128,False,1,1),(128,256,True,1,1),(128,256,False,1,1),(64,128,False,1,1),(64,256,False,1,1),(128,128,True,2,1),(128,128,True,1,2)]:
  result=dict(config=[tm,tn,ping,cm,cn]);record['candidates'].append(result)
  try:
   candidate=Contractions(plan,tm,tn,ping,cm,cn)
   plan.p.dl.fill_(float('nan'));plan.p.dr.fill_(float('nan'));candidate();torch.cuda.synchronize()
   result['contraction_errors']=[error(a,b) for a,b in zip((plan.p.dl,plan.p.dr),deriv)]
   plan.b7();result['errors']={n:error(x,y) for n,x,y in zip(names[1:],plan.p.outputs,expected)}
   result['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in result['errors'].items())
   result['finite']=bool(plan.p.dl.isfinite().all() and plan.p.dr.isfinite().all())
   if result['strict'] and result['finite']:
    def newbwd():plan.b1();candidate();return plan.b7()
    def newfull():return plan.forward(),newbwd()
    result['times']=paired(dict(old_contract=old,new_contract=candidate,old_bwd=plan.backward,new_bwd=newbwd,old_full=plan,new_full=newfull))
   print('CANDIDATE',result['config'],result.get('strict'),result.get('errors'),{k:v['median_us'] for k,v in result.get('times',{}).items()},flush=True)
  except Exception as e:
   result['exception']=repr(e);traceback.print_exc()
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
