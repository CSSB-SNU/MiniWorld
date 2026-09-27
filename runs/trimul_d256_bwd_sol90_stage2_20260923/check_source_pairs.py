from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from source_pairs import PairB7
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-source-pairs-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);plan();old=plan.b7;expected=[x.clone() for x in plan.p.outputs];gp=[x.clone() for x in plan.p.gp]
 new=PairB7(plan.p,old);new();torch.cuda.synchronize()
 es={n:error(x,y) for n,x,y in zip(names[1:],plan.p.outputs,expected)}
 good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
 record.update(errors=es,strict=good,gp=[error(x,y) for x,y in zip(plan.p.gp,gp)],cubin=str(new.cubin));print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2));assert good
 def make(b7,full):
  def run():
   prior=plan.b7;plan.b7=b7
   try:return plan() if full else plan.backward()
   finally:plan.b7=prior
  return run
 if os.environ.get('SANITIZE')!='1':
  record['times']=paired(dict(old_b7=old,new_b7=new,old_bwd=make(old,False),new_bwd=make(new,False),old_full=make(old,True),new_full=make(new,True)))
  print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
