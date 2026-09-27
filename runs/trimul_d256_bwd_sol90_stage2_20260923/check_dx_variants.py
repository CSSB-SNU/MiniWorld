from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from dx_ln import DxLN
from dx_ln_rows import DxLNRows
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates={})
path=THIS/f'result-dx-variants-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 os.environ['DX_N256']='0';plan=Training(leaves,mask,ds,dy);plan();old=plan.b7.wide_finish;expected=[x.clone() for x in plan.p.outputs]
 def make(dx,full):
  def run():
   prior=plan.b7.wide_finish;plan.b7.wide_finish=dx
   try:return plan() if full else plan.backward()
   finally:plan.b7.wide_finish=prior
  return run
 fns=dict(base_dx=old,base_bwd=make(old,False),base_full=make(old,True));keep=[]
 for name,cls,wide in [('n256',DxLN,1),('rows128',DxLNRows,0),('rows256',DxLNRows,1)]:
  os.environ['DX_N256']=str(wide);new=cls(plan.p,plan.b7.splits);keep.append(new);new();torch.cuda.synchronize()
  es={n:error(x,y) for n,x,y in zip(names[1:],plan.p.outputs,expected)}
  good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'][name]=dict(errors=es,strict=good,cubin=str(new.cubin));print(name,record['candidates'][name],flush=True);path.write_text(json.dumps(record,indent=2))
  if not good:continue
  fns[name+'_dx']=new;fns[name+'_bwd']=make(new,False);fns[name+'_full']=make(new,True)
 record['times']=paired(fns);print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['strict_all']=all(c['strict'] for c in record['candidates'].values())
record['complete']=True;path.write_text(json.dumps(record,indent=2))
