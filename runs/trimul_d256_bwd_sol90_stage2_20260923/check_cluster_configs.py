"""Same-buffer comparison of complete DSM B7 configurations."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from cluster_b7 import ClusterB7
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates={})
path=THIS/f'result-cluster-configs-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);plan();old=plan.b7;expected=[x.clone() for x in plan.p.outputs]
 def make(b7,full):
  def run():
   prior=plan.b7;plan.b7=b7
   try:return plan() if full else plan.backward()
   finally:plan.b7=prior
  return run
 fns=dict(base_b7=old,base_bwd=make(old,False),base_full=make(old,True));keep=[]
 bulk=int(os.environ.get('CLUSTER_BULK','0'))
 cases=[(16,64,bulk),(16,80,bulk),(16,96,bulk),(12,80,bulk),(14,80,bulk)]
 if os.environ.get('CLUSTER_PREFETCH')=='1':cases=[(14,80,bulk),(14,96,bulk),(14,112,bulk)]
 if os.environ.get('CLUSTER_SWEEP')=='bulk':cases=[(15,80,0),(15,96,0),(15,80,1),(15,96,1),(14,96,1),(12,96,1)]
 if os.environ.get('CLUSTER_PLANE')=='1':cases=[(15,80,0),(15,96,0),(14,80,0),(12,80,0)]
 for splits,regs,bulk in cases:
  name=f's{splits}r{regs}b{bulk}';os.environ.update(CLUSTER_SPLITS=str(splits),CLUSTER_DX_REGS=str(regs),CLUSTER_DUMP='0',CLUSTER_BULK=str(bulk))
  new=ClusterB7(plan.p,plan.f.mask);keep.append(new);out=new();torch.cuda.synchronize()
  es={n:error(x,y) for n,x,y in zip(names[1:],out,expected)}
  good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'][name]=dict(errors=es,strict=good,cubin=str(new.cubin));print(name,record['candidates'][name],flush=True);path.write_text(json.dumps(record,indent=2));assert good
  fns[name+'_b7']=new;fns[name+'_bwd']=make(new,False);fns[name+'_full']=make(new,True)
 record['times']=paired(fns);print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record.update(complete=True,config={k:v for k,v in os.environ.items() if k.startswith(('CLUSTER_','GP_','DX_','LN_','SAVE_'))});path.write_text(json.dumps(record,indent=2))
