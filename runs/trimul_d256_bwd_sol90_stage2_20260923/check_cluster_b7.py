from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from cluster_b7 import ClusterB7
N=int(os.environ.get('LENGTH','384'));leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);new=ClusterB7(plan.p,plan.f.mask)
 if os.environ.get('CLUSTER_COMPILE_ONLY')=='1':print('COMPILE_ONLY_DONE',flush=True);raise SystemExit
 plan();p=plan.p;old=plan.b7;expected=[x.clone() for x in p.outputs]
 if new.dump:gp=[x.clone() for x in p.gp]
 out=new();torch.cuda.synchronize()
 if os.environ.get('CLUSTER_CLOCK')=='1':
  import statistics
  record['clock_right']=new.clock.cpu().tolist()
  for role,points in ((0,[0,1,6,7]),(1,list(range(7))),(2,list(range(7))),(3,list(range(6)))):
   selected=[row[role] for row in record['clock_right'] if all(row[role][i] for i in points)]
   print('CLOCK_RIGHT',role,{f'{a}-{b}':statistics.median(x[b]-x[a] for x in selected) for a,b in zip(points,points[1:])},flush=True)
  if new.plane:
   selected=[row[3] for row in record['clock_right'] if row[3][8]]
   print('CLOCK_PLANE_GATE',statistics.median(x[101]-x[100] for x in selected),flush=True)
   for step in range(16):
    a=8+step*4
    print('CLOCK_STEP',step,{name:statistics.median(x[a+e]-x[a+s] for x in selected) for name,s,e in [('wait',0,1),('mma',1,2),('release',2,3)]},flush=True)
 record['errors']={n:error(x,y) for n,x,y in zip(names[1:],out,expected)}
 if new.dump:record['gp']=[error(x,y) for x,y in zip(p.gp,gp)];print('GP',record['gp'],flush=True)
 record['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 print('ERROR',record['errors'],'STRICT',record['strict'],flush=True)
 path=THIS/f'result-cluster-b7-L{N}-{record["job"]}.json';path.write_text(json.dumps(record,indent=2));assert record['strict']
 def make(b7,full):
  def run():
   prior=plan.b7;plan.b7=b7
   try:return plan() if full else plan.backward()
   finally:plan.b7=prior
  return run
 if os.environ.get('SANITIZE')!='1':
  record['times']=paired(dict(old_b7=old,new_b7=new,old_bwd=make(old,False),new_bwd=make(new,False),old_full=make(old,True),new_full=make(new,True)))
  print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record.update(complete=True,cubin=str(new.cubin),config={k:v for k,v in os.environ.items() if k.startswith(('CLUSTER_','GP_','DX_','LN_','SAVE_'))})
path.write_text(json.dumps(record,indent=2))
