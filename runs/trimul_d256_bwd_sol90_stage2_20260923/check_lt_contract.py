from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from shared_candidate import attach
from lt_contract import LtBmm
N=int(('384','768')[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))])
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,algorithms=[])
path=THIS/f'result-lt-contract-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);record['control']=attach(plan);plan()
 expected=[x.clone() for x in plan.p.outputs]
 p=plan.p;ab=p.front.ab;d=256;h=512
 args=[(p.dt[:d],ab[h:h+d],p.dl[:d]),(p.dt[:d].transpose(-1,-2),ab[:d],p.dr[:d]),
       (ab[h+d:],p.dt[d:].transpose(-1,-2),p.dl[d:]),(ab[d:h],p.dt[d:],p.dr[d:])]
 workspace=torch.empty(64*1024*1024,device=leaves[0].device,dtype=torch.uint8)
 choices=[]
 for op,operands in enumerate(args):
  kernel=LtBmm(*operands,workspace);wanted=operands[-1].clone();rows=[]
  for i in kernel.indices:
   kernel.index=i;operands[-1].fill_(float('nan'));kernel();torch.cuda.synchronize()
   err=error(operands[-1],wanted);row=dict(index=i,error=err,algo=list(kernel.heuristics[i].algo.data),workspace=kernel.heuristics[i].workspaceSize)
   if err==0:
    row['time_us']=paired(dict(candidate=kernel))['candidate']['median_us']
   rows.append(row)
  valid=[r for r in rows if 'time_us' in r];assert valid,(op,rows)
  winner=min(valid,key=lambda r:r['time_us']);kernel.index=winner['index'];choices.append(kernel)
  record['algorithms'].append(dict(op=op,candidates=rows,selected=winner))
  path.write_text(json.dumps(record,indent=2));print('SELECT',op,len(rows),winner,flush=True)
 def candidate():
  for k in choices:k()
 def old():
  for a,b,c in args:torch.bmm(a,b,out=c)
 def newbwd():plan.b1();candidate();return plan.b7()
 def newfull():return plan.forward(),newbwd()
 newfull();es={n:error(x,y) for n,x,y in zip(names[1:],p.outputs,expected)}
 record['errors']=es;record['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
 assert record['strict'],es
 record['times']=paired(dict(old_contract=old,new_contract=candidate,old_bwd=plan.backward,new_bwd=newbwd,old_full=plan,new_full=newfull))
 print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
