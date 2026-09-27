"""Compare finish tile choices on one plan and the same fused-source buffers."""
from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_hybrid_b1 import WideB1
from wide_b7 import WideB7
from wide_dx import WideDx
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,*_=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates=[])
path=THIS/f'result-wide-dx-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn),packed=f.w)
 b1=WideB1(p,leaves);b7=WideB7(p,f.mask)
 def prefix():
  b1();ab=p.front.ab;h=2*D
  torch.bmm(p.dt[:D],ab[h:h+D],out=p.dl[:D]);torch.bmm(p.dt[:D].transpose(-1,-2),ab[:D],out=p.dr[:D])
  torch.bmm(ab[h+D:],p.dt[D:].transpose(-1,-2),out=p.dl[D:]);torch.bmm(ab[D:h],p.dt[D:],out=p.dr[D:])
  b7.source_only()
 prefix();b7.finish_only();expected=[t.clone() for t in p.outputs]
 names=['dx','dWL','dWLg','dWR','dWRg','dWgate','dWproj','dgamma_in','dbeta_in','dgamma_out','dbeta_out']
 for tile,groups in ((128,1),(128,2),(64,1)):
  finish=WideDx(p,tile,groups);p.tensors[10].fill_(float('nan'));finish();torch.cuda.synchronize()
  es={n:error(a,b) for n,a,b in zip(names,p.outputs,expected)}
  good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  row=dict(tile=tile,groups=groups,errors=es,strict=good,cubin=str(finish.cubin),grid=finish.grid,smem=finish.smem);record['candidates'].append(row)
  if good:
   def old():prefix();b7.finish_only();return p.outputs
   def new():prefix();finish();return p.outputs
   def oldfull():return f(),old()
   def newfull():return f(),new()
   row['times']=paired(dict(old_finish=b7.finish_only,new_finish=finish,old_bwd=old,new_bwd=new,old_full=oldfull,new_full=newfull))
  print('CANDIDATE',tile,groups,good,es,{k:v['median_us'] for k,v in row.get('times',{}).items()},flush=True)
  path.write_text(json.dumps(record,indent=2))
record['complete']=True;path.write_text(json.dumps(record,indent=2))
