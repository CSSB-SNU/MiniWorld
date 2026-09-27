from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_hybrid_b1 import WideB1
from wide_b7 import WideB7
from wide_saved_norm import SavedOutput
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-saved-norm-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn),packed=f.w)
 b1=WideB1(p,leaves);b7=WideB7(p,f.mask);oldout=f.output;newout=SavedOutput(f,p)
 def backward():
  b1();ab=p.front.ab;h=2*D
  torch.bmm(p.dt[:D],ab[h:h+D],out=p.dl[:D]);torch.bmm(p.dt[:D].transpose(-1,-2),ab[:D],out=p.dr[:D])
  torch.bmm(ab[h+D:],p.dt[D:].transpose(-1,-2),out=p.dl[D:]);torch.bmm(ab[D:h],p.dt[D:],out=p.dr[D:])
  return b7()
 def run(saved):
  f.output=newout if saved else oldout;b1.use_saved_norm=saved
  return f(),backward()
 def old():return run(False)
 def new():return run(True)
 y,g=old();yold=y.clone();expected=[t.clone() for t in g];norm=p.tensors[6].clone();mu=p.floats[5].clone();rs=p.floats[6].clone()
 p.tensors[6].fill_(float('nan'));p.floats[5].fill_(float('nan'));p.floats[6].fill_(float('nan'))
 y,g=new();torch.cuda.synchronize()
 record['errors']={n:error(a,b) for n,a,b in zip(names[1:],g,expected)}
 record['saves']={n:error(a,b) for n,a,b in [('norm',p.tensors[6],norm),('mu',p.floats[5],mu),('rs',p.floats[6],rs)]}
 record['forward_error']=error(y,yold)
 record['strict']=record['forward_error']==0 and all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 record['cubin']=str(newout.cubin)
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  record['times']=paired(dict(old_full=old,new_full=new));print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
