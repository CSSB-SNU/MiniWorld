from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_hybrid_b1 import WideB1
from wide_b7 import WideB7
from validate_engine import capture
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False)
path=THIS/f'result-wide-b7-D{D}-L{N}-{record["job"]}.json'
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn),packed=f.w)
 b1=WideB1(p,leaves);candidate=WideB7(p,f.mask)
 def contractions():
  ab=p.front.ab;h=2*D
  torch.bmm(p.dt[:D],ab[h:h+D],out=p.dl[:D]);torch.bmm(p.dt[:D].transpose(-1,-2),ab[:D],out=p.dr[:D])
  torch.bmm(ab[h+D:],p.dt[D:].transpose(-1,-2),out=p.dl[D:]);torch.bmm(ab[D:h],p.dt[D:],out=p.dr[D:])
 def old7():W.launch(p.ks['b7'],p.params7,p.grid7,D=D,gp=p.gp_native);return p.outputs
 def old():b1();contractions();return old7()
 def new():b1();contractions();return candidate()
 def oldfull():return f(),old()
 def newfull():return f(),new()
 expected=[t.clone() for t in old()];gp=p.gp_all.clone();dxn=p.tensors[10].clone()
 p.gp_all.fill_(float('nan'));p.floats[7].fill_(float('nan'));p.tensors[10].fill_(float('nan'))
 actual=[t.clone() for t in new()];torch.cuda.synchronize()
 record['errors']={n:error(a,b) for n,a,b in zip(names[1:],actual,expected)}
 record['gp_error']=error(p.gp_all,gp);record['dxn_error']=error(p.tensors[10],dxn)
 record['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 record['cubins']=[str(candidate.source_cubin),str(candidate.finish_cubin)]
 print('CHECK',record,flush=True);path.write_text(json.dumps(record,indent=2))
 if record['strict']:
  graph,out=capture(newfull);graph.replay();torch.cuda.synchronize()
  record['graph_errors']=[error(a,b) for a,b in zip(out[1],actual)];assert max(record['graph_errors'])<5e-6
  record['times']=paired(dict(old_b7=old7,new_b7=candidate,source=candidate.source_only,finish=candidate.finish_only,old_bwd=old,new_bwd=new,old_full=oldfull,new_full=newfull))
  print('TIMES',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;path.write_text(json.dumps(record,indent=2))
