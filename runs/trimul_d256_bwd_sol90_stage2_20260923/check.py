from pathlib import Path
import sys,os,json
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS));from b1 import FastB1
N=int(os.environ.get('LENGTH','384'))
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,complete=False)
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn))
 b1=B1(p);new1=FastB1(p,b1);b7=B7(p,8 if N==384 else 16)
 if os.environ.get('B1_MODE') in ('hybrid','cute'):
  from hybrid_b1 import HybridB1
  new1=HybridB1(p,leaves)
 new7=b7
 if os.environ.get('DX_MODE')=='chunk':
  from chunk_b7 import ChunkB7
  new7=ChunkB7(p,leaves)
 if os.environ.get('DX_MODE')=='ring3':
  from ring3 import Ring3
  new7=Ring3(p,leaves)
 if os.environ.get('DX_MODE')=='ring2':
  from ring2 import Ring2
  new7=Ring2(p,leaves)
 if os.environ.get('DX_MODE')=='tma':
  from tma_b7 import TmaB7
  new7=TmaB7(p,leaves)
 if os.environ.get('DX_MODE')=='cute':
  from cute_dx import CuteDx
  new7=CuteDx(p,leaves)
 if os.environ.get('DX_MODE')=='n256':
  from n256_b7 import N256B7
  new7=N256B7(p,leaves)
 if os.environ.get('DX_MODE')=='pipe':
  from pipe_b7 import PipeB7
  new7=PipeB7(p,leaves)
 if os.environ.get('DX_MODE')=='allblas':
  from blas_b7 import BlasB7
  new7=BlasB7(p,leaves)
 if os.environ.get('DX_MODE') in ('blas','blasrm'):
  from blas_dx import BlasDxB7
  new7=BlasDxB7(p,leaves,row_major=os.environ.get('DX_MODE')=='blasrm')
 if os.environ.get('DX_LN')=='1':
  from dx_ln import DxLN
  new7.wide_finish=DxLN(p,new7.splits)
 def contract():
  ab=p.front.ab;d=256;h=512
  torch.bmm(p.dt[:d],ab[h:h+d],out=p.dl[:d]);torch.bmm(p.dt[:d].transpose(-1,-2),ab[:d],out=p.dr[:d]);torch.bmm(ab[h+d:],p.dt[d:].transpose(-1,-2),out=p.dl[d:]);torch.bmm(ab[d:h],p.dt[d:],out=p.dr[d:])
 def old():b1();contract();return b7()
 def new():new1();contract();return new7()
 def oldfull():f();return old()
 def newfull():f();return new()
 expected=[x.clone() for x in old()];result=[x.clone() for x in new()]
 record['errors']={n:error(x,y) for n,x,y in zip(names[1:],result,expected)};print('ERROR',record['errors'],flush=True)
 record['strict']=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in record['errors'].items())
 print('STRICT',record['strict'],flush=True)
 record['times']=paired(dict(old_b1=b1,new_b1=new1,old_bwd=old,new_bwd=new,old_full=oldfull,new_full=newfull))
 print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True;record['env']={k:v for k,v in os.environ.items() if k.startswith(('CUTE_','GP_','RCOHORT','RCONSUM','RSLOTS','DX_','B1_'))};(THIS/f'result-L{N}-{os.environ.get("B1_MODE","native")}-{os.environ.get("DX_MODE","native")}-{os.environ.get("SLURM_JOB_ID","local")}.json').write_text(json.dumps(record,indent=2))
