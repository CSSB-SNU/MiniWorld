"""Use identical buffers and prepare/B7 kernels to isolate fused-stage ordering."""
from pathlib import Path
import sys,os,json,itertools
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from selected_current import Training
from dn_slim import DNSlim
N=int(os.environ.get('LENGTH','768'));leaves,dy,mask,ds,ref,triton,names=setup(256,N)
record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,candidates={})
with torch.no_grad(),T.native_context(leaves[0].device):
 os.environ['DN_SLIM']='0';plan=Training(leaves,mask,ds,dy);plan();p=plan.p;base=plan.b1
 expected=[x.clone() for x in p.outputs];dns=DNSlim(p)
 ops={'p':lambda:torch.mm(p.tensors[7].t(),p.tensors[6],out=p.dwp),
      'g':lambda:torch.mm(p.dg.t(),p.xn.reshape(p.M,256),out=p.dwg),'n':dns}
 def make_b1(order):
  def run():
   base.prepare()
   for op in order:ops[op]()
  return run
 def make(b1,full):
  def run():
   if full:plan.forward()
   prior=plan.b1;plan.b1=b1
   try:return plan.backward()
   finally:plan.b1=prior
  return run
 fns={'base_b1':base,'base_bwd':make(base,False),'base_full':make(base,True)}
 for order in itertools.permutations('pgn'):
  name=''.join(order);b1=make_b1(order);bwd=make(b1,False);out=bwd()
  es={n:error(x,y) for n,x,y in zip(names[1:],out,expected)}
  good=all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
  record['candidates'][name]=dict(errors=es,strict=good)
  print('CANDIDATE',name,record['candidates'][name],flush=True);assert good
  fns[name+'_b1']=b1;fns[name+'_bwd']=bwd;fns[name+'_full']=make(b1,True)
 record['times']=paired(fns);print({k:v['median_us'] for k,v in record['times'].items()},flush=True)
record['complete']=True
record['config']={k:v for k,v in os.environ.items() if k.startswith(('GP_','DX_','LN_','SAVE_','DNS_'))}
(THIS/f'result-dns-order-L{N}-{record["job"]}.json').write_text(json.dumps(record,indent=2))
