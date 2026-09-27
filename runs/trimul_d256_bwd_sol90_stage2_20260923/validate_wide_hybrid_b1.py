from pathlib import Path
import sys,os,json,hashlib
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923'
sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS))
from wide_hybrid_b1 import WideB1
from validate_engine import capture
index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'));D=(384,512)[index//2];N=(384,768)[index%2]
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),complete=False,stress=[])
path=THIS/f'validation-wide-hybrid-D{D}-L{N}-{record["job"]}.json'
def strict(es):return all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
with torch.no_grad(),T.native_context(leaves[0].device):
 f=F.Forward(leaves,mask,ds);f();p=W.Training(*leaves,mask,ds,dy,saved=(f.front.ab,f.tri,f.front.xn),packed=f.w)
 if D==384:old_b1=B1(p)
 else:
  def old_b1():W.launch(p.ks['b1'],p.params,p.grid,D=D)
 candidate=WideB1(p,leaves)
 artifacts=[candidate.norm_cubin,candidate.ln_cubin]
 if candidate.exact_dwp is not None:artifacts.append(candidate.exact_dwp.cubin)
 record['artifacts']=[dict(path=str(a),sha256=hashlib.sha256(Path(a).read_bytes()).hexdigest()) for a in artifacts]
 record['source_sha256']=hashlib.sha256((THIS/'wide_hybrid_b1.py').read_bytes()).hexdigest()
 print('ARTIFACTS',record['artifacts'],flush=True)
 def tail():
  ab=p.front.ab;h=2*D
  torch.bmm(p.dt[:D],ab[h:h+D],out=p.dl[:D]);torch.bmm(p.dt[:D].transpose(-1,-2),ab[:D],out=p.dr[:D])
  torch.bmm(ab[h+D:],p.dt[D:].transpose(-1,-2),out=p.dl[D:]);torch.bmm(ab[D:h],p.dt[D:],out=p.dr[D:])
  W.launch(p.ks['b7'],p.params7,p.grid7,D=D,gp=p.gp_native)
  return p.outputs
 def forward():
  f.mask.copy_(mask);p.mask.copy_(mask.reshape_as(p.mask));return f()
 def full(b1):y=forward();b1();return y,tail()
 def new():return full(candidate)
 new();torch.cuda.synchronize()
 if os.environ.get('SANITIZE')=='1':
  print('SANITIZER_DONE',D,N,flush=True);raise SystemExit
 graph,_=capture(new)
 for case in ('normal','changed','zero_gamma','mask_zero','dropout_zero'):
  if case=='changed':
   leaves[0].mul_(.7);dy.normal_();leaves[1].mul_(1.1);leaves[2].mul_(.9);leaves[5].mul_(1.05);leaves[6].mul_(.9)
   mask.copy_((torch.rand_like(mask.float())>.3).bfloat16());ds.copy_((torch.rand_like(ds.float())>.4).bfloat16()*(1/.6))
  if case=='zero_gamma':leaves[7][::3]=0;leaves[9][::3]=0
  if case=='mask_zero':mask.zero_()
  if case=='dropout_zero':mask.fill_(1);ds.zero_()
  yo,go=full(old_b1);yo=yo.clone();expected=[t.clone() for t in go]
  yn,gn=new();actual=[t.clone() for t in gn]
  es={n:error(a,b) for n,a,b in zip(names[1:],actual,expected)}
  row=dict(case=case,strict=es,passed=strict(es),forward_error=error(yn,yo));record['stress'].append(row)
  path.write_text(json.dumps(record,indent=2));print('STRESS',row,flush=True)
  assert row['passed'] and row['forward_error']==0,row
  graph.replay();torch.cuda.synchronize();ge={n:error(a,b) for n,a,b in zip(names[1:],p.outputs,actual)}
  row['graph']=ge;assert max(ge.values())<5e-6,(case,ge)
  if case=='normal':normal=[t.clone() for t in actual]
  path.write_text(json.dumps(record,indent=2))
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
y=torch.compile(ref)(*leaves,mask,ds);grads=torch.autograd.grad(y,leaves,dy)
record['pytorch']={n:error(a,b) for n,a,b in zip(names[1:],normal,grads)}
assert max(record['pytorch'].values())<.01,record['pytorch']
record['complete']=True;path.write_text(json.dumps(record,indent=2));print('VALIDATED',D,N,flush=True)
