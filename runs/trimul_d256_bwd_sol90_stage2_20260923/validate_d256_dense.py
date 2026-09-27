from pathlib import Path
import os,sys,json,hashlib
THIS=Path(__file__).resolve().parent;PRE=THIS.parent/'trimul_d256_bwd_sol90_20260923';sys.path.insert(0,str(PRE))
exec((PRE/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0])
sys.path.insert(0,str(THIS));from selected_current import Previous
from d256_dense_checkpoint import Training,configuration,configure
configure()
from validate_engine import capture
N=(384,768)[int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))];leaves,dy,mask,ds,ref,triton,names=setup(256,N);records=[]
strict=lambda es:all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in es.items())
with torch.no_grad(),T.native_context(leaves[0].device):
 old=Previous(leaves,mask,ds,dy);new=Training(leaves,mask,ds,dy)
 from shared_candidate import attach
 candidate=new.metadata
 tag=f'd256-dense-{os.environ.get("SLURM_JOB_ID","local")}'
 path=THIS/f'validation-d256-dense-D256-L{N}-{os.environ.get("SLURM_JOB_ID")}.json'
 record=dict(L=N,job=os.environ.get('SLURM_JOB_ID'),candidate=candidate,stress=records,complete=False,config={k:v for k,v in os.environ.items() if k.startswith(('GP_','DX_','LN_','CUTE_','SAVE_','DN_','DNS_','SHARED_'))})
 record['artifacts']=[dict(path=str(a),sha256=hashlib.sha256(Path(a).read_bytes()).hexdigest()) for a in new.artifacts]
 record['configuration']=configuration(new)
 if os.environ.get('SANITIZE')!='1':path.write_text(json.dumps(record,indent=2))
 new();torch.cuda.synchronize()
 if os.environ.get('SANITIZE')=='1':
  print('SANITIZER_DONE',N,flush=True);raise SystemExit
 graph,_=capture(new)
 for case in ('normal','changed','zero_gamma','mask_zero','dropout_zero'):
  if case=='changed':
   leaves[0].mul_(.7);dy.normal_();leaves[1].mul_(1.1);leaves[2].mul_(.9);leaves[5].mul_(1.05);leaves[6].mul_(.9)
   mask.copy_((torch.rand_like(mask.float())>.3).bfloat16());ds.copy_((torch.rand_like(ds.float())>.4).bfloat16()*(1/.6))
  if case=='zero_gamma':leaves[7][::3]=0;leaves[9][::3]=0
  if case=='mask_zero':mask.zero_()
  if case=='dropout_zero':mask.fill_(1);ds.zero_()
  old.f.mask.copy_(mask.reshape_as(old.f.mask));old.b7.mask.copy_(mask.reshape_as(old.b7.mask))
  yo,go=old();expected=[x.clone() for x in go];yn,gn=new();actual=[x.clone() for x in gn]
  errors={n:error(x,y) for n,x,y in zip(names[1:],actual,expected)}
  assert strict(errors),(case,errors)
  assert error(yn,yo)==0
  graph.replay();torch.cuda.synchronize();ge={n:error(x,y) for n,x,y in zip(names[1:],new.p.outputs,actual)}
  assert max(ge.values())<5e-6,(case,ge)
  records.append(dict(case=case,strict=errors,graph=ge,passed=True));print(case,max(errors.values()),max(ge.values()),flush=True)
  path.write_text(json.dumps(record,indent=2))
  if case=='normal':normal=[x.clone() for x in actual]
# A fresh fixture reproduces the standard case after the stress mutations.
leaves,dy,mask,ds,ref,triton,names=setup(256,N)
y=torch.compile(ref)(*leaves,mask,ds);grads=torch.autograd.grad(y,leaves,dy)
pe={n:error(x,y) for n,x,y in zip(names[1:],normal,grads)};assert max(pe.values())<.01,pe
print('PYTORCH',pe,flush=True)
record.update(pytorch=pe,complete=True);path.write_text(json.dumps(record,indent=2))
