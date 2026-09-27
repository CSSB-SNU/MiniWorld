"""Bounded projection proof. This is not a fused-attention performance test."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,sys
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'oc'))
p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=768)
p.add_argument('--output',default=str(HERE/'check.json'))
p.add_argument('--cases',default='random,large,constant,tiny,outlier')
p.add_argument('--variant',default='proof')
a=p.parse_args();L=a.length
torch.manual_seed(91022)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
module_name='triattn_projected_'+a.variant
spec=importlib.util.spec_from_file_location(module_name,HERE/(module_name+'.so'))
ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
from opt_core.kernels.triattn_surround_tma import _load
baseline=_load()
w=(torch.randn(512,128,device='cuda')/math.sqrt(128)).bfloat16()
# Immutable model-weight cache. This reorder is performed once, like the
# established packed Q/K/V/G weights, and does not depend on attention inputs.
wkv=torch.cat((w[128:256].reshape(4,32,128),w[256:384].reshape(4,32,128)),dim=1).contiguous()
wb=torch.zeros(16,128,device='cuda',dtype=torch.bfloat16)
wb[:4]=(torch.randn(4,128,device='cuda')/math.sqrt(128)).bfloat16()
lnw=(1+torch.randn(128,device='cuda')*.05).bfloat16()
lnb=(torch.randn(128,device='cuda')*.05).bfloat16()
q,k,v=[torch.empty(L,4,L,32,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
qr,kr,vr=[torch.empty_like(q) for _ in range(3)]
norm=torch.empty(L,L,128,device='cuda',dtype=torch.bfloat16)
g=torch.empty_like(norm);gr=torch.empty_like(g)
bias=torch.empty(4,L,L,device='cuda',dtype=torch.float32);br=torch.empty_like(bias)
records=[]
with torch.no_grad():
 for case in a.cases.split(','):
  x=torch.randn_like(norm)
  if case=='large':x.mul_(1024)
  elif case=='constant':x.fill_(2.5)
  elif case=='tiny':x.mul_(2**-12)
  elif case=='outlier':x[:,::17,:8]=512
  elif case!='random':raise ValueError(case)
  for ending in (False,True):
   for t in (norm,g,bias,q,k,v):t.fill_(float('nan'))
   baseline.prologue(x,w,wb,lnw,lnb,qr,kr,vr,gr,br,1e-5,ending)
   ext.prepare(x,w,wb,lnw,lnb,norm,norm,norm,g,bias,1e-5,ending)
   ext.project(norm,w,wkv,q,k,v)
   torch.cuda.synchronize()
   row={'case':case,'ending':ending,'length':L}
   for name,cand,ref in [('q',q,qr),('k',k,kr),('v',v,vr),('g',g,gr),('bias',bias,br)]:
    eq=torch.equal(cand,ref)
    row[name]={'bitwise_equal':eq,'max_error':float((cand.float()-ref.float()).abs().max()),
               'different':int((cand!=ref).sum())}
   records.append(row);print('CHECK',json.dumps(row),flush=True)
   Path(a.output).write_text(json.dumps({'records':records},indent=2))
   assert all(row[n]['bitwise_equal'] for n in ('q','k','v','g','bias')),row
result={'passed':len(records),'length':L,'records':records,
        'sha256':hashlib.sha256((HERE/(module_name+'.so')).read_bytes()).hexdigest(),
        'scope':'standalone projection proof; no fused attention or speedup claim'}
Path(a.output).write_text(json.dumps(result,indent=2))
print('PASS',len(records),'projection cases',flush=True)
