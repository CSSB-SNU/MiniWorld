from pathlib import Path
import sys,json,copy,statistics
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'oc'))
source=(R/'sweep_pro.py').read_text().split('with torch.no_grad():\n    y = call()')[0]
exec(compile(source,str(R/'sweep_pro.py'),'exec'))
from build import build
ext=build()
PRO=pf._carried('fpf_triatt_pro','prologue');original=PRO.triatt_prologue

def cuda_pro(module,z,ending=False,ln_mode='fused',x_ln=None,write_x=False,fma_flags=None,cfg=None):
 assert not ending and ln_mode=='fused' and not write_x
 c=PRO.get_cache(module,z.device);L=z.shape[0]
 q,k,v=[torch.empty((L,4,L,32),device=z.device,dtype=z.dtype) for _ in range(3)]
 g=torch.empty((L,L,128),device=z.device,dtype=z.dtype)
 bias=torch.empty((4,L,L),device=z.device,dtype=torch.float32)
 ext.prologue(z,c['wqkvg'],c['wb'],c['lnw'],c['lnb'],q,k,v,g,bias,c['eps'])
 return q,k,v,g,bias

results={}
with torch.no_grad():
 expected=call().clone()
 refpro=pf.prologue(x,W,impl='fpf')
 for label,fn in [('baseline',original),('cuda_tma',cuda_pro)]:
  PRO.triatt_prologue=fn
  y=call().clone();torch.cuda.synchronize()
  actual=pf.prologue(x,W,impl='fpf');torch.cuda.synchronize()
  perpiece={n:dict(equal=torch.equal(t,r),max_abs=float((t.float()-r.float()).abs().max()),rel_rms=float((t.float()-r.float()).square().mean().sqrt()/r.float().square().mean().sqrt())) for n,t,r in zip(['q','k','v','g','bias'],actual,refpro)}
  err=float((y.float()-ref).square().mean().sqrt()/ref.square().mean().sqrt())
  print('CORRECTNESS',label,'rms',err,'equal',torch.equal(y,expected),json.dumps(perpiece),flush=True)
  assert torch.isfinite(y).all()
  assert err < .0025,err
  def pro():return pf.prologue(x,W,impl='fpf')
  times=[graph_us(pro) for _ in range(3)]
  block=[graph_us(call) for _ in range(3)]
  kernels=cupti(call,10)
  assert not any('flash_triattn' in k for k in kernels),kernels
  r=dict(length=a.length,pro_us=statistics.median(times),pro_rounds=times,block_us=statistics.median(block),block_rounds=block,rel_rms=err,bitwise_equal=torch.equal(y,expected),pieces=perpiece,kernels=kernels)
  print('RESULT',label,json.dumps(r),flush=True);results[label]=r
  Path(a.output).write_text(json.dumps(results,indent=2))
 PRO.triatt_prologue=original
