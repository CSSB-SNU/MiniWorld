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


EPI=pf._carried('fpf_triatt_epi','epilogue');original_epi=EPI.triatt_epilogue

def cuda_epi(o,g,wo16,z=None,*,ending=False,residual=False,out=None,cfg=None,woT16=None,o_layout='ihjd'):
 assert not ending and residual and o_layout=='ihjd'
 target=z if out is None else out
 ext.epilogue(o,g,wo16,z,target)
 return None if out is None else out

results={}
with torch.no_grad():
 expected=call().clone()
 qp,kp,vp,gp,bp=pf.prologue(x,W,impl='fpf')
 op=pf.core_attention(qp,kp,vp,bp,m5,core='tier:triattn_native')
 refepi=pf.epilogue(op,gp,W,x,impl='fpf',residual=True,out=buf).clone()
 for label,pfn,efn in [('baseline',original,original_epi),('pro_tma',cuda_pro,original_epi),('epi_tma',original,cuda_epi),('both_tma',cuda_pro,cuda_epi)]:
  PRO.triatt_prologue=pfn;EPI.triatt_epilogue=efn
  y=call().clone();torch.cuda.synchronize()
  err=float((y.float()-ref).square().mean().sqrt()/ref.square().mean().sqrt())
  ey=pf.epilogue(op,gp,W,x,impl='fpf',residual=True,out=buf).clone()
  print('CHECK',label,err,torch.equal(ey,refepi),float((ey.float()-refepi.float()).abs().max()),flush=True)
  assert torch.isfinite(y).all() and err<.0025
  funcs={'pro':lambda:pf.prologue(x,W,impl='fpf'),'epi':lambda:pf.epilogue(op,gp,W,x,impl='fpf',residual=True,out=buf),'block':call}
  r={'rel_rms':err,'equal':torch.equal(y,expected),'epi_equal':torch.equal(ey,refepi)}
  for n,fn in funcs.items():
   rounds=[graph_us(fn) for _ in range(3)];r[n+'_us']=statistics.median(rounds);r[n+'_rounds']=rounds
  kernels=cupti(call,10)
  assert not any('flash_triattn' in k for k in kernels)
  r['kernels']=kernels;results[label]=r
  print('RESULT',label,json.dumps(r),flush=True)
  Path(a.output).write_text(json.dumps(results,indent=2))
