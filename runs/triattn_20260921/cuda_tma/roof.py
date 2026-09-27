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


with torch.no_grad():
 PRO.triatt_prologue=cuda_pro
 def pro():return pf.prologue(x,W,impl='fpf')
 print('TIME',graph_us(pro),flush=True)
 for _ in range(50):pro()
 torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStart()
 pro()
 torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStop()
