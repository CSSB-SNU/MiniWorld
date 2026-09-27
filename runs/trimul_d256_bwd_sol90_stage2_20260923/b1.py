from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
from ln import LN
R=Path(__file__).resolve().parent
class FastB1:
 def __init__(self,p,prepare):
  self.p,self.prepare=p,prepare;self.ln=LN(p)
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWEIGHT_SPLITS=32','-DWIDTH=256']
  out=T.compile(R/'math.cu',flags);self.k=T.load_unit(str(out),'mw_d256_b1_math').kernel('mw_d256_b1_math');self.k.set_max_dynamic_smem(W.tuning(256)[2])
  drv=self.k.unit.drv
  occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),256,W.tuning(256)[2])))
  self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*min(2,occ)
 def __call__(self):
  b=self.prepare;b.prepare.launch((b.sms,1,1),(128*b.groups,1,1),[b.params],b.smem)
  W.launch(self.k,self.p.params,self.grid,D=256)
  self.ln()
