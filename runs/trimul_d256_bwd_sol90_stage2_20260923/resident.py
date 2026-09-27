from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
R=Path(__file__).resolve().parent
class Resident:
 def __init__(self,p):
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(F.headers()),'-DTMN_SIGMOID_TANH=1','-DMW_MINB=1','-DMW_K1_STREAM=0']
  out=T.compile(R/'resident.cu',flags);self.k=T.load_unit(str(out),'mw_d256_prepare_resident').kernel('mw_d256_prepare_resident');self.smem=229504;self.k.set_max_dynamic_smem(self.smem)
  L=T._launch_module();tri=L.tensor_map(p.tri,[32,64],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='64B',l2='128B')
  self.params=L.Struct([tri,p.maps[0],p.maps[1],p.maps[2],p.dy,p.ds,p.tensors[6],p.tensors[7],p.dg,p.floats[2],p.floats[3],p.floats[5],p.floats[6],p.M,p.n])
  self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count//2*2
 def __call__(self):self.k.launch((self.grid,1,1),(256,1,1),[self.params],self.smem)
