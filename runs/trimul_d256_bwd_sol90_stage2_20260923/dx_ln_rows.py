from pathlib import Path
import ctypes,torch,os
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent;PRE=R.parent/'trimul_d256_bwd_sol90_20260923'
class DxLNRows:
 def __init__(self,p,splits):
  assert p.M%128==0
  body=(R/'dx_ln_rows.cu').read_text().replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text())
  if os.environ.get('DX_N256')=='1':
   from dx_n256 import widen
   body=widen(body)
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={splits}']
  self.cubin=T.compile_text(body,flags);print('DX_ROWS_CUBIN',self.cubin,flush=True)
  self.k=T.load_unit(str(self.cubin),'mw_d256_dx_ln_rows').kernel('mw_d256_dx_ln_rows');self.smem=196736;self.k.set_max_dynamic_smem(self.smem)
  L=T._launch_module();t7=p.tensors.copy();t7[22:24]=[p.dl,p.dr]
  self.params=L.Struct([*p.maps7,*t7,*p.floats,p.M,p.n,W.tm(p.x.reshape(p.M,256)),W.tm(p.dy.reshape(p.M,256)),W.tm(p.dx.reshape(p.M,256))])
  drv=self.k.unit.drv;occ=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(self.k.handle)),384,self.smem)))
  self.grid=torch.cuda.get_device_properties(p.x.device).multi_processor_count*occ
 def __call__(self):
  L=T._launch_module();drv=self.k.unit.drv;args=L._Packed([self.params])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.k.handle)),self.grid,1,1,384,1,1,self.smem,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
