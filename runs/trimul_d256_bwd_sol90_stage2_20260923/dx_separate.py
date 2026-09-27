"""Experimental N128 three-CTA dX, with a separate saved-stats input LN."""
from pathlib import Path
import ctypes,os,torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
R=Path(__file__).resolve().parent

class DxSeparate:
 def __init__(self,p,splits,slots=2):
  assert slots in (2,3) and hasattr(p,'input_stats')
  body=(R/'dx_separate.cu').read_text().replace('#include "mma.cuh"',(R.parent/'trimul_d256_bwd_sol90_20260923'/'mma.cuh').read_text())
  ws=int(os.environ.get('DX_SEP_WS','0'));self.threads=256 if ws else 128
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={splits}',f'-DDX_SEP_SLOTS={slots}',f'-DDX_SEP_WS={ws}']
  self.cubin=T.compile_text(body,flags)
  unit=T.load_unit(str(self.cubin),'mw_d256_dx_separate');self.k=unit.kernel('mw_d256_dx_separate');self.ln=unit.kernel('mw_d256_input_ln_separate')
  self.smem=slots*24576+128;self.k.set_max_dynamic_smem(self.smem)
  drv=unit.drv;L=T._launch_module();ff=p.floats.copy();ff[12]=p.input_stats
  self.params=L.Struct([*p.maps7,*p.tensors,*ff,p.M,p.n,W.tm(p.tensors[10])])
  def occ(k,nt,sm):return int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(drv.d.CUfunction(int(k.handle)),nt,sm)))
  sms=torch.cuda.get_device_properties(p.x.device).multi_processor_count
  self.grid=sms*occ(self.k,self.threads,self.smem);self.ln_grid=sms*min(occ(self.ln,128,0),int(os.environ.get('DX_SEP_LN_CTA','4')))
  regs=int(drv._unwrap('cuFuncGetAttribute',drv.d.cuFuncGetAttribute(drv.d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,drv.d.CUfunction(int(self.k.handle)))))
  if ws:assert regs*256>=128*(32+128),(regs,self.grid)
  self.metadata=dict(cubin=str(self.cubin),smem=self.smem,grid=self.grid,ln_grid=self.ln_grid,regs=regs,slots=slots,ws=ws,threads=self.threads)
 def gemm(self):self.k.launch((self.grid,1,1),(self.threads,1,1),[self.params],self.smem)
 def normalize(self):
  L=T._launch_module();drv=self.ln.unit.drv;args=L._Packed([self.params])
  drv._unwrap('cuLaunchCooperativeKernel',drv.d.cuLaunchCooperativeKernel(drv.d.CUfunction(int(self.ln.handle)),self.ln_grid,1,1,128,1,1,0,drv.d.CUstream(int(torch.cuda.current_stream().cuda_stream)),ctypes.addressof(args.array)))
 def __call__(self):self.gemm();self.normalize()
