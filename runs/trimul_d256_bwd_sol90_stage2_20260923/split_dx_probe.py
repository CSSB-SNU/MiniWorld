"""FP32 WGMMA prefix and rank-local contributions, for numerical tests only."""
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
R=Path(__file__).resolve().parent
class SplitDXProbe:
 def __init__(self,p,weights):
  body=(R/'split_dx_probe.cu').read_text().replace('// MMA_HELPERS',(R/'mma_offset.cuh').read_text())
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc')]
  self.cubin=T.compile_text(body,flags);print('SPLIT_PROBE_CUBIN',self.cubin,flush=True)
  unit=T.load_unit(str(self.cubin),'mw_d256_probe_prefix');self.pre=unit.kernel('mw_d256_probe_prefix');self.part=unit.kernel('mw_d256_probe_part')
  self.pre.set_max_dynamic_smem(16512);self.part.set_max_dynamic_smem(8320)
  self.prefix=torch.empty((p.M,256),dtype=torch.float32,device=p.x.device)
  self.parts=torch.empty((32,p.M,256),dtype=torch.float32,device=p.x.device)
  L=T._launch_module();gp=lambda x:L.tensor_map(x,[64,32],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
  wt=lambda x:L.tensor_map(x,[64,32],dims=[256,512],strides_bytes=[256*2],swizzle='128B',l2='128B')
  self.params=L.Struct([p.maps[5],p.maps[2],*[gp(x) for x in p.gp],*[wt(w) for w in weights],self.prefix,self.parts,p.M]);self.grid=p.M//64*4
 def __call__(self):
  self.pre.launch((self.grid,1,1),(128,1,1),[self.params],16512)
  self.part.launch((self.grid,32,1),(128,1,1),[self.params],8320)
  return self.prefix,self.parts
