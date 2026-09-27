from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
from b7 import B7
PRE=Path(__file__).resolve().parent.parent/'trimul_d256_bwd_sol90_20260923'
class BlasDxB7:
 def __init__(self,p,leaves,row_major=False):
  self.p=p;self.native=B7(p,8 if p.n==384 else 16);self.wg=leaves[5]
  self.row_major=row_major;self.g=p.x.new_empty((p.M,2304) if row_major else (2304,p.M));self.weights=p.x.new_empty((2304,256))
  self.gate=self.g[:,:256] if row_major else self.g[:256]
  self.gp=[self.g[:,256+i*512:256+(i+1)*512] for i in range(4)] if row_major else list(self.g[256:].reshape(4,512,p.M).unbind())
  L=T._launch_module();dy=lambda x:L.tensor_map(x,[64,32],dims=[p.M,512],strides_bytes=[p.M*2],swizzle='128B',l2='128B')
  self.params=L.Struct([p.maps[0],W.tm(p.w1),dy(p.dl),dy(p.dr),self.native.mask,*self.gp,p.floats[7],p.M])
  body=(PRE/'ring_reduce.cu').read_text().replace('#include "base.cuh"',(PRE/'base.cuh').read_text())
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=256',f'-DWEIGHT_SPLITS={self.native.splits}']
  self.source=self.native.source
  if row_major:
   source=(PRE/'source.cu').read_text().replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text()).replace('mw_d256_b7_source','mw_d256_b7_source_rm').replace('size_t(col)*p.M+row+r','size_t(row+r)*2304+col')
   out=T.compile_text(source,flags);self.source=T.load_unit(str(out),'mw_d256_b7_source_rm').kernel('mw_d256_b7_source_rm');self.source.set_max_dynamic_smem(114816)
  out=T.compile_text(body,flags);self.reduce=T.load_unit(str(out),'mw_d256_b7_reduce').kernel('mw_d256_b7_reduce');self.reduce.set_max_dynamic_smem(W.tuning(256)[2])
 def __call__(self):
  p=self.p;n=self.native
  torch.cat((self.wg,*p.weights),out=self.weights)
  self.gate.copy_(p.dg if self.row_major else p.dg.t())
  self.source.launch((32*n.splits,1,1),(128,1,1),[self.params],114816)
  torch.mm(self.g if self.row_major else self.g.t(),self.weights,out=p.tensors[10])
  W.launch(self.reduce,p.params7,264,D=256)
  return p.outputs
