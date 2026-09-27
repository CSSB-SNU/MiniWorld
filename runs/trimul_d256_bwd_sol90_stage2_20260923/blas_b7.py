from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
from b7 import B7
R=Path(__file__).resolve().parent;PRE=R.parent/'trimul_d256_bwd_sol90_20260923'
class BlasB7:
 def __init__(self,p,leaves):
  self.p=p;self.native=B7(p,8 if p.n==384 else 16)
  self.pre=p.x.new_empty((2048,p.M));self.w=p.x.new_empty((2048,256));self.dw=p.x.new_empty((2048,256))
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=256','-DWEIGHT_SPLITS=8']
  out=T.compile(R/'blas_b7.cu',flags);self.epi=T.load_unit(str(out),'mw_d256_blas_gp').kernel('mw_d256_blas_gp')
  body=(PRE/'finish.cu').read_text().replace('for(int i=0;i<4;++i)reduce_w(p,(H+D+i*H)*D,H*D,p.t[17+i]);','').replace('mw_d256_b7_finish','mw_d256_blas_finish').replace('#include "base.cuh"',(PRE/'base.cuh').read_text())
  out=T.compile_text(body,flags);self.finish=T.load_unit(str(out),'mw_d256_blas_finish').kernel('mw_d256_blas_finish');self.finish.set_max_dynamic_smem(W.tuning(256)[2])
  self.ep=T._launch_module().Struct([self.pre,p.dl,p.dr,self.native.mask,p.gp_all,p.M])
 def products(self):
  torch.cat(self.p.weights,out=self.w);torch.mm(self.w,self.p.xn.reshape(self.p.M,256).t(),out=self.pre)
 def epilogue(self):self.epi.launch((528,1,1),(256,1,1),[self.ep],0)
 def weights(self):
  torch.mm(self.p.gp_all.reshape(2048,self.p.M),self.p.xn.reshape(self.p.M,256),out=self.dw)
  for i in range(4):self.p.dw[i].copy_(self.dw[i*512:(i+1)*512])
 def __call__(self):
  self.products();self.epilogue();self.weights();W.launch(self.finish,self.p.params7,self.native.grid,D=256);return self.p.outputs
