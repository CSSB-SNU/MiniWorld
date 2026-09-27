from pathlib import Path
import os,torch
from b7 import B7
from quack.gemm import gemm
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
PRE=Path(__file__).resolve().parent.parent/'trimul_d256_bwd_sol90_20260923'
class CuteDx(B7):
 def __init__(self,p,leaves):
  super().__init__(p,8 if p.n==384 else 16)
  self.w=p.x.new_empty((2048,256));self.gatedx=torch.empty((p.M,256),device=p.x.device,dtype=torch.float32)
  self.gargs=(p.dg.unsqueeze(0),leaves[5].t().unsqueeze(0),self.gatedx.unsqueeze(0),None)
  self.args=(p.gp_all.reshape(2048,p.M).t().unsqueeze(0),self.w.t().unsqueeze(0),p.tensors[10].unsqueeze(0),self.gatedx.unsqueeze(0))
  self.cfg=dict(tile_M=int(os.environ.get('DX_TM','128')),tile_N=int(os.environ.get('DX_TN','128')),cluster_M=1,cluster_N=1,pingpong=os.environ.get('DX_PING','1')=='1')
  body=(PRE/'ring_reduce.cu').read_text().replace('#include "base.cuh"',(PRE/'base.cuh').read_text())
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),'-DWIDTH=256',f'-DWEIGHT_SPLITS={self.splits}']
  out=T.compile_text(body,flags);self.reduce=T.load_unit(str(out),'mw_d256_b7_reduce').kernel('mw_d256_b7_reduce');self.reduce.set_max_dynamic_smem(W.tuning(256)[2])
 def dx(self):
  torch.cat(self.p.weights,out=self.w)
  gemm(*self.gargs,None,128,128,1,1,pingpong=True)
  gemm(*self.args,None,**self.cfg)
 def __call__(self):
  self.source.launch((32*self.splits,1,1),(128,1,1),[self.params],114816)
  self.dx();W.launch(self.reduce,self.p.params7,264,D=256);return self.p.outputs
