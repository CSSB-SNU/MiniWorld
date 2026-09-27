"""Overlap the preceding dW WGMMA group with the next recompute group."""
from pathlib import Path
from b7 import B7
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
PRE=Path(__file__).resolve().parent.parent/'trimul_d256_bwd_sol90_20260923'
class PipeB7(B7):
 def __init__(self,p,leaves):
  super().__init__(p,8 if p.n==384 else 16)
  body=(PRE/'source.cu').read_text().replace('#include "mma.cuh"',(PRE/'mma.cuh').read_text()).replace('mw_d256_b7_source','mw_d256_b7_pipe')
  # The old input slot and derivative tile become reusable only when both
  # preceding dW and current recompute have completed.
  prefetch='if(tid==0&&tile+1<end)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);'
  assert body.count(prefetch)==1
  body=body.replace(prefetch,'')
  body=body.replace('wgmma_wait<0>();fence_regs(pre);','wgmma_wait<0>();fence_regs(pre);fence_regs(dw0);fence_regs(dw1);\n  '+prefetch)
  body=body.replace('wgmma_commit();wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);__syncthreads();','wgmma_commit();')
  body=body.replace(' for(int j=0;j<64;++j){',' wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);__syncthreads();\n for(int j=0;j<64;++j){')
  flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DWEIGHT_SPLITS={self.splits}']
  out=T.compile_text(body,flags);self.source=T.load_unit(str(out),'mw_d256_b7_pipe').kernel('mw_d256_b7_pipe');self.source.set_max_dynamic_smem(114816)
