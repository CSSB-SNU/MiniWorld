"""Lossless per-16 adaptive exponent base, mantissa bytes, and exponent nibbles."""
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

class AdaptiveExponentPre:
    def __init__(self,pre):
        self.pre=pre;shape=pre.shape
        assert shape[1]%16==0
        self.low=torch.empty_like(pre,dtype=torch.uint8)
        self.exps=torch.empty((shape[0],shape[1]//2),device=pre.device,dtype=torch.uint8)
        self.bases=torch.empty((shape[0],shape[1]//8),device=pre.device,dtype=torch.uint8)
        body=r"""
#include <stdint.h>
struct P{const uint32_t* raw;uint16_t* low;uint8_t* exps;uint8_t* bases;int pairs;};
extern "C" __global__ void mw_adaptive_exponent_pack(__grid_constant__ const P p){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<p.pairs;i+=gridDim.x*blockDim.x){
  unsigned v=p.raw[i],a=v&65535u,b=v>>16;
  unsigned ea=(a>>7)&255u,eb=(b>>7)&255u;
  unsigned mn=ea<eb?ea:eb,mx=ea>eb?ea:eb;
  #pragma unroll
  for(int q=4;q;q>>=1){
   unsigned n=__shfl_xor_sync(0xffffffff,mn,q,8),x=__shfl_xor_sync(0xffffffff,mx,q,8);
   mn=mn<n?mn:n;mx=mx>x?mx:x;
  }
  unsigned base=(mx-mn<=15u && mn<255u)?mn:255u;
  p.low[i]=(a&127u)|((a>>8)&128u)|((b&127u)<<8)|(b&32768u);
  p.exps[i]=((ea-mn)&15u)|(((eb-mn)&15u)<<4);
  if((i&7)==0){p.bases[i/4]=base;p.bases[i/4+1]=base;}
 }
}
"""
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_adaptive_exponent_pack').kernel('mw_adaptive_exponent_pack')
        self.params=T._launch_module().Struct([pre,self.low,self.exps,self.bases,pre.numel()//2])
    def __call__(self):self.k.launch((1056,1,1),(256,1,1),[self.params],0)
