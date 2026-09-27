"""Three lossless planes: 8 sign/mantissa bits and two exponent nibbles."""
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from inline_compressed_pool import InlineCompressedPool

class ExponentNibblePre:
    def __init__(self,pre):
        self.pre=pre;shape=pre.shape
        self.low=torch.empty_like(pre,dtype=torch.uint8)
        self.explo=torch.empty((shape[0],shape[1]//2),device=pre.device,dtype=torch.uint8)
        self.pool=InlineCompressedPool()
        with self.pool.context():self.exphi=torch.empty_like(self.explo)
        body='''
#include <stdint.h>
struct P{const uint32_t* raw;uint16_t* low;uint8_t* elo;uint8_t* ehi;int pairs;};
extern "C" __global__ void mw_exponent_nibble_pack(__grid_constant__ const P p){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<p.pairs;i+=gridDim.x*blockDim.x){
  unsigned v=p.raw[i],a=v&65535u,b=v>>16;
  unsigned ea=(((a>>7)&255u)-116u)&255u,eb=(((b>>7)&255u)-116u)&255u;
  p.low[i]=(a&127u)|((a>>8)&128u)|((b&127u)<<8)|(b&32768u);
  p.elo[i]=(ea&15u)|((eb&15u)<<4);p.ehi[i]=(ea>>4)|(eb&240u);
 }
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v']
        self.cubin=T.compile_text(body,flags)
        self.k=T.load_unit(str(self.cubin),'mw_exponent_nibble_pack').kernel('mw_exponent_nibble_pack')
        self.params=T._launch_module().Struct([pre,self.low,self.explo,self.exphi,pre.numel()//2])
    def __call__(self):self.k.launch((1056,1,1),(256,1,1),[self.params],0)
