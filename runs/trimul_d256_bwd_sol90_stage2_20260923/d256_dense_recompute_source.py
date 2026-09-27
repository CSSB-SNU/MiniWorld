"""Separate ordered projection, native packed GLU and split FP32 input dW."""
import torch
from lt_contract import LtBmm
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


class DenseRecomputeSource:
    def __init__(self,plan):
        p=plan.p;self.p=p;assert (p.D,p.n)==(256,384)
        self.pre=torch.empty((2048,p.M),device=p.x.device,dtype=torch.bfloat16)
        self.workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
        self.products=LtBmm(p.w1.unsqueeze(0),p.xn.reshape(p.M,256).t().unsqueeze(0),self.pre.unsqueeze(0),self.workspace)
        splits=plan.b7.splits;step=p.M//splits
        self.partial=p.floats[7].reshape(-1)[3*256*256:].as_strided((splits,2048,256),(11*256*256,256,1))
        a=p.gp_all.as_strided((splits,2048,step),(step,p.M,1))
        b=p.xn.as_strided((splits,step,256),(step*256,256,1))
        self.weights=LtBmm(a,b,self.partial,self.workspace)
        body='''#include "tmn_kernels.cuh"
using namespace tmn;using bf=__nv_bfloat16;
struct Params{const bf *pre,*dl,*dr,*mask;bf *gp;int M;};
extern "C" __global__ void mw_d256_dense_recompute_glu(__grid_constant__ const Params p){
 for(int i=blockIdx.x*256+threadIdx.x;i<1024*p.M/2;i+=gridDim.x*256){
  int channel=i/(p.M/2),r=i%(p.M/2),rank=channel/32,c=channel%32,side=channel/512;
  const uint32_t* pre=reinterpret_cast<const uint32_t*>(p.pre);
  uint32_t gr=pre[(rank*64+c)*(p.M/2)+r],pr=pre[(rank*64+c+32)*(p.M/2)+r];
  uint32_t incoming=reinterpret_cast<const uint32_t*>(side?p.dr:p.dl)[(channel%512)*(p.M/2)+r];
  uint32_t mask=reinterpret_cast<const uint32_t*>(p.mask)[r],masked;
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(incoming),"r"(mask));
  float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr));
  uint32_t* out=reinterpret_cast<uint32_t*>(p.gp);
  int base=(side*1024+channel%512)*(p.M/2)+r;
  out[base]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  out[base+512*(p.M/2)]=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1-gb));
 }
}
'''
        self.source_text=body
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc')]
        self.cubin=T.compile_text(body,flags);self.k=T.load_unit(str(self.cubin),'mw_d256_dense_recompute_glu').kernel('mw_d256_dense_recompute_glu')
        self.params=T._launch_module().Struct([self.pre,p.dl,p.dr,plan.b7.params.fields[8],p.gp_all,p.M])

    def epi(self):self.k.launch((1056,1,1),(256,1,1),[self.params],0)
    def __call__(self):self.products();self.epi();self.weights()
