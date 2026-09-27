"""Independent weight gradients with explicit FP32 split partials."""
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from lt_contract import LtBmm

class SplitWeightProduct:
    def __init__(self,a,b,out,splits):
        rows,m=a.shape;rows2,n=b.shape
        assert rows==rows2 and rows%splits==0
        self.partial=torch.empty((splits,m,n),device=a.device,dtype=torch.float32)
        self.workspace=torch.empty(64*1024*1024,device=a.device,dtype=torch.uint8)
        step=rows//splits
        at=a.as_strided((splits,m,step),(step*a.stride(0),a.stride(1),a.stride(0)))
        bt=b.as_strided((splits,step,n),(step*b.stride(0),b.stride(0),b.stride(1)))
        self.matmul=LtBmm(at,bt,self.partial,self.workspace)
        body='''
#include "tmn_kernels.cuh"
struct P{const float* partial;__nv_bfloat16* out;int elements;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_split_weight_reduce(__grid_constant__ const P p){
 for(int i=blockIdx.x*256+threadIdx.x;i<p.elements;i+=gridDim.x*256){float v=0;
  #pragma unroll
  for(int s=0;s<SPLITS;++s)v+=p.partial[size_t(s)*p.elements+i];
  p.out[i]=__float2bfloat16_rn(v);
 }
}
'''
        flags=['-std=c++17','-O3','-arch=sm_90a','--cubin','-lineinfo','-Xptxas=-v','-I'+str(T._upstream()/'csrc'),f'-DSPLITS={splits}']
        self.cubin=T.compile_text(body,flags)
        self.reduce=T.load_unit(str(self.cubin),'mw_split_weight_reduce').kernel('mw_split_weight_reduce')
        self.params=T._launch_module().Struct([self.partial,out,m*n])
    def __call__(self):
        self.matmul();self.reduce.launch((264,1,1),(256,1,1),[self.params],0)
