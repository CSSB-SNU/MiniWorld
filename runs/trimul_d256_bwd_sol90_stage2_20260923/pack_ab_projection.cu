#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
#include "ab_projection_codec.cuh"
extern "C" __global__ void mw_pack_ab_projection(const bf* pre,const bf* ab,const bf* mask,uint8_t* codes,bf* escapes,unsigned long long* stats,int M,int D){
 unsigned long long counts[5]={};
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(D)*M;i+=size_t(gridDim.x)*blockDim.x){
  size_t base=i*4;int ch=base/M,row=base%M;size_t pi=size_t((ch/32)*64+ch%32)*M+row;
  uint32_t packed=0;
  #pragma unroll
  for(int q=0;q<4;++q){
   uint32_t g=__bfloat16_as_ushort(pre[pi+q]),p=__bfloat16_as_ushort(pre[pi+size_t(32)*M+q]);
   uint32_t a=__bfloat16_as_ushort(ab[base+q]),m=__bfloat16_as_ushort(mask[row+q]);
   unsigned code=3;
   if((m&0x7fffu)==0&&(g&0x7f80u)!=0x7f80u&&(p&0x7f80u)!=0x7f80u){code=0;counts[4]++;}
   else{
    unsigned estimated=ab_projection_estimate(a,math::sigmoid(__uint_as_float(g<<16)),m);
    unsigned delta=(p-estimated)&0xffffu;
    if(delta==0)code=0;else if(delta==1)code=1;else if(delta==0xffffu)code=2;
   }
   counts[code]++;packed|=code<<(2*q);
   if(code==3)escapes[base+q]=__ushort_as_bfloat16(p);
  }
  codes[i]=packed;
 }
 for(int q=0;q<5;++q){
  for(int off=16;off;off/=2)counts[q]+=__shfl_down_sync(0xffffffffu,counts[q],off);
  if(threadIdx.x%32==0)atomicAdd(stats+q,counts[q]);
 }
}
