// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=1024,ROWS=16,NT=256,SB=ROWS*H*2;
struct Params {CUtensorMap tri,norm,legacy;const float *gamma,*beta;float *mu,*rs;int M;};
// TRANSPOSE_HELPERS
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));return v;}
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
extern "C" __global__ __launch_bounds__(NT,2)
void mw_wide_parallel_dual_norm(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+2*SB);
 auto gamma=reinterpret_cast<float*>(sm+2*SB+128);auto beta=gamma+H;
 int tid=threadIdx.x,lane=tid%32,warp=(tid/32)%4,wg=tid/128;
 for(int c=tid;c<H;c+=NT){gamma[c]=p.gamma[c];beta[c]=p.beta[c];}
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();int phase=0;
 for(int row=blockIdx.x*ROWS;row<p.M;row+=gridDim.x*ROWS){
  if(tid==0){mbar_arrive_expect_tx(bar,SB);for(int c=0;c<H;c+=64)tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar,row,c);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();transpose32<false>(sm);
  for(int r=warp;r<ROWS;r+=4){
   float values[32],s=0,mu,rs;
   if(wg==1){
    #pragma unroll
    for(int q=0;q<32;++q){values[q]=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,lane+q*32)]);s+=values[q];}
   }else{
    #pragma unroll
    for(int q=0;q<16;++q){uint32_t v=*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,2*lane+64*q));values[2*q]=bf16lo(v);values[2*q+1]=bf16hi(v);s+=values[2*q]+values[2*q+1];}
   }
   // Both arithmetic orders have read this row before either publishes it.
   __syncthreads();mu=sumwarp(s)/H;s=0;
   if(wg==1){
    #pragma unroll
    for(int q=0;q<32;++q){float z=values[q]-mu;s+=z*z;}
   }else{
    #pragma unroll
    for(int q=0;q<16;++q){float a=values[2*q]-mu,b=values[2*q+1]-mu;s+=a*a+b*b;}
   }
   rs=rsqrtf(sumwarp(s)/H+1e-5f);
   if(wg==1){
    if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}
    #pragma unroll
    for(int q=0;q<32;++q){int c=lane+q*32;reinterpret_cast<bf*>(sm+SB)[pos(r,c)]=__float2bfloat16_rn(fmaf((values[q]-mu)*rs,gamma[c],beta[c]));}
   }else{
    #pragma unroll
    for(int q=0;q<16;++q){int c=2*lane+64*q;*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c))=pack_bf16(fmaf((values[2*q]-mu)*rs,gamma[c],beta[c]),fmaf((values[2*q+1]-mu)*rs,gamma[c+1],beta[c+1]));}
   }
  }
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<H;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.norm),"r"(smem_u32(sm+(c/64)*(ROWS*128))),"r"(c),"r"(row):"memory");
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.legacy),"r"(smem_u32(sm+SB+(c/64)*(ROWS*128))),"r"(c),"r"(row):"memory");
  }tma_store_commit();tma_store_wait_all();}__syncthreads();
 }
}
