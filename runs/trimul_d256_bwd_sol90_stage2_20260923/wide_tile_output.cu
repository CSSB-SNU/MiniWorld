// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=WIDTH,H=2*D,ROWS=OUTPUT_ROWS,NT=OUTPUT_THREADS,SB=ROWS*H*2;
struct Params {CUtensorMap tri,norm;const float *gamma,*beta;float *mu,*rs;int M;};
// TRANSPOSE_HELPERS
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));return v;}
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
extern "C" __global__ __launch_bounds__(NT,2)
void mw_wide_tile_norm(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+SB);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();int phase=0;
 for(int row=blockIdx.x*ROWS;row<p.M;row+=gridDim.x*ROWS){
  if(tid==0){mbar_arrive_expect_tx(bar,SB);for(int c=0;c<H;c+=64)tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar,row,c);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();transpose32<false>(sm);
  for(int r=warp;r<ROWS;r+=NT/32){
   float s=0,mu,rs;
   if constexpr(D<512){
    float values[H/32];
    #pragma unroll
    for(int q=0;q<H/32;++q){values[q]=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,lane+q*32)]);s+=values[q];}
    mu=sumwarp(s)/H;s=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){float z=values[q]-mu;s+=z*z;}
    rs=rsqrtf(sumwarp(s)/H+1e-5f);
    #pragma unroll
    for(int q=0;q<H/32;++q){int c=lane+q*32;reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf((values[q]-mu)*rs,p.gamma[c],p.beta[c]));}
   }else{
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){uint32_t v=*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));s+=bf16lo(v)+bf16hi(v);}
    mu=sumwarp(s)/H;s=0;
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){uint32_t v=*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));float a=bf16lo(v)-mu,b=bf16hi(v)-mu;s+=a*a+b*b;}
    rs=rsqrtf(sumwarp(s)/H+1e-5f);
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){auto ptr=reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));uint32_t v=*ptr;
     *ptr=pack_bf16(fmaf((bf16lo(v)-mu)*rs,p.gamma[c],p.beta[c]),fmaf((bf16hi(v)-mu)*rs,p.gamma[c+1],p.beta[c+1]));
    }
   }
   if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}
  }
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<H;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.norm),"r"(smem_u32(sm+(c/64)*(ROWS*128))),"r"(c),"r"(row):"memory");
  }tma_store_commit();tma_store_wait_all();}__syncthreads();
 }
}
