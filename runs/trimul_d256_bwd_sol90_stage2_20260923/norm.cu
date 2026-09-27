// SPDX-License-Identifier: Apache-2.0
// MiniWorld D256 output-LN gradient: TMA load/store and warpgroup transposes.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=512,NT=256,SB=65536;
struct Params {CUtensorMap tri,norm;const float *gamma,*beta;float *mu,*rs;int M;};
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
// This operation is its own inverse: [c,r] <-> [r,c] within each 64x64 tile.
TMN_DEVI void transpose(uint8_t* sm){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 for(int kc=tid/128;kc<8;kc+=2){
  uint32_t f[4][4],base=smem_u32(sm+kc*8192);int mat=lane/8,r8=lane%8,wiw=warp%4;
  #pragma unroll
  for(int q=0;q<4;++q)ldsm_x4_t(f[q],base+swz128(16*q+r8+((mat&2)?8:0),(16*wiw+((mat&1)?8:0))*2));
  named_bar_sync(1+tid/128,128);
  #pragma unroll
  for(int q=0;q<4;++q){int r=16*wiw+r8+((mat&1)?8:0),c=16*q+((mat&2)?8:0);stsm_x4(base+swz128(r,c*2),f[q][0],f[q][1],f[q][2],f[q][3]);}
 }
 __syncthreads();
}
TMN_DEVI void store_tile(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}
extern "C" __global__ __launch_bounds__(NT,2) void mw_d256_norm(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+SB);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();int phase=0;
 for(int row=blockIdx.x*64;row<p.M;row+=gridDim.x*64){
  if(tid==0){mbar_arrive_expect_tx(bar,SB);for(int c=0;c<H;c+=64)tma_load_2d(sm+c*128,&p.tri,bar,row,c);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();transpose(sm);
  for(int r=warp;r<64;r+=8){float v[16],s=0;
   #pragma unroll
   for(int q=0;q<16;++q){int c=lane+q*32;v[q]=__bfloat162float(reinterpret_cast<bf*>(sm+(c/64)*8192)[swz128(r,(c%64)*2)/2]);s+=v[q];}
   float mu=sumwarp(s)/H;s=0;
   #pragma unroll
   for(int q=0;q<16;++q){float z=v[q]-mu;s+=z*z;}
   float rs=rsqrtf(sumwarp(s)/H+1e-5f);if(lane==0){p.mu[row+r]=mu;p.rs[row+r]=rs;}
   #pragma unroll
   for(int q=0;q<16;++q){int c=lane+q*32;reinterpret_cast<bf*>(sm+(c/64)*8192)[swz128(r,(c%64)*2)/2]=__float2bfloat16_rn(fmaf((v[q]-mu)*rs,p.gamma[c],p.beta[c]));}
  }
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<H;c+=64)store_tile(&p.norm,sm+c*128,c,row);tma_store_commit();tma_store_wait_all();}__syncthreads();
 }
}
struct Epi {const bf *proj,*gate,*dy,*ds;bf *dp,*dg;int elements,period;};
extern "C" __global__ __launch_bounds__(256,4) void mw_d256_gate_grad(__grid_constant__ const Epi p){
 for(int i=blockIdx.x*256+threadIdx.x;i<p.elements/2;i+=gridDim.x*256){
  uint32_t pr=reinterpret_cast<const uint32_t*>(p.proj)[i],ga=reinterpret_cast<const uint32_t*>(p.gate)[i],dy=reinterpret_cast<const uint32_t*>(p.dy)[i],ds=reinterpret_cast<const uint32_t*>(p.ds)[i%(p.period/2)];
  float a=math::round_bf16(bf16lo(dy)*bf16lo(ds)),b=math::round_bf16(bf16hi(dy)*bf16hi(ds));float g0=math::sigmoid(bf16lo(ga)),g1=math::sigmoid(bf16hi(ga));
  reinterpret_cast<uint32_t*>(p.dp)[i]=pack_bf16(a*g0,b*g1);
  reinterpret_cast<uint32_t*>(p.dg)[i]=pack_bf16(((a*bf16lo(pr))*g0)*(1-g0),((b*bf16hi(pr))*g1)*(1-g1));
 }
}
