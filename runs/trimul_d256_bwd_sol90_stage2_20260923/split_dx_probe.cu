// SPDX-License-Identifier: Apache-2.0
// Numerical probe only. Materializes FP32 partials; not a performance candidate.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
// MMA_HELPERS
struct Params {CUtensorMap dg,wg,gp[4],w[4];float *prefix,*part;int M;};
TMN_DEVI void save(float* out,float (&v)[32],int row,int col){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);out[size_t(row+r)*256+col+c]=v[j];});
}
extern "C" __global__ __launch_bounds__(128,2) void mw_d256_probe_prefix(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto b=reinterpret_cast<uint64_t*>(sm+16384);
 if(threadIdx.x==0){mbar_init(b,1);fence_barrier_init();}__syncthreads();
 int row=(blockIdx.x/4)*64,col=(blockIdx.x%4)*64;float v[32]={};
 for(int k=0;k<4;++k){
  if(threadIdx.x==0){mbar_arrive_expect_tx(b,16384);tma_load_2d(sm,&p.dg,b,k*64,row);tma_load_2d(sm+8192,&p.wg,b,col,k*64);}
  mbar_wait(b,k&1);__syncthreads();fence_regs(v);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;mma64_off<q*32,q*2048,0,1>(v,smem_desc(smem_u32(sm),16,1024,1),smem_desc(smem_u32(sm+8192),16,1024,1),k>0||q>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
 }
 save(p.prefix,v,row,col);
}
extern "C" __global__ __launch_bounds__(128,2) void mw_d256_probe_part(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto b=reinterpret_cast<uint64_t*>(sm+8192);
 if(threadIdx.x==0){mbar_init(b,1);fence_barrier_init();}__syncthreads();
 int row=(blockIdx.x/4)*64,col=(blockIdx.x%4)*64,rank=blockIdx.y%16,side=blockIdx.y/16;float v[32]={};
 for(int plane=0;plane<2;++plane){
  int which=side*2+plane;
  if(threadIdx.x==0){mbar_arrive_expect_tx(b,8192);tma_load_2d(sm,&p.gp[which],b,row,rank*32);tma_load_2d(sm+4096,&p.w[which],b,col,rank*32);}
  mbar_wait(b,plane&1);__syncthreads();fence_regs(v);wgmma_fence();
  static_for<2>([&](auto qq){constexpr int q=decltype(qq)::value;mma64_off<q*2048,q*2048,1,1>(v,smem_desc(smem_u32(sm),16,1024,1),smem_desc(smem_u32(sm+4096),16,1024,1),plane>0||q>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
 }
 save(p.part+size_t(blockIdx.y)*p.M*256,v,row,col);
}
