// SPDX-License-Identifier: Apache-2.0
// Row-major dGate -> channel-major prefix, using coalesced TMA tiles.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
struct Params{CUtensorMap src,dst;int M,D;};
extern "C" __global__ __launch_bounds__(128,4)
void mw_prefix_transpose(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+8192);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,phase=0;
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();
 for(int tile=blockIdx.x;tile<(p.M/64)*(p.D/64);tile+=gridDim.x){
  int row=(tile/(p.D/64))*64,col=(tile%(p.D/64))*64;
  if(tid==0){mbar_arrive_expect_tx(bar,8192);tma_load_2d(sm,&p.src,bar,col,row);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();
  uint32_t f[4][4],base=smem_u32(sm);int mat=lane/8,r8=lane%8;
  #pragma unroll
  for(int q=0;q<4;++q)ldsm_x4_t(f[q],base+swz128(16*q+r8+((mat&2)?8:0),(16*warp+((mat&1)?8:0))*2));
  __syncthreads();
  #pragma unroll
  for(int q=0;q<4;++q){int r=16*warp+r8+((mat&1)?8:0),c=16*q+((mat&2)?8:0);stsm_x4(base+swz128(r,c*2),f[q][0],f[q][1],f[q][2],f[q][3]);}
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dst),"r"(base),"r"(row),"r"(col):"memory");
   tma_store_commit();tma_store_wait_all();
  }
  __syncthreads();
 }
}
