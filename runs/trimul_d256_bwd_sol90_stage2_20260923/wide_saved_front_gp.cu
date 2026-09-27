// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=WIDTH,H=2*D,RANKS=D/8;
struct Params {CUtensorMap pre,dl,dr,gp[4];const bf* mask;int M;};
// PACKED_GLU
extern "C" __global__ __launch_bounds__(128,4)
void mw_wide_saved_front_gp(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+20480);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();int phase=0;
 for(int tile=blockIdx.x;tile<(p.M/64)*RANKS;tile+=gridDim.x){
  int rank=tile%RANKS,row=(tile/RANKS)*64,side=rank/(H/32),col=(rank%(H/32))*32;
  if(tid==0){mbar_arrive_expect_tx(bar,12288);tma_load_2d(sm,&p.pre,bar,0,rank*p.M+row);tma_load_2d(sm+8192,side?&p.dr:&p.dl,bar,row,col);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();float pre[32];
  #pragma unroll
  for(int j=0;j<32;++j){int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);pre[j]=__bfloat162float(reinterpret_cast<bf*>(sm)[swz128(r,c*2)/2]);}
  int ra=warp*16+lane/4;
  uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[row+ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(p.mask[row+ra+8]))*0x10001u;
  packed_glu(pre,sm,sm+12288,ma,mb);
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.gp[2*side]),"r"(smem_u32(sm+16384)),"r"(row),"r"(col):"memory");
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.gp[2*side+1]),"r"(smem_u32(sm+12288)),"r"(row),"r"(col):"memory");
   tma_store_commit();tma_store_wait_all();
  }
  __syncthreads();
 }
}
