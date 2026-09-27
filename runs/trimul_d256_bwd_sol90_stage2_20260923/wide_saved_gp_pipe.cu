// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=WIDTH,H=2*D,RANKS=D/8,SLOTS=GP_PIPE?2:1,OUT=12288*SLOTS,BAR=OUT+8192;
struct Params {CUtensorMap pre,dl,dr,gp[4];const bf* mask;int M;};
// SAVED_GLU
TMN_DEVI void coords(int tile,int M,int& rank,int& row){
 if constexpr(GP_RANK_MAJOR){rank=tile/(M/64);row=(tile%(M/64))*64;}
 else {rank=tile%RANKS;row=(tile/RANKS)*64;}
}
TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int tile,int slot){
 int rank,row;coords(tile,p.M,rank,row);int side=rank/(H/32),col=(rank%(H/32))*32;
 mbar_arrive_expect_tx(bar+slot,12288);
 tma_load_2d(sm+slot*12288,&p.pre,bar+slot,0,rank*p.M+row);
 tma_load_2d(sm+slot*12288+8192,side?&p.dr:&p.dl,bar+slot,row,col);
}
extern "C" __global__ __launch_bounds__(128,4)
void mw_wide_saved_gp_pipe(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,total=(p.M/64)*RANKS;
 if(tid==0){for(int s=0;s<SLOTS;++s)mbar_init(bar+s,1);fence_barrier_init();
  if(blockIdx.x<total)load(p,sm,bar,blockIdx.x,0);
 }
 __syncthreads();
 for(int tile=blockIdx.x,it=0;tile<total;tile+=gridDim.x,++it){
  int slot=it%SLOTS,rank,row;coords(tile,p.M,rank,row);
  int side=rank/(H/32),col=(rank%(H/32))*32;
  mbar_wait(bar+slot,(it/SLOTS)&1);__syncthreads();
  if constexpr(GP_PIPE){if(tid==0 && tile+gridDim.x<total)load(p,sm,bar,tile+gridDim.x,1-slot);}
  int ra=warp*16+lane/4;
  uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[row+ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(p.mask[row+ra+8]))*0x10001u;
  saved_glu(sm+slot*12288,sm+slot*12288,sm+OUT,ma,mb);
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.gp[2*side]),"r"(smem_u32(sm+OUT+4096)),"r"(row),"r"(col):"memory");
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.gp[2*side+1]),"r"(smem_u32(sm+OUT)),"r"(row),"r"(col):"memory");
   tma_store_commit();tma_store_wait_read<0>();
   if constexpr(!GP_PIPE){if(tile+gridDim.x<total)load(p,sm,bar,tile+gridDim.x,0);}
  }
  __syncthreads();
 }
 if(tid==0)tma_store_wait_all();
}
