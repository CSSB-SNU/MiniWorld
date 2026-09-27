// SPDX-License-Identifier: Apache-2.0
// MiniWorld D256 bounded derivative ring, based on the D128 dual-consumer design.
#include <cooperative_groups.h>
#include "ring_source.cuh"
template<bool GP> TMN_DEVI void consumer_gemm(const Params& p,float (&v)[64],int cid,int slot,int plane,int row,int col,uint8_t* sm,uint64_t* bars,int& phase,bool add){
 int end=GP?H:D;
 auto load=[&](int k,int s){
  mbar_arrive_expect_tx(bars+s,24576);uint8_t* dst=sm+s*24576;
  if constexpr(GP)tma_load_3d(dst,&p.ringmap,bars+s,0,k,(cid*RINGS+slot)*4+plane);
  else tma_load_2d(dst,&p.dg,bars+s,k,row);
  for(int n=0;n<2;++n)tma_load_2d(dst+8192*(1+n),GP?&p.weights[plane]:&p.wgate,bars+s,col+64*n,k);
 };
 if(threadIdx.x==0)load(0,0);
 for(int k=0,it=0;k<end;k+=64,++it){int s=it%2;uint8_t* dst=sm+s*24576;
  if(threadIdx.x==0&&k+64<end)load(k+64,1-s);
  mbar_wait(bars+s,(phase>>s)&1);phase^=1<<s;__syncthreads();fence_regs(v);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma128<GP?1:0,1>(v,smem_desc(smem_u32(dst+(GP?q*2048:q*32)),16,1024,1),smem_desc(smem_u32(dst+8192+q*2048),8192,1024,1),add||k>0||q>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
 }
}
TMN_DEVI void ring_consumer(const Params& p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+114688);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,cid=blockIdx.x/GROUP,rank=blockIdx.x%GROUP-SOURCES;
 if(tid==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}__syncthreads();int phase=0;
 int rounds=(p.M/64-1-cid)/p.cohorts+1;
 for(int seq=rank;seq<rounds;seq+=CONSUMERS){
  int row=(cid+seq*p.cohorts)*64,slot=seq%RINGS;unsigned* flags=p.flags+(cid*RINGS+slot)*33;
  if(tid<32)wait_flag(flags+tid,seq+1);__syncthreads();
  for(int col=0;col<D;col+=128){float dx[64]={};
   consumer_gemm<false>(p,dx,cid,slot,0,row,col,sm,bar,phase,false);
   for(int plane=0;plane<4;++plane)consumer_gemm<true>(p,dx,cid,slot,plane,row,col,sm,bar,phase,true);
   for(int j=0;j<64;++j){int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
    p.dxn[size_t(row+r)*D+col+c]=__float2bfloat16_rn(dx[j]);
   }__syncthreads();
  }
  if(tid==0)signal_flag(flags+32,seq+1);
 }
}
extern "C" __global__ __launch_bounds__(128,2) void mw_d256_b7_ring(__grid_constant__ const Params p){
 auto grid=cooperative_groups::this_grid();
 for(int i=blockIdx.x*128+threadIdx.x;i<p.cohorts*RINGS*33;i+=gridDim.x*128)p.flags[i]=0;
 grid.sync();
 if(blockIdx.x%GROUP<SOURCES)ring_source(p);else ring_consumer(p);
}
