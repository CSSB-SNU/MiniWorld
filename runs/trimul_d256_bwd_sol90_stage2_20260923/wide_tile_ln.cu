// SPDX-License-Identifier: Apache-2.0
// Small TMA tiles permit more resident output-LN CTAs at wide dimensions.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=2*WIDTH,NT=128,ROWS=LN_ROWS,SB=ROWS*H*2;
struct Params {CUtensorMap tri,dt,dnmap;const bf* dn;const float *mu,*rs,*gamma;float *dg,*db;int M;};
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
// TRANSPOSE_HELPERS
// AFFINE_HELPER
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI float getdy(const Params& p,uint8_t* sm,int row,int r,int c){
 if constexpr(LN_DN_TMA)return __bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]);
 else return __bfloat162float(p.dn[size_t(row+r)*H+c]);
}
extern "C" __global__ __launch_bounds__(NT,LN_MINBLOCKS)
void mw_wide_tile_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+SB*(1+LN_DN_TMA));
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 for(int c=blockIdx.x*NT+tid;c<H;c+=gridDim.x*NT){p.dg[c]=0;p.db[c]=0;}
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}
 __syncthreads();cooperative_groups::this_grid().sync();
 float gg[H/32]={},bb[H/32]={};int phase=0;
 for(int row=blockIdx.x*ROWS;row<p.M;row+=gridDim.x*ROWS){
  if(tid==0){mbar_arrive_expect_tx(bar,SB*(1+LN_DN_TMA));
   for(int c=0;c<H;c+=64){
    tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar,row,c);
    if constexpr(LN_DN_TMA)tma_load_2d(sm+SB+(c/64)*(ROWS*128),&p.dnmap,bar,c,row);
   }
  }
  mbar_wait(bar,phase);phase^=1;__syncthreads();transpose32<false>(sm);
  for(int r=warp;r<ROWS;r+=4){
   float mu=p.mu[row+r],rs=p.rs[row+r],s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    float dy=getdy(p,sm,row,r,c),v=dy*p.gamma[c];
    s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;
    if constexpr(LN_FENCE)asm volatile("":::"memory");
   }
   s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    float dy=getdy(p,sm,row,r,c),centered=fmaf(dy,p.gamma[c],-s0);
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,centered)*rs);
    if constexpr(LN_FENCE)asm volatile("":::"memory");
   }
  }
  __syncthreads();transpose32<true>(sm);fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<H;c+=64)put_tile(&p.dt,sm+(c/64)*(ROWS*128),row,c);tma_store_commit();tma_store_wait_all();}
  __syncthreads();
 }
 aggregate_ln<H,NT>(gg,bb,reinterpret_cast<float*>(sm),p.dg,p.db);
}
