// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=2*WIDTH,NT=128,ROWS=LN_ROWS,SB=ROWS*H*2;
struct Params {CUtensorMap tri,dt,dnmap;const bf* dn;const float *mu,*rs,*gamma;float *dg,*db;int M;};
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
// TRANSPOSE_HELPERS
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
extern "C" __global__ __launch_bounds__(NT,LN_MINBLOCKS)
void mw_wide_pair_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+2*SB);
 float* gamma=reinterpret_cast<float*>(sm+2*SB+128);
 float* sums=gamma+H;
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,pair=tid/64,channel=tid%64;
 for(int c=blockIdx.x*NT+tid;c<H;c+=gridDim.x*NT){p.dg[c]=0;p.db[c]=0;}
 for(int c=tid;c<H;c+=NT)gamma[c]=p.gamma[c];
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}
 __syncthreads();cooperative_groups::this_grid().sync();
 float gg[H/64]={},bb[H/64]={};int phase=0;
 for(int row=blockIdx.x*ROWS;row<p.M;row+=gridDim.x*ROWS){
  if(tid==0){mbar_arrive_expect_tx(bar,2*SB);
   for(int c=0;c<H;c+=64){
    tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar,row,c);
    tma_load_2d(sm+SB+(c/64)*(ROWS*128),&p.dnmap,bar,c,row);
   }
  }
  mbar_wait(bar,phase);phase^=1;__syncthreads();transpose32<false>(sm);
  for(int r=pair;r<ROWS;r+=2){
   float mu=p.mu[row+r],rs=p.rs[row+r],s0=0,s1=0;
   float z[H/64],dy[H/64];
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=channel+q*64;
    z[q]=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    dy[q]=__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]);
    float v=dy[q]*gamma[c];s0+=v;s1+=v*z[q];gg[q]+=dy[q]*z[q];bb[q]+=dy[q];
   }
   s0=sumwarp(s0);s1=sumwarp(s1);
   if(lane==0){sums[warp*2]=s0;sums[warp*2+1]=s1;}__syncthreads();
   s0=(sums[pair*4]+sums[pair*4+2])/H;s1=(sums[pair*4+1]+sums[pair*4+3])/H;
   #pragma unroll
   for(int q=0;q<H/64;++q){int c=channel+q*64;float centered=fmaf(dy[q],gamma[c],-s0);
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z[q],s1,centered)*rs);
   }
   __syncthreads();
  }
  transpose32<true>(sm);fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<H;c+=64)put_tile(&p.dt,sm+(c/64)*(ROWS*128),row,c);tma_store_commit();tma_store_wait_all();}
  __syncthreads();
 }
 float* acc=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<H/64;++q){int c=channel+q*64;acc[pair*H+c]=gg[q];acc[(2+pair)*H+c]=bb[q];}
 __syncthreads();
 for(int c=tid;c<H;c+=NT){atomicAdd(p.dg+c,acc[c]+acc[H+c]);atomicAdd(p.db+c,acc[2*H+c]+acc[3*H+c]);}
}
