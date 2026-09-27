// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=2*WIDTH,ROWS=16,NT=128,SB=ROWS*H*2,BAR=3*SB;
struct Params {CUtensorMap tri,dt,dnmap;const bf* dn;const float *mu,*rs,*gamma;float *dg,*db;int M;};
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
// TRANSPOSE_HELPERS
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int row,int slot){
 mbar_arrive_expect_tx(bar+slot,2*SB);
 for(int c=0;c<H;c+=64){
  tma_load_2d(sm+slot*2*SB+(c/64)*(ROWS*128),&p.tri,bar+slot,row,c);
  tma_load_2d(sm+slot*2*SB+SB+(c/64)*(ROWS*128),&p.dnmap,bar+slot,c,row);
 }
}
TMN_DEVI void normalizer(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;const float* gamma=reinterpret_cast<float*>(sm+BAR+128);
 int first=blockIdx.x*ROWS,step=gridDim.x*ROWS;uint8_t* out=sm+2*SB;
 if(tid==0)load(p,sm,bar,first,0);
 for(int row=first,it=0;row<p.M;row+=step,++it){
  mbar_wait(bar,it&1);named_bar_sync(1,128);transpose32<false>(sm);
  if(tid==0)mbar_arrive(bar+5);
  for(int r=warp;r<ROWS;r+=4){
   float mu=p.mu[row+r],rs=p.rs[row+r],s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]),v=dy*((c<H-32)?gamma[c]:p.gamma[c]);s0+=v;s1+=v*z;
   }
   s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]);
    reinterpret_cast<bf*>(out)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,fmaf(dy,((c<H-32)?gamma[c]:p.gamma[c]),-s0))*rs);
   }
  }
  mbar_wait(bar+2,it&1);named_bar_sync(1,128);
  transpose32<true>(out);fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){for(int c=0;c<H;c+=64)put_tile(&p.dt,out+(c/64)*(ROWS*128),row,c);tma_store_commit();
   if(row+step<p.M)load(p,sm,bar,row+step,0);
   tma_store_wait_read<0>();
  }
  named_bar_sync(1,128);
 }
 if(tid==0){tma_store_wait_all();mbar_arrive(bar+4);}
}
TMN_DEVI void affine(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;float gg[H/32]={},bb[H/32]={};
 for(int row=blockIdx.x*ROWS,it=0;row<p.M;row+=gridDim.x*ROWS,++it){int slot=0;uint8_t* tile=sm;
  mbar_wait(bar+5,it&1);named_bar_sync(2,64);
  for(int r=warp;r<ROWS;r+=2){float mu=p.mu[row+r],rs=p.rs[row+r];
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(tile)[pos(r,c)])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(tile+SB)[pos(r,c)]);gg[q]+=dy*z;bb[q]+=dy;
   }
  }
  named_bar_sync(2,64);if(tid==0)mbar_arrive(bar+2+slot);
 }
 mbar_wait(bar+4,0);named_bar_sync(2,64);float* acc=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<H/32;++q){int c=lane+q*32;acc[warp*H+c]=gg[q];acc[(2+warp)*H+c]=bb[q];}
 named_bar_sync(2,64);
 for(int c=tid;c<H;c+=64){float g=0,b=0;
  #pragma unroll
  for(int w=0;w<2;++w){g+=acc[w*H+c];b+=acc[(2+w)*H+c];}
  atomicAdd(p.dg+c,g);atomicAdd(p.db+c,b);
 }
}
extern "C" __global__ __launch_bounds__(192,3)
void mw_wide_mixed_affine_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);int tid=threadIdx.x;
 for(int c=blockIdx.x*192+tid;c<H;c+=gridDim.x*192){p.dg[c]=0;p.db[c]=0;}
 for(int c=tid;c<H-32;c+=192)reinterpret_cast<float*>(sm+BAR+128)[c]=p.gamma[c];
 if(tid==0){for(int i=0;i<7;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();cooperative_groups::this_grid().sync();
 if(tid<128){normalizer(p,sm,bar);}
 else{affine(p,sm,bar);}
}
