// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=WIDTH,H=2*D,NT=256,ROWS=32,DP=64*D*2,DN=DP+32768,BAR=DN+64*H*2,GAM=DP;
struct Params{CUtensorMap tri,dt,dp,wp;const float *mu,*rs,*gamma;float *dg,*db;bf* dn;int M;};
// MMA_HELPERS
// TRANSPOSE_HELPERS
// AFFINE_HELPER
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI void load_weight(const Params& p,uint8_t* sm,uint64_t* bar,int col,int ki,int slot){
 mbar_arrive_expect_tx(bar+1+slot,16384);
 for(int g=0;g<4;++g)tma_load_2d(sm+DP+slot*16384+g*4096,&p.wp,bar+1+slot,col+g*64,ki);
}
extern "C" __global__ __launch_bounds__(NT,1)
void mw_wide_fused_dn_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,wg=tid/128,ww=warp%4;
 for(int c=blockIdx.x*NT+tid;c<H;c+=gridDim.x*NT){p.dg[c]=0;p.db[c]=0;}
 if(tid==0){for(int i=0;i<4;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();cooperative_groups::this_grid().sync();
 float gg[H/32]={},bb[H/32]={};
 for(int row=blockIdx.x*64,round=0;row<p.M;row+=gridDim.x*64,++round){
  if(tid==0){mbar_arrive_expect_tx(bar,DP);for(int c=0;c<D;c+=64)tma_load_2d(sm+(c/64)*8192,&p.dp,bar,c,row);}
  mbar_wait(bar,round&1);__syncthreads();
  for(int col=0;col<H;col+=256){
   float v[64]={};if(tid==0)load_weight(p,sm,bar,col,0,0);
   for(int ki=0,it=0;ki<D;ki+=32,++it){
    int slot=it%2;mbar_wait(bar+1+slot,(it/2)&1);__syncthreads();
    if(tid==0 && ki+32<D)load_weight(p,sm,bar,col,ki+32,1-slot);
    fence_regs(v);wgmma_fence();
    static_for<2>([&](auto qq){constexpr int q=decltype(qq)::value;
     mma128_off<q*32,q*2048,0,1>(v,smem_desc(smem_u32(sm+(ki/64)*8192+(ki%64)*2),16,1024,1),smem_desc(smem_u32(sm+DP+slot*16384+wg*8192),4096,1024,1),it>0||q>0);
    });wgmma_commit();wgmma_wait<0>();fence_regs(v);
   }
   static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
    int r=ww*16+lane/4+8*((j/2)&1),c=col+wg*128+(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
    uint32_t raw=pack_bf16(v[j],v[j+1]);*reinterpret_cast<uint32_t*>(sm+DN+2*(r*H+c))=raw;
    if constexpr(EMIT_DN)*reinterpret_cast<uint32_t*>(p.dn+size_t(row+r)*H+c)=raw;
   });__syncthreads();
  }
  // The input/weight area becomes one 32-row triangle and output tile.
  for(int c=tid;c<H;c+=NT)reinterpret_cast<float*>(sm+GAM)[c]=p.gamma[c];
  __syncthreads();
  for(int half=0;half<2;++half){
   if(tid==0){mbar_arrive_expect_tx(bar+3,ROWS*H*2);
    for(int c=0;c<H;c+=64)tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar+3,row+half*ROWS,c);
   }
   mbar_wait(bar+3,half&1);__syncthreads();transpose32<false>(sm);
   for(int r=warp;r<ROWS;r+=NT/32){
    float mu=p.mu[row+half*ROWS+r],rs=p.rs[row+half*ROWS+r],s0=0,s1=0;
    #pragma unroll
    for(int q=0;q<H/32;++q){int c=lane+q*32;
     float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
     float dy=__bfloat162float(reinterpret_cast<bf*>(sm+DN)[(half*ROWS+r)*H+c]),v=dy*reinterpret_cast<float*>(sm+GAM)[c];
     s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;
    }
    s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
    #pragma unroll
    for(int q=0;q<H/32;++q){int c=lane+q*32;
     float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
     float dy=__bfloat162float(reinterpret_cast<bf*>(sm+DN)[(half*ROWS+r)*H+c]);
     float centered=fmaf(dy,reinterpret_cast<float*>(sm+GAM)[c],-s0);
     reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,centered)*rs);
    }
   }
   __syncthreads();transpose32<true>(sm);fence_proxy_async();__syncthreads();
   if(tid==0){
    for(int c=0;c<H;c+=64)asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dt),"r"(smem_u32(sm+(c/64)*(ROWS*128))),"r"(row+half*ROWS),"r"(c):"memory");
    tma_store_commit();tma_store_wait_read<0>();
   }__syncthreads();
  }
 }
 if(tid==0)tma_store_wait_all();__syncthreads();
 aggregate_ln<H,NT>(gg,bb,reinterpret_cast<float*>(sm),p.dg,p.db);
}
