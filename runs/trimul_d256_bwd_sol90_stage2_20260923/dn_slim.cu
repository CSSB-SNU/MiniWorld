// SPDX-License-Identifier: Apache-2.0
// D256 dNorm + output LN: 64 KiB dNorm and 32 KiB reusable weights/LN tile.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=512,BUF=65536,BARS=98304;
constexpr int ROWS=DNS_LN_ROWS,LB=ROWS*H*2,NBUF=32/ROWS;
struct Params {CUtensorMap tri,dt,dp,wp;const bf* dn;const float *mu,*rs,*gamma;float *dg,*db;int M;};
// RS_MMA_HELPER
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
TMN_DEVI int dnpos(int r,int c){return (c/64)*4096+swz128(r,(c%64)*2)/2;}
TMN_DEVI int tripos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI uint32_t rawpos(int c,int r){uint32_t v=c*(ROWS*2)+r*2;if constexpr(ROWS==32)return sw64(v);else return v^(((v>>7)&1u)<<4);}
// One or two warps transpose each 64-channel by ROWS-row tile in place.
template<bool INVERSE> TMN_DEVI void transpose32(uint8_t* sm){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,mat=lane/8,r8=lane%8,w=warp%(ROWS/16);
 for(int kc=warp/(ROWS/16);kc<8;kc+=4/(ROWS/16)){
  uint32_t f[4][4],base=smem_u32(sm+kc*(ROWS*128));
  #pragma unroll
  for(int q=0;q<4;++q){
   uint32_t raw=rawpos(16*q+r8+((mat&2)?8:0),16*w+((mat&1)?8:0));
   uint32_t row=swz128(16*w+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
   ldsm_x4_t(f[q],base+(INVERSE?row:raw));
  }
  named_bar_sync(1,128);
  #pragma unroll
  for(int q=0;q<4;++q){
   uint32_t raw=rawpos(16*q+r8+((mat&2)?8:0),16*w+((mat&1)?8:0));
   uint32_t row=swz128(16*w+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
   stsm_x4(base+(INVERSE?raw:row),f[q][0],f[q][1],f[q][2],f[q][3]);
  }
 }
 named_bar_sync(1,128);
}
TMN_DEVI void put_tile(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}
TMN_DEVI void producer(const Params& p,uint8_t* sm,uint64_t* bars){
 if(threadIdx.x)return;
 for(int row=blockIdx.x*64,round=0;row<p.M;row+=gridDim.x*64,++round){
  if constexpr(DNS_STATS_TMA){
   mbar_arrive_expect_tx(bars+6+2*NBUF,512);
   asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(sm+BARS+128)),"l"(p.mu+row),"n"(256),"r"(smem_u32(bars+6+2*NBUF)):"memory");
   asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(smem_u32(sm+BARS+384)),"l"(p.rs+row),"n"(256),"r"(smem_u32(bars+6+2*NBUF)):"memory");
  }
  mbar_arrive_expect_tx(bars,32768);
  for(int k=0;k<4;++k)tma_load_2d(sm+BUF+k*8192,&p.dp,bars,k*64,row);
  mbar_wait(bars+1,round&1);
  for(int step=0;step<16;++step){
   int slot=step%2,it=round*16+step;
   if(it>=2)mbar_wait(bars+4+slot,((it/2)-1)&1);
   mbar_arrive_expect_tx(bars+2+slot,16384);
   for(int g=0;g<2;++g)tma_load_2d(sm+BUF+slot*16384+g*8192,&p.wp,bars+2+slot,(step/4)*128+g*64,(step%4)*64);
  }
  mbar_wait(bars+4,1);mbar_wait(bars+5,1);
  for(int half=0;half<64/ROWS;++half){int slot=half%NBUF;
   if(half>=NBUF)mbar_wait(bars+6+NBUF+slot,((half/NBUF)-1)&1);
   mbar_arrive_expect_tx(bars+6+slot,LB);
   for(int c=0;c<H;c+=64)tma_load_2d(sm+BUF+slot*LB+(c/64)*(ROWS*128),&p.tri,bars+6+slot,row+half*ROWS,c);
  }
  for(int slot=0;slot<NBUF;++slot)mbar_wait(bars+6+NBUF+slot,1);
 }
}
TMN_DEVI void consumer(const Params& p,uint8_t* sm,uint64_t* bars){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
 float gg[16]={},bb[16]={};
 float gamma[16];
 if constexpr(DNS_CACHE_GAMMA){
  #pragma unroll
  for(int q=0;q<16;++q)gamma[q]=p.gamma[lane+q*32];
 }
 for(int row=blockIdx.x*64,round=0;row<p.M;row+=gridDim.x*64,++round){
  mbar_wait(bars,round&1);named_bar_sync(1,128);
  uint32_t fa[16][4];load_frag_bf16<16,8192>(fa,smem_u32(sm+BUF),warp*16,lane);
  uint32_t dep=0;
  static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;dep^=fa[k][0]^fa[k][1]^fa[k][2]^fa[k][3];});
  fence_proxy_async();named_bar_sync(1,128);if(tid==0)mbar_arrive_dep(bars+1,zero_dep(dep));
  for(int col=0;col<H;col+=128){
   float v0[32]={},v1[32]={};
   static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value,slot=k%2;
    mbar_wait(bars+2+slot,(k/2)&1);named_bar_sync(1,128);
    uint64_t b0=smem_desc(smem_u32(sm+BUF+slot*16384),16,1024,1);
    uint64_t b1=smem_desc(smem_u32(sm+BUF+slot*16384+8192),16,1024,1);
    fence_regs(v0);fence_regs(v1);wgmma_fence();
    static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
     mma_rs_trans<q*2048>(v0,fa[k*4+q],uint32_t(b0),uint32_t(b0>>32),k>0||q>0);
     mma_rs_trans<q*2048>(v1,fa[k*4+q],uint32_t(b1),uint32_t(b1>>32),k>0||q>0);
    });wgmma_commit();wgmma_wait<0>();fence_regs(v0);fence_regs(v1);named_bar_sync(1,128);
    if(tid==0)mbar_arrive(bars+4+slot);
   });
   static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;
    int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
    reinterpret_cast<bf*>(sm)[dnpos(r,col+c)]=__float2bfloat16_rn(v0[j]);
    reinterpret_cast<bf*>(sm)[dnpos(r,col+64+c)]=__float2bfloat16_rn(v1[j]);
   });
   named_bar_sync(1,128);
  }
  if constexpr(DNS_STATS_TMA)mbar_wait(bars+6+2*NBUF,round&1);
  for(int half=0;half<64/ROWS;++half){int slot=half%NBUF;uint8_t* tri=sm+BUF+slot*LB;
   mbar_wait(bars+6+slot,(half/NBUF)&1);named_bar_sync(1,128);transpose32<false>(tri);
   for(int r=warp;r<ROWS;r+=4){
    float mu=DNS_STATS_TMA?reinterpret_cast<float*>(sm+BARS+128)[half*ROWS+r]:p.mu[row+half*ROWS+r];
    float rs=DNS_STATS_TMA?reinterpret_cast<float*>(sm+BARS+384)[half*ROWS+r]:p.rs[row+half*ROWS+r];
    float z[16],s0=0,s1=0;
    #pragma unroll
    for(int q=0;q<16;++q){int c=lane+q*32;
     z[q]=(__bfloat162float(reinterpret_cast<bf*>(tri)[tripos(r,c)])-mu)*rs;
     float dy=__bfloat162float(reinterpret_cast<bf*>(sm)[dnpos(half*ROWS+r,c)]),v=dy*(DNS_CACHE_GAMMA?gamma[q]:p.gamma[c]);
     s0+=v;s1+=v*z[q];gg[q]+=dy*z[q];bb[q]+=dy;
    }
    s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
    #pragma unroll
    for(int q=0;q<16;++q){int c=lane+q*32;
     float dy=__bfloat162float(reinterpret_cast<bf*>(sm)[dnpos(half*ROWS+r,c)]);
     float centered=fmaf(dy,DNS_CACHE_GAMMA?gamma[q]:p.gamma[c],-s0);
     reinterpret_cast<bf*>(tri)[tripos(r,c)]=__float2bfloat16_rn(fmaf(-z[q],s1,centered)*rs);
    }
   }
   named_bar_sync(1,128);transpose32<true>(tri);fence_proxy_async();named_bar_sync(1,128);
   if(tid==0){for(int c=0;c<H;c+=64)put_tile(&p.dt,tri+(c/64)*(ROWS*128),row+half*ROWS,c);tma_store_commit();if constexpr(DNS_STORE_READ)tma_store_wait_read<0>();else tma_store_wait_all();}
   named_bar_sync(1,128);if(tid==0)mbar_arrive(bars+6+NBUF+slot);
  }
 }
 if(tid==0)tma_store_wait_all();named_bar_sync(1,128);
 float* parts=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<16;++q){int c=lane+q*32;parts[warp*H+c]=gg[q];parts[(4+warp)*H+c]=bb[q];}
 named_bar_sync(1,128);
 for(int c=tid;c<H;c+=128){float g=0,b=0;
  #pragma unroll
  for(int w=0;w<4;++w){g+=parts[w*H+c];b+=parts[(4+w)*H+c];}
  atomicAdd(p.dg+c,g);atomicAdd(p.db+c,b);
 }
}
extern "C" __global__ __launch_bounds__(256,2) void mw_d256_dn_slim(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bars=reinterpret_cast<uint64_t*>(sm+BARS);int tid=threadIdx.x;
 if(tid==0){for(int i=0;i<6+2*NBUF+DNS_STATS_TMA;++i)mbar_init(bars+i,1);fence_barrier_init();}
 for(int c=blockIdx.x*256+tid;c<H;c+=gridDim.x*256){p.dg[c]=0;p.db[c]=0;}
 __syncthreads();cooperative_groups::this_grid().sync();
 if(tid<128){setmaxnreg_dec<32>();producer(p,sm,bars);}else{setmaxnreg_inc<224>();consumer(p,sm,bars);}
}
