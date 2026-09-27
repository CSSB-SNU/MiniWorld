// SPDX-License-Identifier: Apache-2.0
// Ordered dX GEMM and input LN share storage; two consumers cover all channels.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
constexpr int D=WIDTH,H=2*D,NC=D/128,STAGE=(D/64+1)*8192,SB=D*128;
constexpr int STEPS=9*D/64,SHARED=3*SB>2*STAGE?3*SB:2*STAGE;
struct Params {CUtensorMap map[16];bf* t[24];float* f[13];int M,L;CUtensorMap x,res,dx;};
TMN_DEVI int pos(int r,int c){return (c/64)*4096+swz128(r,(c%64)*2)/2;}
TMN_DEVI float rd(uint8_t* sm,int r,int c){return __bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)]);}
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
TMN_DEVI void produce(const Params& p,uint8_t* sm,uint64_t* bar){
 if(threadIdx.x)return;
 for(int row=blockIdx.x*64,round=0;row<p.M;row+=gridDim.x*64,++round){
  if(round)mbar_wait(bar+5,(round-1)&1);
  for(int step=0;step<STEPS;++step){
   int it=round*STEPS+step,slot=it%2;
   if(it>=2)mbar_wait(bar+3+slot,((it/2)-1)&1);
   int plane=(step-D/64)/(H/64),k=(step<D/64?step:(step-D/64)%(H/64))*64;
   uint8_t* dst=sm+slot*STAGE;mbar_arrive_expect_tx(bar+slot,STAGE);
   if(step<D/64)tma_load_2d(dst,&p.map[5],bar+slot,k,row);
   else tma_load_2d(dst,&p.map[6+plane],bar+slot,row,k);
   for(int c=0;c<D/64;++c)tma_load_2d(dst+8192*(1+c),&p.map[step<D/64?2:10+plane],bar+slot,c*64,k);
  }
 }
}
template<int WG> TMN_DEVI void consume(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,ct=WG*128+tid,cwarp=ct/32;
 float gg[D/32]={},bb[D/32]={};
 for(int row=blockIdx.x*64,round=0;row<p.M;row+=gridDim.x*64,++round){
  float acc[NC][32]={};
  for(int step=0;step<STEPS;++step){
   int it=round*STEPS+step,slot=it%2;uint8_t* buf=sm+slot*STAGE;
   mbar_wait(bar+slot,(it/2)&1);named_bar_sync(2+WG,128);
   static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(acc[c]);});wgmma_fence();
   if(step<D/64){
    static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
     static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;
      mma64<0,1>(acc[c],smem_desc(smem_u32(buf+q*32),16,1024,1),smem_desc(smem_u32(buf+8192*(1+WG*NC+c)+q*2048),8192,1024,1),step>0||q>0);
     });
    });
   }else{
    static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
     static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;
      mma64<1,1>(acc[c],smem_desc(smem_u32(buf+q*2048),16,1024,1),smem_desc(smem_u32(buf+8192*(1+WG*NC+c)+q*2048),8192,1024,1),1);
     });
    });
   }
   wgmma_commit();wgmma_wait<0>();
   static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(acc[c]);});
   named_bar_sync(2+WG,128);if(tid==0)mbar_arrive(bar+3+slot);
  }
  // All input stages must be consumed before shared buffers become epilogue data.
  named_bar_sync(1,256);
  static_for<NC>([&](auto cc){constexpr int col=decltype(cc)::value;
   static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;
    int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
    reinterpret_cast<bf*>(sm)[pos(r,(WG*NC+col)*64+c)]=__float2bfloat16_rn(acc[col][j]);
   });
  });
  named_bar_sync(1,256);
  if(ct==0){mbar_arrive_expect_tx(bar+2,2*SB);for(int c=0;c<D;c+=64){tma_load_2d(sm+SB+c*128,&p.x,bar+2,c,row);tma_load_2d(sm+2*SB+c*128,&p.res,bar+2,c,row);}}
  mbar_wait(bar+2,round&1);named_bar_sync(1,256);
  for(int r=cwarp;r<64;r+=8){
   float xv[D/32],s=0;
   #pragma unroll
   for(int q=0;q<D/32;++q){xv[q]=rd(sm+SB,r,lane+q*32);s+=xv[q];}
   float mu=sumwarp(s)/D;s=0;
   #pragma unroll
   for(int q=0;q<D/32;++q){float z=xv[q]-mu;s+=z*z;}
   float rs=rsqrtf(sumwarp(s)/D+1e-5f),s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<D/32;++q){int c=lane+q*32;float z=(xv[q]-mu)*rs,dy=rd(sm,r,c),v=dy*p.f[0][c];s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;}
   s0=sumwarp(s0)/D;s1=sumwarp(s1)/D;
   #pragma unroll
   for(int q=0;q<D/32;++q){int c=lane+q*32;float z=(xv[q]-mu)*rs,v=rd(sm,r,c)*p.f[0][c];
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn((v-s0-z*s1)*rs+rd(sm+2*SB,r,c));
   }
  }
  named_bar_sync(1,256);fence_proxy_async();named_bar_sync(1,256);
  if(ct==0){for(int c=0;c<D;c+=64){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dx),"r"(smem_u32(sm+c*128)),"r"(c),"r"(row):"memory");}tma_store_commit();tma_store_wait_all();}
  named_bar_sync(1,256);if(ct==0)mbar_arrive(bar+5);
 }
 float* part=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<D/32;++q){int c=lane+q*32;part[cwarp*D+c]=gg[q];part[(8+cwarp)*D+c]=bb[q];}
 named_bar_sync(1,256);
 for(int c=ct;c<D;c+=256){float g=0,b=0;for(int w=0;w<8;++w){g+=part[w*D+c];b+=part[(8+w)*D+c];}atomicAdd(p.f[8]+c,g);atomicAdd(p.f[9]+c,b);}
 for(int which=0;which<4;++which){
  for(int i=blockIdx.x*256+ct;i<H*D;i+=gridDim.x*256){float v=0;
   #pragma unroll
   for(int s=0;s<32;++s)v+=p.f[7][size_t(s)*11*D*D+(3+2*which)*D*D+i];
   p.t[17+which][i]=__float2bfloat16_rn(v);
  }
 }
}
extern "C" __global__ __launch_bounds__(384,1)
void mw_wide_fused_dx(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+SHARED);
 int tid=threadIdx.x;
 if(tid==0){for(int i=0;i<6;++i)mbar_init(bar+i,(i==3||i==4)?2:1);fence_barrier_init();}
 for(int c=blockIdx.x*384+tid;c<D;c+=gridDim.x*384){p.f[8][c]=0;p.f[9][c]=0;}
 __syncthreads();cooperative_groups::this_grid().sync();
 if(tid<128){setmaxnreg_dec<32>();produce(p,sm,bar);}
 else{setmaxnreg_inc<232>();if(tid<256)consume<0>(p,sm,bar);else consume<1>(p,sm,bar);}
}
