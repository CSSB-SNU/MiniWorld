// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=WIDTH,H=2*D,NT=D,ROWS=16,SB=ROWS*H*2,INPUT=4096+64*D,DN=2*INPUT,BAR=DN+64*H*2;
struct Params{CUtensorMap tri,dt,dp,wp;const float *mu,*rs,*gamma;float* partial;bf* dn;int M;};
// MMA_HELPERS
// TRANSPOSE_HELPERS
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));return v;}
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int row,int col,int ki,int slot){
 mbar_arrive_expect_tx(bar+slot,INPUT);
 tma_load_2d(sm+slot*INPUT,&p.dp,bar+slot,ki,row);
 for(int c=0;c<D;c+=64)tma_load_2d(sm+slot*INPUT+4096+(c/64)*4096,&p.wp,bar+slot,col+c,ki);
}
extern "C" __global__ __launch_bounds__(NT,1)
void mw_wide_stream_dn_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,wg=tid/128,ww=warp%4,row=blockIdx.x*64;
 if(tid==0){for(int i=0;i<3;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 for(int col=0;col<H;col+=D){
  float v[64]={};if(tid==0)load(p,sm,bar,row,col,0,0);
  for(int ki=0,it=0;ki<D;ki+=32,++it){int slot=it%2;
   mbar_wait(bar+slot,(it/2)&1);__syncthreads();
   if(tid==0&&ki+32<D)load(p,sm,bar,row,col,ki+32,1-slot);
   fence_regs(v);wgmma_fence();
   static_for<2>([&](auto qq){constexpr int q=decltype(qq)::value;
    mma128_off<q*32,q*2048,0,1>(v,smem_desc(smem_u32(sm+slot*INPUT),16,512,2),smem_desc(smem_u32(sm+slot*INPUT+4096+wg*8192),4096,1024,1),it>0||q>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
  }
  static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
   int r=ww*16+lane/4+8*((j/2)&1),c=col+wg*128+(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   uint32_t raw=pack_bf16(v[j],v[j+1]);*reinterpret_cast<uint32_t*>(sm+DN+2*(r*H+c))=raw;
   if constexpr(EMIT_DN)*reinterpret_cast<uint32_t*>(p.dn+size_t(row+r)*H+c)=raw;
  });__syncthreads();
 }
 // GEMM accumulators are dead before any LN/affine accumulators become live.
 float gg[H/NT]={},bb[H/NT]={};float* gamma=reinterpret_cast<float*>(sm+SB);
 for(int c=tid;c<H;c+=NT)gamma[c]=p.gamma[c];__syncthreads();
 for(int half=0;half<64/ROWS;++half){
  if(tid==0){mbar_arrive_expect_tx(bar+2,SB);
   for(int c=0;c<H;c+=64)tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar+2,row+half*ROWS,c);
  }
  mbar_wait(bar+2,half&1);__syncthreads();transpose32<false>(sm);
  // Channel ownership keeps just two affine sums per thread.
  #pragma unroll
  for(int q=0;q<H/NT;++q){int c=tid+q*NT;
   #pragma unroll 1
   for(int r=0;r<ROWS;++r){
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-p.mu[row+half*ROWS+r])*p.rs[row+half*ROWS+r];
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm+DN)[(half*ROWS+r)*H+c]);gg[q]+=dy*z;bb[q]+=dy;
   }
  }
  __syncthreads();
  for(int r=warp;r<ROWS;r+=NT/32){
   float mu=p.mu[row+half*ROWS+r],rs=p.rs[row+half*ROWS+r],s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm+DN)[(half*ROWS+r)*H+c]),v=dy*gamma[c];s0+=v;s1+=v*z;
   }
   s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm+DN)[(half*ROWS+r)*H+c]);
    reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,fmaf(dy,gamma[c],-s0))*rs);
   }
  }
  __syncthreads();transpose32<true>(sm);fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<H;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dt),"r"(smem_u32(sm+(c/64)*(ROWS*128))),"r"(row+half*ROWS),"r"(c):"memory");
  }tma_store_commit();tma_store_wait_read<0>();}__syncthreads();
 }
 if(tid==0)tma_store_wait_all();
 #pragma unroll
 for(int q=0;q<H/NT;++q){int c=tid+q*NT;p.partial[size_t(blockIdx.x)*2*H+c]=gg[q];p.partial[size_t(blockIdx.x)*2*H+H+c]=bb[q];}
}
struct Reduce{const float* partial;float* tmp;float *dg,*db;int rows,chunks;};
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_stream_ln_reduce_first(__grid_constant__ const Reduce p){
 __shared__ double sm[512];int tid=threadIdx.x,c=blockIdx.x*128+tid%128,w=tid/128;double g=0,b=0;
 for(int i=blockIdx.y*256+w;i<(blockIdx.y+1)*256&&i<p.rows;i+=2){g+=p.partial[size_t(i)*2*H+c];b+=p.partial[size_t(i)*2*H+H+c];}
 sm[tid]=g;sm[256+tid]=b;__syncthreads();
 if(tid<128){p.tmp[size_t(blockIdx.y)*2*H+c]=float(sm[tid]+sm[128+tid]);p.tmp[size_t(blockIdx.y)*2*H+H+c]=float(sm[256+tid]+sm[384+tid]);}
}
extern "C" __global__ __launch_bounds__(128,4)
void mw_wide_stream_ln_reduce_last(__grid_constant__ const Reduce p){
 int c=blockIdx.x*128+threadIdx.x;double g=0,b=0;
 for(int i=0;i<p.chunks;++i){g+=p.tmp[size_t(i)*2*H+c];b+=p.tmp[size_t(i)*2*H+H+c];}
 p.dg[c]=float(g);p.db[c]=float(b);
}
