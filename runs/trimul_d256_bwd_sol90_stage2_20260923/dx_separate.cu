// SPDX-License-Identifier: Apache-2.0
// D256 dX: N128 CTAs followed by input LN. Preserve the original K order.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
#include "mma.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=256,H=512,STAGE=24576,NSLOT=DX_SEP_SLOTS;
struct Params{CUtensorMap map[16];bf* t[24];float* f[13];int M,L;CUtensorMap dxn;};
TMN_DEVI float rd(const bf* x,size_t i){return __bfloat162float(x[i]);}
TMN_DEVI bf cv(float v){return __float2bfloat16_rn(v);}
TMN_DEVI float wsum(float x){for(int k=16;k;k>>=1)x+=__shfl_xor_sync(0xffffffff,x,k);return x;}
TMN_DEVI void store_tile(const CUtensorMap* map,uint8_t* sm,int c,int row){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(c),"r"(row):"memory");}
TMN_DEVI void load_step(const Params& p,uint8_t* sm,uint64_t* bars,int row,int col,int step){
 int slot=step%NSLOT,plane=(step-4)/8,k=(step<4?step:(step-4)%8)*64;
 uint8_t* dst=sm+slot*STAGE;mbar_arrive_expect_tx(bars+slot,STAGE);
 if(step<4)tma_load_2d(dst,&p.map[5],bars+slot,k,row);
 else tma_load_2d(dst,&p.map[6+plane],bars+slot,row,k);
 for(int c=0;c<2;++c)tma_load_2d(dst+8192*(1+c),&p.map[step<4?2:10+plane],bars+slot,col+c*64,k);
}
TMN_DEVI void produce(const Params& p,uint8_t* sm,uint64_t* bars){
 if(threadIdx.x)return;
 for(int tile=blockIdx.x,round=0;tile<(p.M/64)*2;tile+=gridDim.x,++round){
  int row=(tile/2)*64,col=(tile%2)*128;
  if(round)mbar_wait(bars+2*NSLOT,(round-1)&1);
  for(int step=0;step<36;++step){int slot=step%NSLOT,it=round*36+step;
   if(it>=NSLOT)mbar_wait(bars+NSLOT+slot,((it/NSLOT)-1)&1);
   int plane=(step-4)/8,k=(step<4?step:(step-4)%8)*64;
   uint8_t* dst=sm+slot*STAGE;mbar_arrive_expect_tx(bars+slot,STAGE);
   if(step<4)tma_load_2d(dst,&p.map[5],bars+slot,k,row);
   else tma_load_2d(dst,&p.map[6+plane],bars+slot,row,k);
   for(int c=0;c<2;++c)tma_load_2d(dst+8192*(1+c),&p.map[step<4?2:10+plane],bars+slot,col+c*64,k);
  }
 }
}
TMN_DEVI void consume(const Params& p,uint8_t* sm,uint64_t* bars){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
 for(int tile=blockIdx.x,round=0;tile<(p.M/64)*2;tile+=gridDim.x,++round){
  int row=(tile/2)*64,col=(tile%2)*128;float v[64]={};
  #if !DX_SEP_WS
  if(tid==0)for(int step=0;step<NSLOT;++step)load_step(p,sm,bars,row,col,step);
  #endif
  for(int step=0;step<36;++step){int slot=step%NSLOT,it=round*36+step;uint8_t* buf=sm+slot*STAGE;
   mbar_wait(bars+slot,(it/NSLOT)&1);named_bar_sync(1,128);fence_regs(v);wgmma_fence();
   if(step<4){static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;mma128<0,1>(v,smem_desc(smem_u32(buf+q*32),16,1024,1),smem_desc(smem_u32(buf+8192+q*2048),8192,1024,1),step>0||q>0);});}
   else{static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;mma128<1,1>(v,smem_desc(smem_u32(buf+q*2048),16,1024,1),smem_desc(smem_u32(buf+8192+q*2048),8192,1024,1),1);});}
   wgmma_commit();wgmma_wait<0>();fence_regs(v);named_bar_sync(1,128);if(tid==0)mbar_arrive(bars+NSLOT+slot);
   #if !DX_SEP_WS
   if(tid==0&&step+NSLOT<36)load_step(p,sm,bars,row,col,step+NSLOT);
   #endif
  }
  static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);reinterpret_cast<bf*>(sm+(c/64)*8192)[swz128(r,(c%64)*2)/2]=cv(v[j]);});
  named_bar_sync(1,128);fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){for(int c=0;c<128;c+=64)store_tile(&p.dxn,sm+c*128,col+c,row);tma_store_commit();tma_store_wait_all();}named_bar_sync(1,128);
  if(tid==0)mbar_arrive(bars+2*NSLOT);
 }
}
#if DX_SEP_WS
extern "C" __global__ __launch_bounds__(256,2) void mw_d256_dx_separate(__grid_constant__ const Params p){
#else
extern "C" __global__ __launch_bounds__(128,4) void mw_d256_dx_separate(__grid_constant__ const Params p){
#endif
 extern __shared__ __align__(1024) uint8_t sm[];auto bars=reinterpret_cast<uint64_t*>(sm+NSLOT*STAGE);
 if(threadIdx.x==0){for(int i=0;i<2*NSLOT+1;++i)mbar_init(bars+i,1);fence_barrier_init();}__syncthreads();
 #if DX_SEP_WS
 if(threadIdx.x<128){setmaxnreg_dec<32>();produce(p,sm,bars);}else{setmaxnreg_inc<128>();consume(p,sm,bars);}
 #else
 consume(p,sm,bars);
 #endif
}
extern "C" __global__ __launch_bounds__(128,4) void mw_d256_input_ln_separate(__grid_constant__ const Params p){
 __shared__ float gamma[256],part[2048];int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 for(int c=tid;c<D;c+=128)gamma[c]=p.f[0][c];
 for(int c=blockIdx.x*128+tid;c<D;c+=gridDim.x*128){p.f[8][c]=0;p.f[9][c]=0;}
 __syncthreads();cooperative_groups::this_grid().sync();
 float gg[8]={},bb[8]={};
 for(int row=blockIdx.x*4+warp;row<p.M;row+=gridDim.x*4){
  float mu=p.f[12][2*row],rs=p.f[12][2*row+1],z[8],dy[8],s0=0,s1=0;
  #pragma unroll
  for(int q=0;q<8;++q){int c=lane+32*q;z[q]=(rd(p.t[0],size_t(row)*D+c)-mu)*rs;dy[q]=rd(p.t[10],size_t(row)*D+c);float v=dy[q]*gamma[c];s0+=v;s1+=v*z[q];gg[q]+=dy[q]*z[q];bb[q]+=dy[q];}
  s0=wsum(s0)/D;s1=wsum(s1)/D;
  #pragma unroll
  for(int q=0;q<8;++q){int c=lane+q*32;float v=dy[q]*gamma[c],dx=(v-s0-z[q]*s1)*rs;p.t[11][size_t(row)*D+c]=cv(dx+rd(p.t[2],size_t(row)*D+c));}
 }
 for(int q=0;q<8;++q){int c=lane+q*32;part[warp*D+c]=gg[q];part[(4+warp)*D+c]=bb[q];}__syncthreads();
 for(int c=tid;c<D;c+=128){float g=0,b=0;for(int w=0;w<4;++w){g+=part[w*D+c];b+=part[(4+w)*D+c];}atomicAdd(p.f[8]+c,g);atomicAdd(p.f[9]+c,b);}
 for(int which=0;which<4;++which){int offset=(H+D+which*H)*D;
  for(int i=blockIdx.x*128+tid;i<H*D;i+=gridDim.x*128){float v=0;for(int s=0;s<WEIGHT_SPLITS;++s)v+=p.f[7][size_t(s)*11*D*D+offset+i];p.t[17+which][i]=cv(v);}
 }
}
