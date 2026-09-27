// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=256,H=512,CH=32768,INPUT=40960,WEIGHT=81920,BAR=114688;
struct Params{CUtensorMap xn,w,dl,dr,gmap[4];const bf* mask;float* part;int M;};
// MMA_HELPERS
TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int row,int rank,int slot){
 mbar_arrive_expect_tx(bar+slot,CH+4096+128);
 for(int c=0;c<4;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);
 tma_load_2d(sm+slot*INPUT+CH,rank<16?&p.dl:&p.dr,bar+slot,row,(rank%16)*32);
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],128,[%2];"::"r"(smem_u32(sm+BAR+128+slot*128)),"l"(p.mask+row),"r"(smem_u32(bar+slot)):"memory");
}
TMN_DEVI void store_gp(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}
TMN_DEVI void produce(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x,lane=tid%32,w=tid/32,mat=lane/8;
 int rank=blockIdx.x%32,split=blockIdx.x/32,tiles=p.M/64;
 int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
 if(tid==0){mbar_arrive_expect_tx(bar+2,CH);
  for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
  load_input(p,sm,bar,begin*64,rank,0);
 }
 mbar_wait(bar+2,0);named_bar_sync(1,128);
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2,row=tile*64;uint8_t* xn=sm+slot*INPUT;uint8_t* dg=xn+CH;
  mbar_wait(bar+slot,(it/2)&1);named_bar_sync(1,128);
  float pre[32]={};fence_regs(pre);wgmma_fence();
  static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma64_off<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32,0,0>(pre,smem_desc(smem_u32(xn),16,1024,1),smem_desc(smem_u32(sm+WEIGHT),16,1024,1),k>0);
  });wgmma_commit();
  if(tile+1<end){
   if(it>=1)mbar_wait(bar+5+(1-slot),((it-1)/2)&1);
   if(tid==0)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);
  }
  wgmma_wait<0>();fence_regs(pre);
  int ra=w*16+lane/4;auto mask=reinterpret_cast<bf*>(sm+BAR+128+slot*128);
  uint32_t ma=uint32_t(__bfloat16_as_ushort(mask[ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(mask[ra+8]))*0x10001u;
  uint32_t incoming[2][4];
  #pragma unroll
  for(int q=0;q<2;++q)ldsm_x4_t(incoming[q],smem_u32(dg)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
  // dL/dR is dead after these loads; its storage becomes dGate.
  named_bar_sync(1,128);
  #pragma unroll
  for(int q=0;q<2;++q){uint32_t dg0[4],dp[4];
   #pragma unroll
   for(int j=0;j<4;++j){uint32_t masked,m=(j&1)?mb:ma;
    asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(incoming[q][j]),"r"(m));
    uint32_t gr=pack_bf16(pre[q*8+j*2],pre[q*8+j*2+1]),pr=pack_bf16(pre[(q+2)*8+j*2],pre[(q+2)*8+j*2+1]);
    float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
    dg0[j]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));dp[j]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
   }
   uint32_t off=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);
   stsm_x4_t(smem_u32(dg)+off,dg0[0],dg0[1],dg0[2],dg0[3]);stsm_x4_t(smem_u32(dg+4096)+off,dp[0],dp[1],dp[2],dp[3]);
  }
  named_bar_sync(1,128);fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){store_gp(p.gmap+2*(rank/16),dg+4096,row,(rank%16)*32);store_gp(p.gmap+2*(rank/16)+1,dg,row,(rank%16)*32);
   tma_store_commit();mbar_arrive(bar+3+slot);tma_store_wait_all();
  }named_bar_sync(1,128);
 }
}
TMN_DEVI void consume(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
 int rank=blockIdx.x%32,split=blockIdx.x/32,tiles=p.M/64;
 int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
 float dw0[64]={},dw1[64]={};
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2;uint8_t* xn=sm+slot*INPUT;uint8_t* dg=xn+CH;
  mbar_wait(bar+3+slot,(it/2)&1);named_bar_sync(2,128);
  fence_regs(dw0);fence_regs(dw1);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma128_off<k*32,k*2048,0,1>(dw0,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);
   mma128_off<k*32,k*2048,0,1>(dw1,smem_desc(smem_u32(dg),16,1024,1),smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);
  named_bar_sync(2,128);if(tid==0)mbar_arrive(bar+5+slot);
 }
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
  int which=(rank/16)*2+(r<32?1:0),outrow=(rank%16)*32+r%32;
  size_t ix=size_t(split)*11*D*D+(3+2*which)*D*D+outrow*D+c;
  p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];
 });
}
extern "C" __global__ __launch_bounds__(256,2)
void mw_d256_aliased_pipe_source(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 if(threadIdx.x==0){for(int i=0;i<7;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 if(threadIdx.x<128){setmaxnreg_dec<PRODUCER_REGS>();produce(p,sm,bar);}
 else{setmaxnreg_inc<256-PRODUCER_REGS>();consume(p,sm,bar);}
}
