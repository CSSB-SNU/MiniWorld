// SPDX-License-Identifier: Apache-2.0
// Two row owners exchange GP and own disjoint halves of the input dW columns.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
// PACKED_HELPER
constexpr int D=256,H=512,ONE=36864,INPUT=2*ONE,WEIGHT=2*INPUT,DERIV=WEIGHT+32768,BARS=DERIV+16384;
struct Params{CUtensorMap xn,w,dl,dr,gmap[4];const bf* mask;float* part;int M;};
TMN_DEVI void store_gp(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}
TMN_DEVI void produce(const Params& p,uint8_t* sm,uint64_t* bar){
 if(threadIdx.x)return;
 int rank=blockIdx.x%32,split=blockIdx.x/32,tiles=p.M/128;
 int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
 mbar_arrive_expect_tx(bar+2,32768);
 for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2;
  if(it>=2)mbar_wait(bar+3+slot,((it/2)-1)&1);
  mbar_arrive_expect_tx(bar+slot,INPUT+256);
  for(int wg=0;wg<2;++wg){int row=tile*128+wg*64;uint8_t* dst=sm+slot*INPUT+wg*ONE;
   for(int c=0;c<4;++c)tma_load_2d(dst+c*8192,&p.xn,bar+slot,c*64,row);
   tma_load_2d(dst+32768,rank<16?&p.dl:&p.dr,bar+slot,row,(rank%16)*32);
  }
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],256,[%2];"::"r"(smem_u32(sm+BARS+128+slot*256)),"l"(p.mask+tile*128),"r"(smem_u32(bar+slot)):"memory");
 }
}
template<int WG> TMN_DEVI void consume(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,rank=blockIdx.x%32,split=blockIdx.x/32;
 int tiles=p.M/128,begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
 mbar_wait(bar+2,0);named_bar_sync(1,256);
 float dw[64]={};
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2,row=tile*128+WG*64;uint8_t* xn=sm+slot*INPUT+WG*ONE;
  mbar_wait(bar+slot,(it/2)&1);named_bar_sync(1,256);
  float pre[32]={};fence_regs(pre);wgmma_fence();
  static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma64_off<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32,0,0>(pre,smem_desc(smem_u32(xn),16,1024,1),smem_desc(smem_u32(sm+WEIGHT),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(pre);
  int ra=warp*16+lane/4;bf* mask=reinterpret_cast<bf*>(sm+BARS+128+slot*256)+WG*64;
  uint32_t ma=uint32_t(__bfloat16_as_ushort(mask[ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(mask[ra+8]))*0x10001u;
  packed_glu(pre,xn,sm+DERIV+WG*8192,ma,mb);
  fence_proxy_async();named_bar_sync(1,256);
  if(tid==0){
   store_gp(p.gmap+(rank/16)*2,sm+DERIV+WG*8192+4096,row,(rank%16)*32);
   store_gp(p.gmap+(rank/16)*2+1,sm+DERIV+WG*8192,row,(rank%16)*32);tma_store_commit();
  }
  fence_regs(dw);wgmma_fence();
  static_for<2>([&](auto rr){constexpr int src=decltype(rr)::value;
   static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
    mma128_off<k*32,k*2048,0,1>(dw,smem_desc(smem_u32(sm+DERIV+src*8192),16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+src*ONE+WG*16384),8192,1024,1),it>0||src>0||k>0);
   });
  });wgmma_commit();wgmma_wait<0>();fence_regs(dw);
  if(tid==0)tma_store_wait_read<0>();named_bar_sync(1,256);
  if(threadIdx.x==128)mbar_arrive(bar+3+slot);
 }
 if(tid==0)tma_store_wait_all();named_bar_sync(1,256);
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
  int which=(rank/16)*2+(r<32?1:0),outrow=(rank%16)*32+r%32;
  size_t ix=size_t(split)*11*D*D+(3+2*which)*D*D+outrow*D+c+WG*128;
  p.part[ix]=dw[j];
 });
}
extern "C" __global__ __launch_bounds__(384,1) void mw_d256_joint_rows_source(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BARS);
 if(threadIdx.x==0){for(int i=0;i<5;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();
 if(threadIdx.x<128){setmaxnreg_dec<32>();produce(p,sm,bar);}
 else{setmaxnreg_inc<CONSUMER_REGS>();if(threadIdx.x<256)consume<0>(p,sm,bar);else consume<1>(p,sm,bar);}
}
