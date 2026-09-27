// SPDX-License-Identifier: Apache-2.0
// Two adjacent output-rank consumers share one normalized-input TMA tile.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
// PACKED_HELPER
constexpr int D=256,H=512,INPUT=40960,WEIGHT=81920,DERIV=147456,BARS=163840;
struct Params {CUtensorMap xn,w,dl,dr,gmap[4];const bf* mask;float* part;int M;};
TMN_DEVI void store_gp(const CUtensorMap* map,uint8_t* sm,int row,int c){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");}
TMN_DEVI void producer(const Params& p,uint8_t* sm,uint64_t* b){
 if(threadIdx.x)return;int rank=(blockIdx.x%16)*2,split=blockIdx.x/16,begin=(p.M/64)*split/WEIGHT_SPLITS,end=(p.M/64)*(split+1)/WEIGHT_SPLITS;
 mbar_arrive_expect_tx(b+2,65536);for(int g=0;g<2;++g)for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+g*32768+c*8192,&p.w,b+2,c*64,(rank+g)*64);
 for(int tile=begin,it=0;tile<end;++tile,++it){int slot=it%2;if(it>=2)mbar_wait(b+3+slot,((it/2)-1)&1);mbar_arrive_expect_tx(b+slot,INPUT);
  for(int c=0;c<4;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,b+slot,c*64,tile*64);
  for(int g=0;g<2;++g)tma_load_2d(sm+slot*INPUT+32768+g*4096,rank<16?&p.dl:&p.dr,b+slot,tile*64,((rank+g)%16)*32);
 }
}
template<int WG> TMN_DEVI void consumer(const Params& p,uint8_t* sm,uint64_t* b){
 constexpr int SYNC=WG+1;int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,rank=(blockIdx.x%16)*2+WG,split=blockIdx.x/16;
 int begin=(p.M/64)*split/WEIGHT_SPLITS,end=(p.M/64)*(split+1)/WEIGHT_SPLITS;
 mbar_wait(b+2,0);named_bar_sync(SYNC,128);float dw0[64]={},dw1[64]={};uint8_t* gp=sm+DERIV+WG*8192;
 for(int tile=begin,it=0;tile<end;++tile,++it){int slot=it%2,row=tile*64;uint8_t* xn=sm+slot*INPUT;
  mbar_wait(b+slot,(it/2)&1);named_bar_sync(SYNC,128);float pre[32]={};fence_regs(pre);wgmma_fence();
  static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;mma64_off<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32,0,0>(pre,smem_desc(smem_u32(xn),16,1024,1),smem_desc(smem_u32(sm+WEIGHT+WG*32768),16,1024,1),k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(pre);
  int ra=warp*16+lane/4;uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[row+ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(p.mask[row+ra+8]))*0x10001u;
  packed_cluster_glu(pre,xn+32768+WG*4096,gp,ma,mb);named_bar_sync(SYNC,128);fence_proxy_async();named_bar_sync(SYNC,128);
  if(tid==0){store_gp(p.gmap+(rank/16)*2,gp+4096,row,(rank%16)*32);store_gp(p.gmap+(rank/16)*2+1,gp,row,(rank%16)*32);tma_store_commit();}
  fence_regs(dw0);fence_regs(dw1);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;uint64_t a=smem_desc(smem_u32(gp),16,1024,1);mma128_off<k*32,k*2048,0,1>(dw0,a,smem_desc(smem_u32(xn),8192,1024,1),it>0||k>0);mma128_off<k*32,k*2048,0,1>(dw1,a,smem_desc(smem_u32(xn+16384),8192,1024,1),it>0||k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);if(tid==0)tma_store_wait_all();named_bar_sync(SYNC,128);if(tid==0)mbar_arrive(b+3+slot);
 }
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);int which=(rank/16)*2+(r<32?1:0),outrow=(rank%16)*32+r%32;size_t ix=size_t(split)*11*D*D+(3+2*which)*D*D+outrow*D+c;p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];});
}
extern "C" __global__ __launch_bounds__(384,1) void mw_d256_b7_pairs(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto b=reinterpret_cast<uint64_t*>(sm+BARS);int tid=threadIdx.x;
 if(tid==0){for(int i=0;i<5;++i)mbar_init(b+i,i>=3?2:1);fence_barrier_init();}__syncthreads();
 if(tid<128){setmaxnreg_dec<32>();producer(p,sm,b);}
 else{setmaxnreg_inc<224>();if(tid<256)consumer<0>(p,sm,b);else consumer<1>(p,sm,b);}
}
