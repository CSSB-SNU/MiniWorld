// SPDX-License-Identifier: Apache-2.0
// MiniWorld wide B7: one projection recomputation, two dW consumer warpgroups.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
constexpr int D=WIDTH,H=2*D,RANKS=4*D/32,CH=128*D,INPUT=CH+4096;
constexpr int WEIGHT=2*INPUT,DERIV=WEIGHT+CH,BAR=DERIV+8192,NC=D/128;
struct Params{CUtensorMap xn,w,dl,dr,gmap[4];const bf* mask;float* part;int M;};
// PACKED_GLU
TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int row,int rank,int slot){
 mbar_arrive_expect_tx(bar+slot,INPUT);
 for(int c=0;c<D/64;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);
 tma_load_2d(sm+slot*INPUT+CH,rank<H/32?&p.dl:&p.dr,bar+slot,row,(rank%(H/32))*32);
}
TMN_DEVI void store_gp(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}
extern "C" __global__ __launch_bounds__(256,1)
void mw_wide_b7_source(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x,lane=tid%32,warp=(tid/32)%4,wg=tid/128;
 int rank=blockIdx.x%RANKS,split=blockIdx.x/RANKS,tiles=p.M/64;
 int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
 if(tid==0){
  for(int b=0;b<3;++b)mbar_init(bar+b,1);fence_barrier_init();mbar_arrive_expect_tx(bar+2,CH);
  for(int c=0;c<D/64;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
  load_input(p,sm,bar,begin*64,rank,0);
 }
 __syncthreads();mbar_wait(bar+2,0);__syncthreads();
 float dw[NC][32]={};
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2,row=tile*64;uint8_t* xn=sm+slot*INPUT;
  if(tid==0&&tile+1<end)load_input(p,sm,bar,(tile+1)*64,rank,1-slot);
  mbar_wait(bar+slot,(it/2)&1);__syncthreads();
  if(wg==0){
   float pre[32]={};fence_regs(pre);wgmma_fence();
   static_for<D/16>([&](auto kk){constexpr int k=decltype(kk)::value;
    mma64<0,0>(pre,smem_desc(smem_u32(xn+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+WEIGHT+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(pre);
   int ra=warp*16+lane/4;
   uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[row+ra]))*0x10001u;
   uint32_t mb=uint32_t(__bfloat16_as_ushort(p.mask[row+ra+8]))*0x10001u;
   packed_glu(pre,xn,sm+DERIV,ma,mb);
  }
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){int side=rank/(H/32),col=(rank%(H/32))*32;
   store_gp(p.gmap+2*side,sm+DERIV+4096,row,col);
   store_gp(p.gmap+2*side+1,sm+DERIV,row,col);tma_store_commit();
  }
  static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(dw[c]);});wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;
    mma64<0,1>(dw[c],smem_desc(smem_u32(sm+DERIV+k*32),16,1024,1),
      smem_desc(smem_u32(xn+(wg*NC+c)*8192+k*2048),8192,1024,1),it>0||k>0);
   });
  });wgmma_commit();wgmma_wait<0>();
  static_for<NC>([&](auto cc){constexpr int c=decltype(cc)::value;fence_regs(dw[c]);});
  if(tid==0)tma_store_wait_all();__syncthreads();
 }
 static_for<NC>([&](auto nn){constexpr int n=decltype(nn)::value;
  static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
   int which=(rank/(H/32))*2+(r<32?1:0),outrow=(rank%(H/32))*32+r%32;
   size_t ix=size_t(split)*11*D*D+(3+2*which)*D*D+outrow*D+(wg*NC+n)*64+c;
   p.part[ix]=dw[n][j];
  });
 });
}
