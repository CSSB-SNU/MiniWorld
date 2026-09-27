// SPDX-License-Identifier: Apache-2.0
// Pure four-contraction kernel, fixed K, ordered WGMMA, no GP epilogue.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPER
constexpr int D=256,N=384,NT=128*ROW_GROUPS,INPUT=(ROW_GROUPS+2)*8192,SLOTS=3,BAR=INPUT*SLOTS;
struct Params{CUtensorMap a[4],b[4],out[2];};
template<int MODE> TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni,int step){
 constexpr int TA=MODE==1,TB=MODE!=2;int slot=step%3,ki=step*64;
 mbar_arrive_expect_tx(bar+slot,INPUT);
 #pragma unroll
 for(int g=0;g<ROW_GROUPS;++g){int m=mi+g*64;
  tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*N+(TA?ki:m));
 }
 #pragma unroll
 for(int g=0;g<2;++g){int n=ni+g*64;
  tma_load_2d(sm+slot*INPUT+(ROW_GROUPS+g)*8192,p.b+MODE,bar+slot,TB?n:ki,ch*N+(TB?ki:n));
 }
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int TA=MODE==1,TB=MODE!=2,SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,wg=threadIdx.x/128;
 if(threadIdx.x==0){load<MODE>(p,sm,bar,ch,mi,ni,0);load<MODE>(p,sm,bar,ch,mi,ni,1);}
 float v[64]={};
 #pragma unroll
 for(int step=0;step<6;++step){
  int slot=step%3;mbar_wait(bar+slot,(step/3)&1);__syncthreads();
  if(threadIdx.x==0 && step+2<6)load<MODE>(p,sm,bar,ch,mi,ni,step+2);
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma128_off<k*(TA?2048:32),k*(TB?2048:32),TA,TB>(v,smem_desc(smem_u32(sm+slot*INPUT+wg*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+ROW_GROUPS*8192),TB?8192:16,1024,1),step>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);
 }
 __syncthreads();
 static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  *reinterpret_cast<uint32_t*>(sm+(wg*2+c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(v[j],v[j+1]);
 });
 fence_proxy_async();__syncthreads();
 if(threadIdx.x==0){
  #pragma unroll
  for(int wm=0;wm<ROW_GROUPS;++wm){
   #pragma unroll
   for(int wn=0;wn<2;++wn){
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(p.out+SIDE),"r"(smem_u32(sm+(wm*2+wn)*8192)),"r"(ni+wn*64),"r"((ch+HALF*D)*N+mi+wm*64):"memory");
   }
  }
  tma_store_commit();tma_store_wait_all();
 }
}
extern "C" __global__ __launch_bounds__(NT,MIN_BLOCKS)
void mw_d256_fixed_contract(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 constexpr int MT=N/(64*ROW_GROUPS),TILES=MT*3;
 int mode,ch,tile;
 #if GRID_ORDER==0
 mode=blockIdx.x/(D*TILES);ch=(blockIdx.x/TILES)%D;tile=blockIdx.x%TILES;
 #elif GRID_ORDER==1
 int half=blockIdx.x/(2*D*TILES),rem=blockIdx.x%(2*D*TILES);
 ch=rem/(2*TILES);mode=2*half+rem%2;tile=(rem/2)%TILES;
 #else
 mode=blockIdx.x/(D*TILES);int rem=blockIdx.x%(D*TILES);
 tile=(rem/8)%TILES;ch=rem%8+8*(rem/(8*TILES));
 #endif
 int mi=(tile/3)*(64*ROW_GROUPS),ni=(tile%3)*128;
 if(threadIdx.x==0){for(int i=0;i<3;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 if(mode==0)run<0>(p,sm,bar,ch,mi,ni);
 else if(mode==1)run<1>(p,sm,bar,ch,mi,ni);
 else if(mode==2)run<2>(p,sm,bar,ch,mi,ni);
 else run<3>(p,sm,bar,ch,mi,ni);
}
