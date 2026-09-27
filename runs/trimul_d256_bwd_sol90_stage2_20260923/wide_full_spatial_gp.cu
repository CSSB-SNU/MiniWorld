// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
constexpr int D=WIDTH,H=2*D,N=384,NT=128*ROW_GROUPS;
constexpr int INPUT=(ROW_GROUPS+6)*8192,SLOTS=3,BAR=INPUT*SLOTS,EPI=ROW_GROUPS*49152;
struct Params{CUtensorMap a[4],b[4],pre,out[4];const bf* mask;};
template<int MODE> TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int step){
 constexpr int TA=MODE==1,TB=MODE!=2;int slot=step%3,ki=step*64;
 mbar_arrive_expect_tx(bar+slot,INPUT);
 tma_load_3d(sm+slot*INPUT,p.a+MODE,bar+slot,TA?0:ki,TA?ch*N+ki:0,TA?mi/64:(ch*N+mi)/64);
 tma_load_3d(sm+slot*INPUT+ROW_GROUPS*8192,p.b+MODE,bar+slot,TB?0:ki,TB?ch*N+ki:0,TB?0:ch*N/64);
}
template<int MODE,int PLANE> TMN_DEVI void load_epi(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi){
 constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
 if constexpr(PLANE==0)mbar_arrive_expect_tx(bar+3,2*EPI);
 #pragma unroll
 for(int wm=0;wm<ROW_GROUPS;++wm)
  tma_load_3d(sm+PLANE*EPI+wm*49152,&p.pre,bar+3,0,(rank*64+pc+PLANE*32)*N+mi+wm*64,0);
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi){
 constexpr int TA=MODE==1,TB=MODE!=2,SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,wg=threadIdx.x/128;
 if(threadIdx.x==0){load<MODE>(p,sm,bar,ch,mi,0);load<MODE>(p,sm,bar,ch,mi,1);}
 float v0[128]={},v1[64]={};
 #pragma unroll
 for(int step=0;step<6;++step){
  int slot=step%3;mbar_wait(bar+slot,(step/3)&1);__syncthreads();
  if(threadIdx.x==0){
   if(step+2<6)load<MODE>(p,sm,bar,ch,mi,step+2);
   if constexpr(ROW_GROUPS==1){
    if(step==4)load_epi<MODE,0>(p,sm,bar,ch,mi);
    if(step==5)load_epi<MODE,1>(p,sm,bar,ch,mi);
   }else if(step==5)load_epi<MODE,0>(p,sm,bar,ch,mi);
  }
  fence_regs(v0);fence_regs(v1);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma256_off<k*(TA?2048:32),k*(TB?2048:32),TA,TB>(v0,smem_desc(smem_u32(sm+slot*INPUT+wg*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+ROW_GROUPS*8192),TB?8192:16,1024,1),step>0||k>0);
   mma128_off<k*(TA?2048:32),32768+k*(TB?2048:32),TA,TB>(v1,smem_desc(smem_u32(sm+slot*INPUT+wg*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+ROW_GROUPS*8192),TB?8192:16,1024,1),step>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v0);fence_regs(v1);
 }
 __syncthreads();
 if constexpr(ROW_GROUPS==2)if(threadIdx.x==0)load_epi<MODE,1>(p,sm,bar,ch,mi);
 mbar_wait(bar+3,0);__syncthreads();
 // EPILOGUE
 fence_proxy_async();__syncthreads();
 if(threadIdx.x==0){
  #pragma unroll
  for(int plane=0;plane<2;++plane)for(int wm=0;wm<ROW_GROUPS;++wm){
   asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{0,%2,0}],[%1];"::"l"(p.out+2*SIDE+plane),"r"(smem_u32(sm+plane*EPI+wm*49152)),"r"((ch+HALF*D)*N+mi+wm*64):"memory");
  }
  tma_store_commit();tma_store_wait_all();
 }
}
extern "C" __global__ __launch_bounds__(NT,1)
void mw_wide_full_spatial_gp(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 constexpr int TILES=N/(64*ROW_GROUPS);
 int half=blockIdx.x/(2*D*TILES),rem=blockIdx.x%(2*D*TILES);
 int ch=rem/(2*TILES),mode=2*half+rem%2,mi=((rem/2)%TILES)*(64*ROW_GROUPS);
 if(threadIdx.x==0){for(int i=0;i<4;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 if(mode==0)run<0>(p,sm,bar,ch,mi);
 else if(mode==1)run<1>(p,sm,bar,ch,mi);
 else if(mode==2)run<2>(p,sm,bar,ch,mi);
 else run<3>(p,sm,bar,ch,mi);
}
