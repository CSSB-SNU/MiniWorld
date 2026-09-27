// SPDX-License-Identifier: Apache-2.0
// Three-CTA candidate: two M64N96 warpgroups and two K64 input slots.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA96_HELPER
constexpr int D=WIDTH,H=2*D,N=FIXED_LENGTH,INPUT=28672,PLANE=24576,BAR=3*PLANE;
struct Params {CUtensorMap a[4],b[4];const bf* pre;const bf* mask;bf* gp[4];bf* dl;bf* dr;CUtensorMap premap,maskmap,outmap[4];int unusedN;};
template<int MODE> TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni,int ki,int slot){
 constexpr int TA=MODE==1,TB=MODE!=2;
 mbar_arrive_expect_tx(bar+slot,INPUT);
 for(int g=0;g<2;++g){int m=mi+64*g;
  tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*N+(TA?ki:m));
 }
 for(int g=0;g<3;++g){int n=ni+32*g;
  tma_load_2d(sm+slot*INPUT+16384+g*4096,p.b+MODE,bar+slot,TB?n:ki,ch*N+(TB?ki:n));
 }
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int TA=MODE==1,TB=MODE!=2;
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,WG=threadIdx.x/128;
 float v[48]={};
 if(threadIdx.x==0)load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);
 #pragma unroll 1
 for(int ki=0,it=0;ki<N;ki+=64,++it){
  int slot=it%2;mbar_wait(bar+slot,(it/2)&1);__syncthreads();
  if(threadIdx.x==0&&ki+64<N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+64,slot^1);
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma96_off<k*(TA?2048:32),k*(TB?1024:32),TA,TB>(v,
    smem_desc(smem_u32(sm+slot*INPUT+WG*8192),TA?8192:16,1024,1),
    smem_desc(smem_u32(sm+slot*INPUT+16384),TB?4096:16,TB?512:1024,TB?2:1),it>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);
 }
 __syncthreads();
 if(threadIdx.x==0){
  constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
  int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
  mbar_arrive_expect_tx(bar+2,3*PLANE);
  for(int g=0;g<3;++g){
   tma_load_2d(sm+g*8192,&p.premap,bar+2,ni+g*32,(rank*64+pc)*N+mi);
   tma_load_2d(sm+PLANE+g*8192,&p.premap,bar+2,ni+g*32,(rank*64+pc+32)*N+mi);
   tma_load_2d(sm+2*PLANE+g*8192,&p.maskmap,bar+2,ni+g*32,mi);
  }
 }
 mbar_wait(bar+2,0);__syncthreads();
 static_for<24>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  uint32_t off=(c/32)*8192+sw64((WG*64+r)*64+(c%32)*2);
  uint32_t raw=pack_bf16(v[j],v[j+1]),masked,mask=*reinterpret_cast<uint32_t*>(sm+2*PLANE+off);
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
  uint32_t gr=*reinterpret_cast<uint32_t*>(sm+off),pr=*reinterpret_cast<uint32_t*>(sm+PLANE+off);
  float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
  uint32_t dp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  uint32_t dg=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
  *reinterpret_cast<uint32_t*>(sm+off)=dp;*reinterpret_cast<uint32_t*>(sm+PLANE+off)=dg;
 });
 fence_proxy_async();__syncthreads();
 if(threadIdx.x==0){
  constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);int outch=ch+HALF*D;
  for(int g=0;g<2;++g)for(int tile=0;tile<3;++tile){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(p.outmap+2*SIDE+g),"r"(smem_u32(sm+g*PLANE+tile*8192)),"r"(ni+tile*32),"r"(outch*N+mi):"memory");
  }
  tma_store_commit();tma_store_wait_all();
 }
}
extern "C" __global__ __launch_bounds__(256,3)
void mw_wide_n96_contract_gp(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 constexpr int TM=N/128,TN=N/96,TS=TM*TN;
 int half=blockIdx.x/(2*D*TS),rem=blockIdx.x%(2*D*TS),ch=rem/(2*TS),mode=2*half+rem%2,tile=(rem/2)%TS;
 int mi=(tile/TN)*128,ni=(tile%TN)*96;
 if(threadIdx.x==0){for(int i=0;i<3;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 if(mode==0)run<0>(p,sm,bar,ch,mi,ni);
 else if(mode==1)run<1>(p,sm,bar,ch,mi,ni);
 else if(mode==2)run<2>(p,sm,bar,ch,mi,ni);
 else run<3>(p,sm,bar,ch,mi,ni);
}
