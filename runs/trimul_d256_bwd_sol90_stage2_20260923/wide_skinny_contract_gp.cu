// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
constexpr int D=WIDTH,H=2*D,INPUT=24576,SLOTS=INPUT_SLOTS,BAR=INPUT*SLOTS;
struct Params {CUtensorMap a[4],b[4];const bf* pre;const bf* mask;bf* gp[4];bf* dl;bf* dr;CUtensorMap premap,maskmap,outmap[4];int N;};
template<int MODE> TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni,int ki,int slot){
 constexpr int TA=MODE==1,TB=MODE!=2;
 mbar_arrive_expect_tx(bar+slot,INPUT);
 for(int g=0;g<2;++g){int m=mi+64*g;
  tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*p.N+(TA?ki:m));
 }
 tma_load_2d(sm+slot*INPUT+16384,p.b+MODE,bar+slot,TB?ni:ki,ch*p.N+(TB?ki:ni));
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int TA=MODE==1,TB=MODE!=2,SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int WG=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
 float v[32]={};
 if(threadIdx.x==0){
  load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);
  if constexpr(SLOTS==3)load_input<MODE>(p,sm,bar,ch,mi,ni,64,1);
 }
 for(int ki=0,it=0;ki<p.N;ki+=64,++it){
  int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);__syncthreads();
  if(threadIdx.x==0 && ki+(SLOTS-1)*64<p.N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+(SLOTS-1)*64,(it+SLOTS-1)%SLOTS);
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma64_off<q*(TA?2048:32),q*(TB?2048:32),TA,TB>(v,smem_desc(smem_u32(sm+slot*INPUT+WG*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+16384),TB?8192:16,1024,1),it>0||q>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);
 }
 __syncthreads();
 int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
 if(threadIdx.x==0){
  mbar_arrive_expect_tx(bar+SLOTS,49152);
  for(int wm=0;wm<2;++wm){int q=wm*8192;
   tma_load_2d(sm+q,&p.premap,bar+SLOTS,ni,(rank*64+pc)*p.N+mi+wm*64);
   tma_load_2d(sm+16384+q,&p.premap,bar+SLOTS,ni,(rank*64+pc+32)*p.N+mi+wm*64);
   tma_load_2d(sm+32768+q,&p.maskmap,bar+SLOTS,ni,mi+wm*64);
  }
 }
 mbar_wait(bar+SLOTS,0);__syncthreads();
 static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  uint32_t off=WG*8192+swz128(r,c*2),raw=pack_bf16(v[j],v[j+1]),masked,mask=*reinterpret_cast<uint32_t*>(sm+32768+off);
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
  uint32_t gr=*reinterpret_cast<uint32_t*>(sm+off),pr=*reinterpret_cast<uint32_t*>(sm+16384+off);
  float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
  *reinterpret_cast<uint32_t*>(sm+off)=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  *reinterpret_cast<uint32_t*>(sm+16384+off)=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
 });fence_proxy_async();__syncthreads();
 if(threadIdx.x==0){
  for(int g=0;g<2;++g)for(int wm=0;wm<2;++wm){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(p.outmap+2*SIDE+g),"r"(smem_u32(sm+g*16384+wm*8192)),"r"(ni),"r"(outch*p.N+mi+wm*64):"memory");
  }tma_store_commit();tma_store_wait_all();
 }
}
extern "C" __global__ __launch_bounds__(256,MIN_BLOCKS)
void mw_wide_skinny_contract_gp(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tiles=(p.N/128)*(p.N/64),mode=blockIdx.x/(D*tiles),rem=blockIdx.x%(D*tiles);
 int tile=(rem/CHANNEL_GROUP)%tiles,ch=(rem%CHANNEL_GROUP)+CHANNEL_GROUP*(rem/(CHANNEL_GROUP*tiles));
 int mi=(tile/(p.N/64))*128,ni=(tile%(p.N/64))*64;
 if(threadIdx.x==0){for(int i=0;i<=SLOTS;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 if(mode==0)run<0>(p,sm,bar,ch,mi,ni);
 else if(mode==1)run<1>(p,sm,bar,ch,mi,ni);
 else if(mode==2)run<2>(p,sm,bar,ch,mi,ni);
 else run<3>(p,sm,bar,ch,mi,ni);
}
