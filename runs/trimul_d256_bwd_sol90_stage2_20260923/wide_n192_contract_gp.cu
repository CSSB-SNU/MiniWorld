// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
constexpr int D=WIDTH,H=2*D,INPUT=40960,SLOTS=2,BAR=114688;
struct Params {CUtensorMap a[4],b[4];const bf* pre;const bf* mask;bf* gp[4];bf* dl;bf* dr;CUtensorMap premap,maskmap,outmap[4];int N;};
template<int MODE> TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni,int ki,int slot){
 constexpr int TA=MODE==1,TB=MODE!=2;
 mbar_arrive_expect_tx(bar+slot,INPUT);
 for(int g=0;g<2;++g){int m=mi+64*g;
  tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*p.N+(TA?ki:m));
 }
 for(int g=0;g<3;++g){int n=ni+g*64;tma_load_2d(sm+slot*INPUT+16384+g*8192,p.b+MODE,bar+slot,TB?n:ki,ch*p.N+(TB?ki:n));}
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int TA=MODE==1,TB=MODE!=2,SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int WG=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
 float v[96]={};
 if(threadIdx.x==0){
  load_input<MODE>(p,sm,bar,ch,mi,ni,0,0);
  if constexpr(SLOTS==3)load_input<MODE>(p,sm,bar,ch,mi,ni,64,1);
 }
 for(int ki=0,it=0;ki<p.N;ki+=64,++it){
  int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);__syncthreads();
  if(threadIdx.x==0 && ki+(SLOTS-1)*64<p.N)load_input<MODE>(p,sm,bar,ch,mi,ni,ki+(SLOTS-1)*64,(it+SLOTS-1)%SLOTS);
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma192_off<q*(TA?2048:32),q*(TB?2048:32),TA,TB>(v,smem_desc(smem_u32(sm+slot*INPUT+WG*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+16384),TB?8192:16,1024,1),it>0||q>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);
 }
 __syncthreads();
 int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
 if(threadIdx.x==0){
  mbar_arrive_expect_tx(bar+2,98304);
  for(int wm=0;wm<2;++wm)for(int wn=0;wn<3;++wn){int q=(wm*3+wn)*8192;
   tma_load_2d(sm+q,&p.premap,bar+2,ni+wn*64,(rank*64+pc)*p.N+mi+wm*64);
   tma_load_2d(sm+49152+q,&p.premap,bar+2,ni+wn*64,(rank*64+pc+32)*p.N+mi+wm*64);
  }
 }
 mbar_wait(bar+2,0);__syncthreads();
 static_for<3>([&](auto cc){constexpr int chunk=decltype(cc)::value;
  if(threadIdx.x==0){
   mbar_arrive_expect_tx(bar+3,16384);
   for(int wm=0;wm<2;++wm)tma_load_2d(sm+98304+wm*8192,&p.maskmap,bar+3,ni+chunk*64,mi+wm*64);
  }
  mbar_wait(bar+3,chunk&1);__syncthreads();
  static_for<16>([&](auto jj){constexpr int j=chunk*32+decltype(jj)::value*2;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   uint32_t in=WG*8192+swz128(r,(c%64)*2),off=(WG*3+chunk)*8192+swz128(r,(c%64)*2);
   uint32_t raw=pack_bf16(v[j],v[j+1]),masked,mask=*reinterpret_cast<uint32_t*>(sm+98304+in);
   asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
   uint32_t gr=*reinterpret_cast<uint32_t*>(sm+off),pr=*reinterpret_cast<uint32_t*>(sm+49152+off);
   float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
   *reinterpret_cast<uint32_t*>(sm+off)=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
   *reinterpret_cast<uint32_t*>(sm+49152+off)=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
  });fence_proxy_async();__syncthreads();
  if(threadIdx.x==0){
   for(int g=0;g<2;++g)for(int wm=0;wm<2;++wm){
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(p.outmap+2*SIDE+g),"r"(smem_u32(sm+g*49152+(wm*3+chunk)*8192)),"r"(ni+chunk*64),"r"(outch*p.N+mi+wm*64):"memory");
   }tma_store_commit();
  }
 });
 if(threadIdx.x==0)tma_store_wait_all();

}
extern "C" __global__ __launch_bounds__(256,2)
void mw_wide_n192_contract_gp(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tiles=(p.N/128)*(p.N/192),mode=blockIdx.x/(D*tiles),rem=blockIdx.x%(D*tiles);
 int tile=(rem/CHANNEL_GROUP)%tiles,ch=(rem%CHANNEL_GROUP)+CHANNEL_GROUP*(rem/(CHANNEL_GROUP*tiles));
 int mi=(tile/(p.N/192))*128,ni=(tile%(p.N/192))*192;
 if(threadIdx.x==0){for(int i=0;i<4;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 if(mode==0)run<0>(p,sm,bar,ch,mi,ni);
 else if(mode==1)run<1>(p,sm,bar,ch,mi,ni);
 else if(mode==2)run<2>(p,sm,bar,ch,mi,ni);
 else run<3>(p,sm,bar,ch,mi,ni);
}
