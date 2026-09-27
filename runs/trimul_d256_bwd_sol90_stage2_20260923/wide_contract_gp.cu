// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
#ifndef PRE_TRANSPOSED
#define PRE_TRANSPOSED 0
#endif
// MMA_HELPERS
constexpr int D=WIDTH,H=2*D,INPUT=16384,BAR=2*INPUT;
struct Params {CUtensorMap a[4],b[4];const bf* pre;const bf* mask;bf* gp[4];bf* dl;bf* dr;int N;};
template<int MODE> TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni,int ki,int slot){
 constexpr int TA=MODE==1,TB=MODE!=2;
 mbar_arrive_expect_tx(bar+slot,INPUT);
 tma_load_2d(sm+slot*INPUT,p.a+MODE,bar+slot,TA?mi:ki,ch*p.N+(TA?ki:mi));
 tma_load_2d(sm+slot*INPUT+8192,p.b+MODE,bar+slot,TB?ni:ki,ch*p.N+(TB?ki:ni));
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int mi,int ni){
 constexpr int TA=MODE==1,TB=MODE!=2;
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;float v[32]={};
 if(tid==0)load<MODE>(p,sm,bar,ch,mi,ni,0,0);
 for(int ki=0,it=0;ki<p.N;ki+=64,++it){
  int slot=it%2;mbar_wait(bar+slot,(it/2)&1);__syncthreads();
  if(tid==0 && ki+64<p.N)load<MODE>(p,sm,bar,ch,mi,ni,ki+64,1-slot);
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma64_off<k*(TA?2048:32),k*(TB?2048:32),TA,TB>(v,smem_desc(smem_u32(sm+slot*INPUT),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+8192),TB?8192:16,1024,1),it>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);
 }
 constexpr int SIDE=(MODE==1||MODE==3),HALF=(MODE>=2);
 int outch=ch+HALF*D,M=p.N*p.N,rank=SIDE*(H/32)+outch/32,pc=outch%32;
 static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=mi+warp*16+lane/4+8*((j/2)&1),c=ni+(j/8)*16+2*(lane%4)+8*((j/2)%4/2),row=r*p.N+c;
  uint32_t raw=pack_bf16(v[j],v[j+1]);
  if constexpr(FUSED_GP){
   uint32_t mask=pack_bf16(__bfloat162float(p.mask[row]),__bfloat162float(p.mask[row+1])),masked;
   asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
   float ga,gb,pa,pb;
   if constexpr(PRE_TRANSPOSED){
    size_t ix=(size_t(rank)*64+pc)*M+row;
    uint32_t gr=*reinterpret_cast<const uint32_t*>(p.pre+ix),pr=*reinterpret_cast<const uint32_t*>(p.pre+ix+size_t(32)*M);
    ga=math::sigmoid(bf16lo(gr));gb=math::sigmoid(bf16hi(gr));pa=bf16lo(pr);pb=bf16hi(pr);
   }else{
    size_t ix=(size_t(rank)*M+row)*64+pc;
    ga=math::sigmoid(__bfloat162float(p.pre[ix]));gb=math::sigmoid(__bfloat162float(p.pre[ix+64]));
    pa=__bfloat162float(p.pre[ix+32]);pb=__bfloat162float(p.pre[ix+96]);
   }
   uint32_t dp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
   uint32_t dg=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
   *reinterpret_cast<uint32_t*>(p.gp[2*SIDE]+size_t(outch)*M+row)=dp;
   *reinterpret_cast<uint32_t*>(p.gp[2*SIDE+1]+size_t(outch)*M+row)=dg;
  }else{*reinterpret_cast<uint32_t*>((SIDE?p.dr:p.dl)+size_t(outch)*M+row)=raw;}
 });
}
extern "C" __global__ __launch_bounds__(128,4)
void mw_wide_contract_gp(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tiles=p.N/64,mode=blockIdx.x/(D*tiles*tiles),rem=blockIdx.x%(D*tiles*tiles);
 int tile=(rem/32)%(tiles*tiles),ch=(rem%32)+32*(rem/(32*tiles*tiles));
 int mi=(tile/tiles)*64,ni=(tile%tiles)*64;
 if(threadIdx.x==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();}__syncthreads();
 if(mode==0)run<0>(p,sm,bar,ch,mi,ni);
 else if(mode==1)run<1>(p,sm,bar,ch,mi,ni);
 else if(mode==2)run<2>(p,sm,bar,ch,mi,ni);
 else run<3>(p,sm,bar,ch,mi,ni);
}
