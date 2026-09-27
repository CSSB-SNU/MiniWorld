// Three independently advancing stages: TMA, ordered contraction, GP.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
constexpr int D=WIDTH,H=2*D,INPUT=24576,SLOTS=2,PRE=INPUT*SLOTS,RAW=PRE+49152,BAR=RAW+16384;
struct Params {CUtensorMap a[4],b[4];const bf* pre;const bf* mask;bf* gp[4];bf* dl;bf* dr;CUtensorMap premap,maskmap,outmap[4];int N;};
TMN_DEVI void decode(const Params& p,int work,int& ch,int& mi,int& ni){
 int mt=p.N/128,nt=p.N/64;ch=work/(mt*nt);int tile=work%(mt*nt);mi=(tile/nt)*128;ni=(tile%nt)*64;
}
template<int MODE> TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int seq,int it){
 constexpr int TA=MODE==1,TB=MODE!=2;
 int work=blockIdx.x/4+seq*(gridDim.x/4),ch,mi,ni;decode(p,work,ch,mi,ni);
 int global=seq*(p.N/64)+it,slot=global%SLOTS,ki=it*64;
 if(global>=SLOTS)mbar_wait(bar+SLOTS+slot,((global/SLOTS)-1)&1);
 mbar_arrive_expect_tx(bar+slot,INPUT);
 for(int g=0;g<2;++g){int m=mi+64*g;
  tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*p.N+(TA?ki:m));
 }
 tma_load_2d(sm+slot*INPUT+16384,p.b+MODE,bar+slot,TB?ni:ki,ch*p.N+(TB?ki:ni));
}
template<int MODE> TMN_DEVI void load_pre(const Params& p,uint8_t* sm,uint64_t* bar,int seq){
 constexpr int SIDE=MODE==1||MODE==3,HALF=MODE>=2;
 int work=blockIdx.x/4+seq*(gridDim.x/4),ch,mi,ni;decode(p,work,ch,mi,ni);
 int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
 mbar_arrive_expect_tx(bar+4,49152);
 for(int wm=0;wm<2;++wm){int q=wm*8192;
  tma_load_2d(sm+PRE+q,&p.premap,bar+4,ni,(rank*64+pc)*p.N+mi+64*wm);
  tma_load_2d(sm+PRE+16384+q,&p.premap,bar+4,ni,(rank*64+pc+32)*p.N+mi+64*wm);
  tma_load_2d(sm+PRE+32768+q,&p.maskmap,bar+4,ni,mi+64*wm);
 }
}
template<int MODE,int WG> TMN_DEVI void contract(const Params& p,uint8_t* sm,uint64_t* bar){
 constexpr int TA=MODE==1,TB=MODE!=2;
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,tiles=p.N/128,steps=p.N/64,total=2*D*tiles*tiles;
 for(int work=blockIdx.x/4,seq=0;work<total;work+=gridDim.x/4,++seq){float v[32]={};
  if(threadIdx.x==0){load_input<MODE>(p,sm,bar,seq,0);load_input<MODE>(p,sm,bar,seq,1);}
  for(int it=0;it<steps;++it){int global=seq*steps+it,slot=global%SLOTS;
   mbar_wait(bar+slot,(global/SLOTS)&1);named_bar_sync(1+WG,128);
   fence_regs(v);wgmma_fence();
   static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
    mma64<TA,TB>(v,smem_desc(smem_u32(sm+slot*INPUT+WG*8192+k*(TA?2048:32)),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+16384+k*(TB?2048:32)),TB?8192:16,1024,1),it>0||k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(v);
   named_bar_sync(1+WG,128);if(tid==0)mbar_arrive(bar+SLOTS+slot);
   if(threadIdx.x==0 && it+2<steps)load_input<MODE>(p,sm,bar,seq,it+2);
  }
  if(seq)mbar_wait(bar+6,(seq-1)&1);
  if(threadIdx.x==0)load_pre<MODE>(p,sm,bar,seq);
  static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value*2;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   uint32_t off=WG*8192+swz128(r,(c%64)*2);
   *reinterpret_cast<uint32_t*>(sm+RAW+off)=pack_bf16(v[j],v[j+1]);
  });
  named_bar_sync(1+WG,128);if(tid==0)mbar_arrive(bar+5);
 }
}
template<int MODE> TMN_DEVI void pointwise(const Params& p,uint8_t* sm,uint64_t* bar){
 constexpr int SIDE=MODE==1||MODE==3,HALF=MODE>=2;
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,tiles=p.N/128,total=2*D*tiles*tiles;
 for(int work=blockIdx.x/4,seq=0;work<total;work+=gridDim.x/4,++seq){
  mbar_wait(bar+4,seq&1);mbar_wait(bar+5,seq&1);named_bar_sync(3,256);
  int wg=threadIdx.x/128-2;
  static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value*2;
    int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
    uint32_t off=wg*8192+swz128(r,c*2);
    uint32_t raw=*reinterpret_cast<uint32_t*>(sm+RAW+off),masked,mask=*reinterpret_cast<uint32_t*>(sm+PRE+32768+off);
    asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
    uint32_t gr=*reinterpret_cast<uint32_t*>(sm+PRE+off),pr=*reinterpret_cast<uint32_t*>(sm+PRE+16384+off);
    float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
    uint32_t dp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
    uint32_t dg=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
    *reinterpret_cast<uint32_t*>(sm+PRE+off)=dp;*reinterpret_cast<uint32_t*>(sm+PRE+16384+off)=dg;
  });
  fence_proxy_async();named_bar_sync(3,256);
  if(threadIdx.x==256){int ch,mi,ni;decode(p,work,ch,mi,ni);int outch=ch+HALF*D;
   for(int g=0;g<2;++g)for(int wm=0;wm<2;++wm){int wn=0;
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(p.outmap+2*SIDE+g),"r"(smem_u32(sm+PRE+g*16384+wm*8192)),"r"(ni+wn*64),"r"(outch*p.N+mi+wm*64):"memory");
   }
   tma_store_commit();
   if constexpr(STORE_READ_CREDIT)tma_store_wait_read<0>();else tma_store_wait_all();
   mbar_arrive(bar+6);
  }
 }
 if(threadIdx.x==256)tma_store_wait_all();
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar){
 if(threadIdx.x<128)contract<MODE,0>(p,sm,bar);
 else if(threadIdx.x<256)contract<MODE,1>(p,sm,bar);
 else pointwise<MODE>(p,sm,bar);
}
extern "C" __global__ __launch_bounds__(512,STREAM_MINBLOCKS)
void mw_wide_compact_stream_contract_gp(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 if(threadIdx.x==0){for(int i=0;i<7;++i)mbar_init(bar+i,(i>=2&&i<4)||i==5?2:1);fence_barrier_init();}
 __syncthreads();int mode=blockIdx.x%4;
 if(mode==0)run<0>(p,sm,bar);else if(mode==1)run<1>(p,sm,bar);else if(mode==2)run<2>(p,sm,bar);else run<3>(p,sm,bar);
}
