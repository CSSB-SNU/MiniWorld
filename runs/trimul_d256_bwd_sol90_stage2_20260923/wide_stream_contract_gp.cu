// Three independently advancing stages: TMA, ordered contraction, GP.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
constexpr int D=WIDTH,H=2*D,INPUT=32768,SLOTS=3,PRE=INPUT*SLOTS,RAW=PRE+98304,BAR=RAW+32768,GPWG=GP_GROUPS;
struct Params {CUtensorMap a[4],b[4];const bf* pre;const bf* mask;bf* gp[4];bf* dl;bf* dr;CUtensorMap premap,maskmap,outmap[4];int N;};
TMN_DEVI void decode(const Params& p,int work,int& ch,int& mi,int& ni){
 int tiles=p.N/128;ch=work/(tiles*tiles);int tile=work%(tiles*tiles);mi=(tile/tiles)*128;ni=(tile%tiles)*128;
}
template<int MODE> TMN_DEVI void produce(const Params& p,uint8_t* sm,uint64_t* bar){
 if(threadIdx.x!=0)return;
 constexpr int TA=MODE==1,TB=MODE!=2,SIDE=MODE==1||MODE==3,HALF=MODE>=2;
 int tiles=p.N/128,steps=p.N/64,total=D*tiles*tiles;
 for(int work=blockIdx.x/4,seq=0;work<total;work+=gridDim.x/4,++seq){
  int ch,mi,ni;decode(p,work,ch,mi,ni);
  for(int it=0;it<steps;++it){int global=seq*steps+it,slot=global%SLOTS,ki=it*64;
   if(global>=SLOTS)mbar_wait(bar+SLOTS+slot,((global/SLOTS)-1)&1);
   mbar_arrive_expect_tx(bar+slot,INPUT);
   for(int g=0;g<2;++g){int m=mi+64*g,n=ni+64*g;
    tma_load_2d(sm+slot*INPUT+g*8192,p.a+MODE,bar+slot,TA?m:ki,ch*p.N+(TA?ki:m));
    tma_load_2d(sm+slot*INPUT+16384+g*8192,p.b+MODE,bar+slot,TB?n:ki,ch*p.N+(TB?ki:n));
   }
  }
  if(seq)mbar_wait(bar+8,(seq-1)&1);
  int outch=ch+HALF*D,rank=SIDE*(H/32)+outch/32,pc=outch%32;
  mbar_arrive_expect_tx(bar+6,98304);
  for(int wm=0;wm<2;++wm)for(int wn=0;wn<2;++wn){int q=(wm*2+wn)*8192;
   tma_load_2d(sm+PRE+q,&p.premap,bar+6,ni+64*wn,(rank*64+pc)*p.N+mi+64*wm);
   tma_load_2d(sm+PRE+32768+q,&p.premap,bar+6,ni+64*wn,(rank*64+pc+32)*p.N+mi+64*wm);
   tma_load_2d(sm+PRE+65536+q,&p.maskmap,bar+6,ni+64*wn,mi+64*wm);
  }
 }
}
template<int MODE,int WG> TMN_DEVI void contract(const Params& p,uint8_t* sm,uint64_t* bar){
 constexpr int TA=MODE==1,TB=MODE!=2;
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,tiles=p.N/128,steps=p.N/64,total=D*tiles*tiles;
 for(int work=blockIdx.x/4,seq=0;work<total;work+=gridDim.x/4,++seq){float v[64]={};
  for(int it=0;it<steps;++it){int global=seq*steps+it,slot=global%SLOTS;
   mbar_wait(bar+slot,(global/SLOTS)&1);named_bar_sync(1+WG,128);
   fence_regs(v);wgmma_fence();
   static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
    mma128_off<k*(TA?2048:32),k*(TB?2048:32),TA,TB>(v,smem_desc(smem_u32(sm+slot*INPUT+WG*8192),TA?8192:16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+16384),TB?8192:16,1024,1),it>0||k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(v);
   named_bar_sync(1+WG,128);if(tid==0)mbar_arrive(bar+SLOTS+slot);
  }
  if(seq)mbar_wait(bar+8,(seq-1)&1);
  static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   uint32_t off=(WG*2+c/64)*8192+swz128(r,(c%64)*2);
   *reinterpret_cast<uint32_t*>(sm+RAW+off)=pack_bf16(v[j],v[j+1]);
  });
  named_bar_sync(1+WG,128);if(tid==0)mbar_arrive(bar+7);
 }
}
template<int MODE> TMN_DEVI void pointwise(const Params& p,uint8_t* sm,uint64_t* bar){
 constexpr int SIDE=MODE==1||MODE==3,HALF=MODE>=2;
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,tiles=p.N/128,total=D*tiles*tiles;
 for(int work=blockIdx.x/4,seq=0;work<total;work+=gridDim.x/4,++seq){
  mbar_wait(bar+6,seq&1);mbar_wait(bar+7,seq&1);named_bar_sync(3,128*GPWG);
  static_for<(GPWG==4?1:2/GPWG)>([&](auto ww){int gw=threadIdx.x/128-3,wg=GPWG==4?gw/2:gw*(2/GPWG)+decltype(ww)::value;
   static_for<(GPWG==4?16:32)>([&](auto jj){constexpr int j=decltype(jj)::value*2;
    int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(GPWG==4?(gw%2)*64:0);
    uint32_t off=(wg*2+c/64)*8192+swz128(r,(c%64)*2);
    uint32_t raw=*reinterpret_cast<uint32_t*>(sm+RAW+off),masked,mask=*reinterpret_cast<uint32_t*>(sm+PRE+65536+off);
    asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(raw),"r"(mask));
    uint32_t gr=*reinterpret_cast<uint32_t*>(sm+PRE+off),pr=*reinterpret_cast<uint32_t*>(sm+PRE+32768+off);
    float ga=math::sigmoid(bf16lo(gr)),gb=math::sigmoid(bf16hi(gr)),pa=bf16lo(pr),pb=bf16hi(pr);
    uint32_t dp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
    uint32_t dg=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));
    *reinterpret_cast<uint32_t*>(sm+PRE+off)=dp;*reinterpret_cast<uint32_t*>(sm+PRE+32768+off)=dg;
   });
  });
  fence_proxy_async();named_bar_sync(3,128*GPWG);
  if(threadIdx.x==384){int ch,mi,ni;decode(p,work,ch,mi,ni);int outch=ch+HALF*D;
   for(int g=0;g<2;++g)for(int wm=0;wm<2;++wm)for(int wn=0;wn<2;++wn){
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(p.outmap+2*SIDE+g),"r"(smem_u32(sm+PRE+g*32768+(wm*2+wn)*8192)),"r"(ni+wn*64),"r"(outch*p.N+mi+wm*64):"memory");
   }
   tma_store_commit();
   if constexpr(STORE_READ_CREDIT)tma_store_wait_read<0>();else tma_store_wait_all();
   mbar_arrive(bar+8);
  }
 }
 if(threadIdx.x==384)tma_store_wait_all();
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar){
 if(threadIdx.x<128){if constexpr(GPWG==4)setmaxnreg_dec<24>();produce<MODE>(p,sm,bar);}
 else if(threadIdx.x<256){if constexpr(GPWG==4)setmaxnreg_inc<112>();contract<MODE,0>(p,sm,bar);}
 else if(threadIdx.x<384){if constexpr(GPWG==4)setmaxnreg_inc<112>();contract<MODE,1>(p,sm,bar);}
 else{if constexpr(GPWG==4)setmaxnreg_dec<64>();pointwise<MODE>(p,sm,bar);}
}
extern "C" __global__ __launch_bounds__((3+GPWG)*128,1)
void mw_wide_stream_contract_gp(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 if(threadIdx.x==0){for(int i=0;i<9;++i)mbar_init(bar+i,(i>=3&&i<6)||i==7?2:1);fence_barrier_init();}
 __syncthreads();int mode=blockIdx.x%4;
 if(mode==0)run<0>(p,sm,bar);else if(mode==1)run<1>(p,sm,bar);else if(mode==2)run<2>(p,sm,bar);else run<3>(p,sm,bar);
}
