// Three spatial row CTAs retain their A and B panels and exchange B through DSM.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;
// MMA_HELPER
constexpr int D=256,N=384,A=0,B=98304,TEMP=196608,BAR=229376;
struct Params{CUtensorMap a[4],b[4],out[2];};
TMN_DEVI uint32_t remote_addr(const void* ptr,int rank){
 uint32_t value;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(value):"r"(smem_u32(ptr)),"r"(rank));return value;
}
TMN_DEVI void push_panel(uint8_t* sm,uint64_t* bar,int receiver,int step){
 int slot=step%2;
 asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0],[%1],16384,[%2];"
  ::"r"(remote_addr(sm+TEMP+slot*16384,receiver)),"r"(smem_u32(sm+B+step*16384)),"r"(remote_addr(bar+1+slot,receiver)):"memory");
}
TMN_DEVI void return_credit(uint64_t* bar,int sender,int slot){
 asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote_addr(bar+3+slot,sender)):"memory");
}
TMN_DEVI void wait_credit(uint64_t* bar,int phase){
 uint32_t ready;do{asm volatile("{.reg .pred P;mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P,[%1],%2;selp.u32 %0,1,0,P;}"
  :"=r"(ready):"r"(smem_u32(bar)),"r"(phase):"memory");}while(!ready);
}
template<int MODE> TMN_DEVI void run(const Params& p,uint8_t* sm,uint64_t* bar,int ch,int rank){
 constexpr int TA=MODE==1,TB=MODE!=2,SIDE=(MODE==1||MODE==3),HALF=MODE>=2;
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,wg=threadIdx.x/128,mi=rank*128;
 if(threadIdx.x==0){
  mbar_arrive_expect_tx(bar,196608);
  #pragma unroll
  for(int step=0;step<6;++step){int ki=step*64;
   #pragma unroll
   for(int g=0;g<2;++g){int m=mi+g*64,n=rank*128+g*64;
    tma_load_2d(sm+A+step*16384+g*8192,p.a+MODE,bar,TA?m:ki,ch*N+(TA?ki:m));
    tma_load_2d(sm+B+step*16384+g*8192,p.b+MODE,bar,TB?n:ki,ch*N+(TB?ki:n));
   }
  }
 }
 mbar_wait(bar,0);__syncthreads();cooperative_groups::this_cluster().sync();
 for(int phase=0;phase<3;++phase){
  int owner=(rank+phase)%3,receiver=(rank-phase+3)%3;
  if(phase){
   if(threadIdx.x==0){mbar_arrive_expect_tx(bar+1,16384);mbar_arrive_expect_tx(bar+2,16384);}
   __syncthreads();cooperative_groups::this_cluster().sync();
   if(threadIdx.x==0){push_panel(sm,bar,receiver,0);push_panel(sm,bar,receiver,1);}
  }
  float v[64]={};
  #pragma unroll
  for(int step=0;step<6;++step){
   int slot=step%2;
   if(phase)mbar_wait(bar+1+slot,((phase-1)*3+step/2)&1);
   __syncthreads();
   uint8_t* ap=sm+A+step*16384+wg*8192;
   uint8_t* bp=phase?sm+TEMP+slot*16384:sm+B+step*16384;
   fence_regs(v);wgmma_fence();
   static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
    mma128_off<k*(TA?2048:32),k*(TB?2048:32),TA,TB>(v,smem_desc(smem_u32(ap),TA?8192:16,1024,1),smem_desc(smem_u32(bp),TB?8192:16,1024,1),step>0||k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
   if(phase && step+2<6 && threadIdx.x==0){
    mbar_arrive_expect_tx(bar+1+slot,16384);
    return_credit(bar,owner,slot);
    wait_credit(bar+3+slot,step/2);
    push_panel(sm,bar,receiver,step+2);
   }
  }
  __syncthreads();
  static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   *reinterpret_cast<uint32_t*>(sm+TEMP+(wg*2+c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(v[j],v[j+1]);
  });fence_proxy_async();__syncthreads();
  if(threadIdx.x==0){
   #pragma unroll
   for(int wm=0;wm<2;++wm)for(int wn=0;wn<2;++wn){
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"
     ::"l"(p.out+SIDE),"r"(smem_u32(sm+TEMP+(wm*2+wn)*8192)),"r"(owner*128+wn*64),"r"((ch+HALF*D)*N+mi+wm*64):"memory");
   }
   tma_store_commit();tma_store_wait_all();
  }
  __syncthreads();cooperative_groups::this_cluster().sync();
 }
}
extern "C" __global__ __launch_bounds__(256,1)
void mw_d256_resident_cluster_contract(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int cluster=blockIdx.x/3,rank=blockIdx.x%3,half=cluster/(2*D),rem=cluster%(2*D),ch=rem/2,mode=half*2+rem%2;
 if(threadIdx.x==0){for(int i=0;i<5;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 if(mode==0)run<0>(p,sm,bar,ch,rank);
 else if(mode==1)run<1>(p,sm,bar,ch,rank);
 else if(mode==2)run<2>(p,sm,bar,ch,rank);
 else run<3>(p,sm,bar,ch,rank);
}
