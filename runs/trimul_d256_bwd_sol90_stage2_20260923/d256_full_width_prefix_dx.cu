// SPDX-License-Identifier: Apache-2.0
// Two row warpgroups, whole-map TMA, and an ordered one-group-deep WGMMA pipe.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPER
constexpr int NSLOT=DX_FULL_SLOTS,KD=DX_FULL_K,STAGE=768*KD,BAR=STAGE*NSLOT,STEPS=2304/KD;
struct Params{CUtensorMap input,weight,output;};
TMN_DEVI void produce(const Params& p,uint8_t* sm,uint64_t* bars){
 if(threadIdx.x)return;
 for(int step=0;step<STEPS;++step){
  int slot=step%NSLOT;
  if(step>=NSLOT)mbar_wait(bars+NSLOT+slot,((step/NSLOT)-1)&1);
  mbar_arrive_expect_tx(bars+slot,STAGE);
  tma_load_3d(sm+slot*STAGE,&p.input,bars+slot,0,step*KD,blockIdx.x*2);
  tma_load_3d(sm+slot*STAGE+256*KD,&p.weight,bars+slot,0,step*KD,0);
 }
}
template<int WG> TMN_DEVI void consume(const Params& p,uint8_t* sm,uint64_t* bars){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
 float acc[128]={};fence_regs(acc);wgmma_fence();
 for(int step=0;step<STEPS;++step){
  int slot=step%NSLOT;uint8_t* buf=sm+slot*STAGE;
  mbar_wait(bars+slot,(step/NSLOT)&1);named_bar_sync(WG+1,128);
  static_for<KD/16>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma256<1,1>(acc,smem_desc(smem_u32(buf+WG*128*KD+q*2048),16,1024,1),smem_desc(smem_u32(buf+256*KD+q*2048),128*KD,1024,1),step>0||q>0);
  });wgmma_commit();
  if(step>=DX_FULL_DEPTH){wgmma_wait<DX_FULL_DEPTH>();named_bar_sync(WG+1,128);if(tid==0)mbar_arrive(bars+NSLOT+(step-DX_FULL_DEPTH)%NSLOT);}
 }
 wgmma_wait<0>();fence_regs(acc);named_bar_sync(WG+1,128);
 if(tid==0)for(int step=STEPS-DX_FULL_DEPTH;step<STEPS;++step)mbar_arrive(bars+NSLOT+step%NSLOT);
 named_bar_sync(3,256);
 uint8_t* out=sm+WG*32768;
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  *reinterpret_cast<uint32_t*>(out+(c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(acc[j],acc[j+1]);
 });
 fence_proxy_async();named_bar_sync(WG+1,128);
 if(tid==0){
  int row=blockIdx.x*128+WG*64;
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{0,%2,0}],[%1];"::"l"(&p.output),"r"(smem_u32(out)),"r"(row):"memory");
  tma_store_commit();tma_store_wait_all();
 }
 named_bar_sync(WG+1,128);
}
extern "C" __global__ __launch_bounds__(384,1)
void mw_d256_full_width_prefix_dx(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bars=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x;
 if(tid==0){for(int i=0;i<2*NSLOT;++i)mbar_init(bars+i,i<NSLOT?1:2);fence_barrier_init();}
 __syncthreads();
 if(tid<128){setmaxnreg_dec<32>();produce(p,sm,bars);}
 else{setmaxnreg_inc<224>();if(tid<256)consume<0>(p,sm,bars);else consume<1>(p,sm,bars);}
}
