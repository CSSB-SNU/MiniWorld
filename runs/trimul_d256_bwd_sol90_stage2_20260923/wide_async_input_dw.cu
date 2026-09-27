// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
// MMA_HELPERS
constexpr int D=WIDTH,GROUPS=D/128,INPUT=8192+128*D,SLOTS=3,BAR=INPUT*SLOTS;
struct Params{CUtensorMap gp,xn;float* part;int M;};
TMN_DEVI void produce(const Params& p,uint8_t* sm,uint64_t* bar,int row,int begin,int end){
 if(threadIdx.x)return;
 for(int ki=begin,it=0;ki<end;ki+=64,++it){int slot=it%SLOTS;
  if(it>=SLOTS)mbar_wait(bar+SLOTS+slot,((it/SLOTS)-1)&1);
  mbar_arrive_expect_tx(bar+slot,INPUT);
  tma_load_2d(sm+slot*INPUT,&p.gp,bar+slot,ki,row);
  for(int c=0;c<D;c+=64)tma_load_2d(sm+slot*INPUT+8192+(c/64)*8192,&p.xn,bar+slot,c,ki);
 }
}
TMN_DEVI void consume(const Params& p,uint8_t* sm,uint64_t* bar,int row,int begin,int end,int split){
 int WG=threadIdx.x/128-1,tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
 float v[64]={};fence_regs(v);wgmma_fence();
 for(int ki=begin,it=0;ki<end;ki+=64,++it){int slot=it%SLOTS;
  mbar_wait(bar+slot,(it/SLOTS)&1);named_bar_sync(1+WG,128);
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma128_off<q*32,q*2048,0,1>(v,smem_desc(smem_u32(sm+slot*INPUT),16,1024,1),smem_desc(smem_u32(sm+slot*INPUT+8192+WG*16384),8192,1024,1),it>0||q>0);
  });wgmma_commit();
  if(it>0){wgmma_wait<1>();named_bar_sync(1+WG,128);if(tid==0)mbar_arrive(bar+SLOTS+(it-1)%SLOTS);}
 }
 wgmma_wait<0>();fence_regs(v);named_bar_sync(1+WG,128);
 if(tid==0)mbar_arrive(bar+SLOTS+((end-begin)/64-1)%SLOTS);
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;
  int r=warp*16+lane/4+8*((j/2)&1),c=WG*128+(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
  p.part[size_t(split)*11*D*D+3*D*D+(row+r)*D+c]=v[j];
 });
}
extern "C" __global__ __launch_bounds__((GROUPS+1)*128,1)
void mw_wide_async_input_dw(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int split=blockIdx.x/(8*D/64),row=(blockIdx.x%(8*D/64))*64,begin=(p.M/WEIGHT_SPLITS)*split,end=(p.M/WEIGHT_SPLITS)*(split+1);
 if(threadIdx.x==0){for(int i=0;i<2*SLOTS;++i)mbar_init(bar+i,i<SLOTS?1:GROUPS);fence_barrier_init();}__syncthreads();
 if(threadIdx.x<128){setmaxnreg_dec<32>();produce(p,sm,bar,row,begin,end);}
 else{setmaxnreg_inc<112>();consume(p,sm,bar,row,begin,end,split);}
}
