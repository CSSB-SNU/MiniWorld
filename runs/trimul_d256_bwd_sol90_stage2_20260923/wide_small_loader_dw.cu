// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
constexpr int D=WIDTH,GROUPS=DW_COLUMNS/128,THREADS=GROUPS*128+32,INPUT=4096+64*DW_COLUMNS,SLOTS=3,BAR=INPUT*SLOTS;
struct Params{CUtensorMap gp,xn;float* part;int M;};
// MMA_HELPERS
extern "C" __global__ __launch_bounds__(THREADS,DW_MINBLOCKS)
void mw_wide_small_loader_dw(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int col=blockIdx.y*DW_COLUMNS,split=blockIdx.x/(8*D/64),row=(blockIdx.x%(8*D/64))*64;
 int begin=(p.M/WEIGHT_SPLITS)*split,end=(p.M/WEIGHT_SPLITS)*(split+1);
 if(threadIdx.x==0){for(int i=0;i<2*SLOTS;++i)mbar_init(bar+i,i<SLOTS?1:GROUPS);fence_barrier_init();}__syncthreads();
 if(threadIdx.x>=GROUPS*128){
  if(threadIdx.x==GROUPS*128){
   for(int ki=begin,it=0;ki<end;ki+=32,++it){int slot=it%SLOTS;
    if(it>=SLOTS)mbar_wait(bar+SLOTS+slot,((it/SLOTS)-1)&1);
    mbar_arrive_expect_tx(bar+slot,INPUT);
    tma_load_2d(sm+slot*INPUT,&p.gp,bar+slot,ki,row);
    for(int c=0;c<DW_COLUMNS;c+=64)tma_load_2d(sm+slot*INPUT+4096+(c/64)*4096,&p.xn,bar+slot,col+c,ki);
   }
  }
 }else{
  int WG=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,warp=tid/32;float v[64]={};
  for(int ki=begin,it=0;ki<end;ki+=32,++it){int slot=it%SLOTS;
   mbar_wait(bar+slot,(it/SLOTS)&1);named_bar_sync(1+WG,128);
   fence_regs(v);wgmma_fence();
   static_for<2>([&](auto qq){constexpr int q=decltype(qq)::value;
    mma128_off<q*32,q*2048,0,1>(v,smem_desc(smem_u32(sm+slot*INPUT),16,512,2),smem_desc(smem_u32(sm+slot*INPUT+4096+WG*8192),4096,1024,1),it>0||q>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(v);
   named_bar_sync(1+WG,128);if(tid==0)mbar_arrive(bar+SLOTS+slot);
  }
  static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;
   int r=warp*16+lane/4+8*((j/2)&1),c=col+WG*128+(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
   if(c<D)p.part[size_t(split)*11*D*D+3*D*D+(row+r)*D+c]=v[j];
  });
 }
}
