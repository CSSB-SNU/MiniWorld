// SPDX-License-Identifier: Apache-2.0
// Two independent 64x128 dW consumers, one shared 64x128 input tile.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
constexpr int D=WIDTH,H=2*D,RANKS=D/8,SHARDS=D/128,STAGE=40960,DERIV=81920,BAR=98304;
struct Params{CUtensorMap xn,w,dl,dr,gmap[4];const bf* mask;float* part;int M;CUtensorMap pre;};
// SAVED_GLU
TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int row,int group,int shard,int slot){
 uint8_t* dst=sm+slot*STAGE;mbar_arrive_expect_tx(bar+slot,STAGE);
 for(int c=0;c<2;++c)tma_load_2d(dst+c*8192,&p.xn,bar+slot,shard*128+c*64,row);
 for(int wg=0;wg<2;++wg){int rank=group*2+wg;
  tma_load_2d(dst+16384+wg*12288,&p.pre,bar+slot,0,rank*p.M+row);
  tma_load_2d(dst+24576+wg*12288,rank<H/32?&p.dl:&p.dr,bar+slot,row,(rank%(H/32))*32);
 }
}
TMN_DEVI void store(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}
extern "C" __global__ __launch_bounds__(256,2)
void mw_wide_saved_dual(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x,lane=tid%32,wg=tid/128,warp=(tid/32)%4,shard=blockIdx.x%SHARDS;
 int group=(blockIdx.x/SHARDS)%(RANKS/2),rank=2*group+wg,split=blockIdx.x/(SHARDS*(RANKS/2));
 int tiles=p.M/64,begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
 if(tid==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();load(p,sm,bar,begin*64,group,shard,0);}
 __syncthreads();float dw[64]={};
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2,row=tile*64;
  mbar_wait(bar+slot,(it/2)&1);__syncthreads();
  if(tid==0 && tile+1<end)load(p,sm,bar,(tile+1)*64,group,shard,1-slot);
  uint8_t* xn=sm+slot*STAGE;uint8_t* gp=sm+DERIV+wg*8192;
  int ra=warp*16+lane/4;
  uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[row+ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(p.mask[row+ra+8]))*0x10001u;
  uint8_t* pre=xn+16384+wg*12288;
  saved_glu(pre,pre,gp,ma,mb);
  named_bar_sync(wg+1,128);fence_proxy_async();named_bar_sync(wg+1,128);
  if(tid%128==0 && shard==0){int side=rank/(H/32),col=(rank%(H/32))*32;
   store(p.gmap+2*side,gp+4096,row,col);store(p.gmap+2*side+1,gp,row,col);tma_store_commit();
  }
  fence_regs(dw);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma128<0,1>(dw,smem_desc(smem_u32(gp+k*32),16,1024,1),smem_desc(smem_u32(xn+k*2048),8192,1024,1),it>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(dw);
  if(tid%128==0 && shard==0)tma_store_wait_all();
  __syncthreads();
 }
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
  int which=(rank/(H/32))*2+(r<32?1:0),outrow=(rank%(H/32))*32+r%32;
  size_t ix=size_t(split)*11*D*D+(3+2*which)*D*D+outrow*D+shard*128+c;
  p.part[ix]=dw[j];
 });
}
