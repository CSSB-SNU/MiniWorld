// SPDX-License-Identifier: Apache-2.0
// One TMA producer and one full-width dX/LN consumer warpgroup.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
#include "mma.cuh"
constexpr int D=256,H=512,STAGE=40960;
struct Params{CUtensorMap map[16];bf* t[24];float* f[13];int M,L;CUtensorMap xmap,resmap,dxmap;};
TMN_DEVI float rd(const bf* x,int i){return __bfloat162float(x[i]);}
TMN_DEVI bf cv(float v){return __float2bfloat16_rn(v);}
TMN_DEVI float wsum(float x){for(int k=16;k;k>>=1)x+=__shfl_xor_sync(0xffffffff,x,k);return x;}
TMN_DEVI int pos(int r,int c){return (c/64)*4096+swz128(r,(c%64)*2)/2;}
TMN_DEVI void put_tile(const CUtensorMap* map,uint8_t* sm,int c,int row){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(c),"r"(row):"memory");}
TMN_DEVI void producer(const Params& p,uint8_t* sm,uint64_t* bars){
 if(threadIdx.x)return;
 for(int row=blockIdx.x*64,round=0;row<p.M;row+=gridDim.x*64,++round){
  if(round)mbar_wait(bars+5,(round-1)&1);
  for(int step=0;step<36;++step){int slot=step%2,it=round*36+step;
   if(it>=2)mbar_wait(bars+3+slot,((it/2)-1)&1);
   int plane=(step-4)/8,k=(step<4?step:(step-4)%8)*64;
   uint8_t* dst=sm+slot*STAGE;mbar_arrive_expect_tx(bars+slot,STAGE);
   if(step<4)tma_load_2d(dst,&p.map[5],bars+slot,k,row);
   else tma_load_2d(dst,&p.map[6+plane],bars+slot,row,k);
   for(int c=0;c<4;++c)tma_load_2d(dst+8192*(1+c),&p.map[step<4?2:10+plane],bars+slot,c*64,k);
  }
 }
}
TMN_DEVI void consumer(const Params& p,uint8_t* sm,uint64_t* bars){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;float gg[8]={},bb[8]={};
 for(int row=blockIdx.x*64,round=0;row<p.M;row+=gridDim.x*64,++round){
  float v0[64]={},v1[64]={};
  for(int step=0;step<36;++step){int slot=step%2;uint8_t* buf=sm+slot*STAGE;
   mbar_wait(bars+slot,(step/2)&1);named_bar_sync(1,128);fence_regs(v0);fence_regs(v1);wgmma_fence();
   if(step<4){static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;uint64_t a=smem_desc(smem_u32(buf+q*32),16,1024,1);mma128<0,1>(v0,a,smem_desc(smem_u32(buf+8192+q*2048),8192,1024,1),step>0||q>0);mma128<0,1>(v1,a,smem_desc(smem_u32(buf+24576+q*2048),8192,1024,1),step>0||q>0);});}
   else{static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;uint64_t a=smem_desc(smem_u32(buf+q*2048),16,1024,1);mma128<1,1>(v0,a,smem_desc(smem_u32(buf+8192+q*2048),8192,1024,1),1);mma128<1,1>(v1,a,smem_desc(smem_u32(buf+24576+q*2048),8192,1024,1),1);});}
   wgmma_commit();wgmma_wait<0>();fence_regs(v0);fence_regs(v1);named_bar_sync(1,128);if(tid==0)mbar_arrive(bars+3+slot);
  }
  static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);reinterpret_cast<bf*>(sm)[pos(r,c)]=cv(v0[j]);reinterpret_cast<bf*>(sm)[pos(r,c+128)]=cv(v1[j]);});
  named_bar_sync(1,128);
  if(tid==0){mbar_arrive_expect_tx(bars+2,65536);for(int c=0;c<D;c+=64){tma_load_2d(sm+32768+c*128,&p.xmap,bars+2,c,row);tma_load_2d(sm+65536+c*128,&p.resmap,bars+2,c,row);}}
  mbar_wait(bars+2,round&1);named_bar_sync(1,128);
  bf* dxn=reinterpret_cast<bf*>(sm);bf* x=reinterpret_cast<bf*>(sm+32768);bf* res=reinterpret_cast<bf*>(sm+65536);
  for(int r=warp;r<64;r+=4){
   float z=0;for(int c=lane;c<D;c+=32)z+=rd(x,pos(r,c));float mu=wsum(z)/D;z=0;
   for(int c=lane;c<D;c+=32){float val=rd(x,pos(r,c))-mu;z+=val*val;}float rs=rsqrtf(wsum(z)/D+1e-5f),s0=0,s1=0;
   for(int c=lane;c<D;c+=32){float zz=(rd(x,pos(r,c))-mu)*rs,dy=rd(dxn,pos(r,c)),vv=dy*p.f[0][c];s0+=vv;s1+=vv*zz;gg[c/32]+=dy*zz;bb[c/32]+=dy;}
   s0=wsum(s0)/D;s1=wsum(s1)/D;
   for(int c=lane;c<D;c+=32){float zz=(rd(x,pos(r,c))-mu)*rs,vv=rd(dxn,pos(r,c))*p.f[0][c];float dx=(vv-s0-zz*s1)*rs;dxn[pos(r,c)]=cv(dx+rd(res,pos(r,c)));}
  }
  named_bar_sync(1,128);fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){for(int c=0;c<D;c+=64)put_tile(&p.dxmap,sm+c*128,c,row);tma_store_commit();tma_store_wait_all();}named_bar_sync(1,128);if(tid==0)mbar_arrive(bars+5);
 }
 float* part=reinterpret_cast<float*>(sm);
 for(int q=0;q<8;++q){int c=lane+q*32;part[warp*D+c]=gg[q];part[(4+warp)*D+c]=bb[q];}named_bar_sync(1,128);
 for(int c=tid;c<D;c+=128){float g=0,b=0;for(int w=0;w<4;++w){g+=part[w*D+c];b+=part[(4+w)*D+c];}atomicAdd(p.f[8]+c,g);atomicAdd(p.f[9]+c,b);}
 for(int which=0;which<4;++which){int offset=(H+D+which*H)*D;
  for(int i=blockIdx.x*128+tid;i<H*D;i+=gridDim.x*128){float v=0;for(int s=0;s<WEIGHT_SPLITS;++s)v+=p.f[7][size_t(s)*11*D*D+offset+i];p.t[17+which][i]=cv(v);}
 }
}
extern "C" __global__ __launch_bounds__(256,2) void mw_d256_dx_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bars=reinterpret_cast<uint64_t*>(sm+98304);int tid=threadIdx.x;
 if(tid==0){for(int i=0;i<6;++i)mbar_init(bars+i,1);fence_barrier_init();}
 for(int c=blockIdx.x*256+tid;c<D;c+=gridDim.x*256){p.f[8][c]=0;p.f[9][c]=0;}
 __syncthreads();cooperative_groups::this_grid().sync();
 if(tid<128){setmaxnreg_dec<32>();producer(p,sm,bars);}else{setmaxnreg_inc<224>();consumer(p,sm,bars);}
}
