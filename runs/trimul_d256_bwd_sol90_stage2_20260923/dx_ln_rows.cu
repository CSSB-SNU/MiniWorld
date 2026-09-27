// SPDX-License-Identifier: Apache-2.0
// Two full-width row consumers share each dX weight tile; one TMA producer.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
#include "mma.cuh"
constexpr int D=256,H=512,STAGE=49152,LNBUF=98304,BARS=196608;
struct Params{CUtensorMap map[16];bf* t[24];float* f[13];int M,L;CUtensorMap xmap,resmap,dxmap;};
TMN_DEVI float rd(const bf* x,int i){return __bfloat162float(x[i]);}
TMN_DEVI bf cv(float x){return __float2bfloat16_rn(x);}
TMN_DEVI float wsum(float x){for(int k=16;k;k>>=1)x+=__shfl_xor_sync(0xffffffff,x,k);return x;}
TMN_DEVI int pos(int r,int c){return (c/64)*4096+swz128(r,(c%64)*2)/2;}
TMN_DEVI void put_tile(const CUtensorMap* map,uint8_t* sm,int c,int row){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(c),"r"(row):"memory");}
TMN_DEVI void producer(const Params& p,uint8_t* sm,uint64_t* b){
 if(threadIdx.x)return;
 for(int row=blockIdx.x*128,round=0;row<p.M;row+=gridDim.x*128,++round){
  if(round)mbar_wait(b+6,(round-1)&1);
  for(int step=0;step<36;++step){int slot=step%2,tick=round*36+step;
   if(tick>=2)mbar_wait(b+2+slot,((tick/2)-1)&1);
   int plane=(step-4)/8,k=(step<4?step:(step-4)%8)*64;uint8_t* dst=sm+slot*STAGE;
   mbar_arrive_expect_tx(b+slot,STAGE);
   for(int g=0;g<2;++g){if(step<4)tma_load_2d(dst+g*8192,&p.map[5],b+slot,k,row+g*64);else tma_load_2d(dst+g*8192,&p.map[6+plane],b+slot,row+g*64,k);}
   for(int c=0;c<4;++c)tma_load_2d(dst+(2+c)*8192,&p.map[step<4?2:10+plane],b+slot,c*64,k);
  }
 }
}
template<int WG> TMN_DEVI void consumer(const Params& p,uint8_t* sm,uint64_t* b){
 constexpr int SYNC=WG+1;int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;float gg[8]={},bb[8]={};uint8_t* ln=sm+WG*LNBUF;
 for(int base=blockIdx.x*128,round=0;base<p.M;base+=gridDim.x*128,++round){int row=base+WG*64;float v0[64]={},v1[64]={};
  for(int step=0;step<36;++step){int slot=step%2;uint8_t* buf=sm+slot*STAGE;
   mbar_wait(b+slot,(step/2)&1);named_bar_sync(SYNC,128);fence_regs(v0);fence_regs(v1);wgmma_fence();
   if(step<4){static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;uint64_t a=smem_desc(smem_u32(buf+WG*8192+q*32),16,1024,1);mma128<0,1>(v0,a,smem_desc(smem_u32(buf+16384+q*2048),8192,1024,1),step>0||q>0);mma128<0,1>(v1,a,smem_desc(smem_u32(buf+32768+q*2048),8192,1024,1),step>0||q>0);});}
   else{static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;uint64_t a=smem_desc(smem_u32(buf+WG*8192+q*2048),16,1024,1);mma128<1,1>(v0,a,smem_desc(smem_u32(buf+16384+q*2048),8192,1024,1),1);mma128<1,1>(v1,a,smem_desc(smem_u32(buf+32768+q*2048),8192,1024,1),1);});}
   wgmma_commit();wgmma_wait<0>();fence_regs(v0);fence_regs(v1);named_bar_sync(SYNC,128);if(tid==0)mbar_arrive(b+2+slot);
  }
  // LN reuses both GEMM stages only after both row consumers finish reading.
  named_bar_sync(3,256);
  static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);reinterpret_cast<bf*>(ln)[pos(r,c)]=cv(v0[j]);reinterpret_cast<bf*>(ln)[pos(r,c+128)]=cv(v1[j]);});
  named_bar_sync(SYNC,128);
  if(tid==0){mbar_arrive_expect_tx(b+4+WG,65536);for(int c=0;c<D;c+=64){tma_load_2d(ln+32768+c*128,&p.xmap,b+4+WG,c,row);tma_load_2d(ln+65536+c*128,&p.resmap,b+4+WG,c,row);}}
  mbar_wait(b+4+WG,round&1);named_bar_sync(SYNC,128);
  bf* dxn=reinterpret_cast<bf*>(ln);bf* x=reinterpret_cast<bf*>(ln+32768);bf* res=reinterpret_cast<bf*>(ln+65536);
  for(int r=warp;r<64;r+=4){
   float z=0;for(int c=lane;c<D;c+=32)z+=rd(x,pos(r,c));float mu=wsum(z)/D;z=0;
   for(int c=lane;c<D;c+=32){float val=rd(x,pos(r,c))-mu;z+=val*val;}float rs=rsqrtf(wsum(z)/D+1e-5f),s0=0,s1=0;
   for(int c=lane;c<D;c+=32){float zz=(rd(x,pos(r,c))-mu)*rs,dy=rd(dxn,pos(r,c)),vv=dy*p.f[0][c];s0+=vv;s1+=vv*zz;gg[c/32]+=dy*zz;bb[c/32]+=dy;}
   s0=wsum(s0)/D;s1=wsum(s1)/D;
   for(int c=lane;c<D;c+=32){float zz=(rd(x,pos(r,c))-mu)*rs,vv=rd(dxn,pos(r,c))*p.f[0][c];float dx=(vv-s0-zz*s1)*rs;dxn[pos(r,c)]=cv(dx+rd(res,pos(r,c)));}
  }
  named_bar_sync(SYNC,128);fence_proxy_async();named_bar_sync(SYNC,128);
  if(tid==0){for(int c=0;c<D;c+=64)put_tile(&p.dxmap,ln+c*128,c,row);tma_store_commit();tma_store_wait_all();}named_bar_sync(SYNC,128);if(tid==0)mbar_arrive(b+6);
 }
 float* part=reinterpret_cast<float*>(ln);
 for(int q=0;q<8;++q){int c=lane+q*32;part[warp*D+c]=gg[q];part[(4+warp)*D+c]=bb[q];}named_bar_sync(SYNC,128);
 for(int c=tid;c<D;c+=128){float g=0,bb0=0;for(int w=0;w<4;++w){g+=part[w*D+c];bb0+=part[(4+w)*D+c];}atomicAdd(p.f[8]+c,g);atomicAdd(p.f[9]+c,bb0);}
 int gtid=WG*128+tid;
 for(int which=0;which<4;++which){int offset=(H+D+which*H)*D;
  for(int i=blockIdx.x*256+gtid;i<H*D;i+=gridDim.x*256){float v=0;for(int s=0;s<WEIGHT_SPLITS;++s)v+=p.f[7][size_t(s)*11*D*D+offset+i];p.t[17+which][i]=cv(v);}
 }
}
extern "C" __global__ __launch_bounds__(384,1) void mw_d256_dx_ln_rows(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto b=reinterpret_cast<uint64_t*>(sm+BARS);int tid=threadIdx.x;
 if(tid==0){for(int i=0;i<7;++i)mbar_init(b+i,(i==2||i==3||i==6)?2:1);fence_barrier_init();}
 for(int c=blockIdx.x*384+tid;c<D;c+=gridDim.x*384){p.f[8][c]=0;p.f[9][c]=0;}
 __syncthreads();cooperative_groups::this_grid().sync();
 if(tid<128){setmaxnreg_dec<32>();producer(p,sm,b);}
 else{setmaxnreg_inc<224>();if(tid<256)consumer<0>(p,sm,b);else consumer<1>(p,sm,b);}
}
