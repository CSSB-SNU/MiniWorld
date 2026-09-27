// SPDX-License-Identifier: Apache-2.0
// Experimental DSM B7: exchange a complete 64 KiB derivative plane per phase.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
// PACKED_HELPER
constexpr int D=256,WEIGHT=40960,GP=106496,FULL=139264,DXB=204800,BARS=221184;
struct Params {CUtensorMap map[10];const bf* mask;float *partial,*part;bf *dxn,*gp[4];uint64_t* clock;const bf* dg;int M;};
TMN_DEVI void mark(const Params& p,int it,int role,int point){if constexpr(CLUSTER_CLOCK){if(it==16&&threadIdx.x%128==0)p.clock[(blockIdx.x*4+role)*128+point]=clock64();}}
TMN_DEVI uint32_t remote(const void* p,int rank){uint32_t a;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(a):"r"(smem_u32(p)),"r"(rank));return a;}
TMN_DEVI void remote_arrive(const void* p,int rank){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote(p,rank)):"memory");}
TMN_DEVI void cluster_wait(const void* p,int phase){uint32_t v;do{asm volatile("{.reg .pred P;mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P,[%1],%2;selp.u32 %0,1,0,P;}":"=r"(v):"r"(smem_u32(p)),"r"(phase):"memory");}while(!v);}
template<int WG> TMN_DEVI void transpose_gp(uint8_t* sm){
 int tid=threadIdx.x%128,lane=tid%32,w=tid/32,mat=lane/8,r8=lane%8;uint32_t f[4][4],base=smem_u32(sm);
 #pragma unroll
 for(int q=0;q<4;++q)ldsm_x4_t(f[q],base+swz128(16*q+r8+((mat&2)?8:0),(16*w+((mat&1)?8:0))*2));
 named_bar_sync(4,256);
 #pragma unroll
 for(int q=0;q<4;++q){uint32_t dst=base-WG*8192+(q/2)*8192+swz128(16*w+r8+((mat&1)?8:0),(WG*32+(q%2)*16+((mat&2)?8:0))*2);stsm_x4(dst,f[q][0],f[q][1],f[q][2],f[q][3]);}
 named_bar_sync(4,256);fence_proxy_async();named_bar_sync(4,256);
}
template<int SIDE> TMN_DEVI void producer(const Params& p,uint8_t* sm,uint64_t* b){
 int rank=blockIdx.x%8,split=blockIdx.x/8,begin=(p.M/64)*split/WEIGHT_SPLITS,end=(p.M/64)*(split+1)/WEIGHT_SPLITS;
 if(threadIdx.x==0){mbar_arrive_expect_tx(b+1,65536);for(int g=0;g<2;++g)for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+g*32768+c*8192,&p.map[1],b+1,c*64,(SIDE*16+rank*2+g)*64);}
 for(int tile=begin,it=0;tile<=end;++tile,++it){
  mark(p,it,0,0);
  if(threadIdx.x==0&&tile<end){mbar_arrive_expect_tx(b,40960);
   for(int c=0;c<4;++c)tma_load_2d(sm+c*8192,&p.map[0],b,c*64,tile*64);
   for(int g=0;g<2;++g)tma_load_2d(sm+32768+g*4096,&p.map[2+SIDE],b,tile*64,(rank*2+g)*32);
  }
  mark(p,it,0,1);
  if(threadIdx.x==32&&rank<4&&it>0){constexpr int NSTEPS=SIDE==0?20:16;
   for(int step=0;step<NSTEPS;++step){int slot=step%2,tick=(it-1)*NSTEPS+step;
    if(tick>=2)mbar_wait(b+6+slot,((tick/2)-1)&1);
    mbar_arrive_expect_tx(b+4+slot,8192);int j=step-(SIDE==0?4:0);
    if(SIDE==0&&step<4)tma_load_2d(sm+DXB+slot*8192,&p.map[5],b+4+slot,rank*64,step*64);
    else tma_load_2d(sm+DXB+slot*8192,&p.map[6+SIDE*2+j/8],b+4+slot,rank*64,(j%8)*64);
   }
  }
  mark(p,it,0,6);cooperative_groups::this_cluster().sync();mark(p,it,0,7);
  if(threadIdx.x==64&&tile<end){
   for(int plane=0;plane<2;++plane){
    if(plane)cluster_wait(b+3,it&1);
    uint8_t* src=sm+GP+(it%2)*16384+(plane==0?8192:0);
    for(int peer=0;peer<4;++peer)asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0],[%1],8192,[%2];"::"r"(remote(sm+FULL+rank*8192,peer)),"r"(smem_u32(src)),"r"(remote(b+2,peer)):"memory");
   }
  }
 }
}
template<int SIDE,int WG> TMN_DEVI void source(const Params& p,uint8_t* sm,uint64_t* b){
 constexpr int SYNC=WG+1;int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,rank=SIDE*16+(blockIdx.x%8)*2+WG,split=blockIdx.x/8;
 int begin=(p.M/64)*split/WEIGHT_SPLITS,end=(p.M/64)*(split+1)/WEIGHT_SPLITS;
 mbar_wait(b+1,0);named_bar_sync(SYNC,128);float dw0[64]={},dw1[64]={};
 for(int tile=begin,it=0;tile<end;++tile,++it){int row=tile*64;uint8_t* gp=sm+GP+(it%2)*16384+WG*8192;
  mark(p,it,WG+1,0);mbar_wait(b,it&1);named_bar_sync(SYNC,128);mark(p,it,WG+1,1);float pre[32]={};fence_regs(pre);wgmma_fence();
  static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;mma64_off<(k/4)*8192+(k%4)*32,(k/4)*8192+(k%4)*32,0,0>(pre,smem_desc(smem_u32(sm),16,1024,1),smem_desc(smem_u32(sm+WEIGHT+WG*32768),16,1024,1),k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(pre);
  mark(p,it,WG+1,2);
  int ra=warp*16+lane/4;uint32_t ma=uint32_t(__bfloat16_as_ushort(p.mask[row+ra]))*0x10001u,mb=uint32_t(__bfloat16_as_ushort(p.mask[row+ra+8]))*0x10001u;
  packed_cluster_glu(pre,sm+32768+WG*4096,gp,ma,mb);named_bar_sync(SYNC,128);fence_proxy_async();named_bar_sync(SYNC,128);
  mark(p,it,WG+1,3);
  fence_regs(dw0);fence_regs(dw1);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;uint64_t a=smem_desc(smem_u32(gp),16,1024,1);mma128_off<k*32,k*2048,0,1>(dw0,a,smem_desc(smem_u32(sm),8192,1024,1),it>0||k>0);mma128_off<k*32,k*2048,0,1>(dw1,a,smem_desc(smem_u32(sm+16384),8192,1024,1),it>0||k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);named_bar_sync(SYNC,128);mark(p,it,WG+1,4);transpose_gp<WG>(gp);mark(p,it,WG+1,5);
  if constexpr(CLUSTER_DUMP){for(int i=tid;i<4096;i+=128){int r=i/64,c=i%64,plane=SIDE*2+(c<32?1:0),col=(rank%16)*32+c%32;uint8_t* src=sm+GP+(it%2)*16384+(c/32)*8192+swz128(r,(WG*32+c%32)*2);p.gp[plane][size_t(col)*p.M+row+r]=*reinterpret_cast<bf*>(src);}}
  cooperative_groups::this_cluster().sync();
  mark(p,it,WG+1,6);
 }
 cooperative_groups::this_cluster().sync();
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);int which=SIDE*2+(r<32?1:0),outrow=(rank%16)*32+r%32;size_t ix=size_t(split)*11*D*D+(3+2*which)*D*D+outrow*D+c;p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];});
}
template<int SIDE> TMN_DEVI void dx(const Params& p,uint8_t* sm,uint64_t* b){
 int rank=blockIdx.x%8,split=blockIdx.x/8,tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
 int begin=(p.M/64)*split/WEIGHT_SPLITS,end=(p.M/64)*(split+1)/WEIGHT_SPLITS;
 if(rank>=4){for(int it=0;it<=end-begin;++it)cooperative_groups::this_cluster().sync();return;}
 cooperative_groups::this_cluster().sync();
 for(int tile=begin,it=0;tile<end;++tile,++it){int row=tile*64;float v[32]={};
  mark(p,it,3,0);if(tid==0)mbar_arrive_expect_tx(b+2,65536);mbar_wait(b+2,0);named_bar_sync(3,128);mark(p,it,3,1);
  if constexpr(SIDE==1){static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);v[j]=p.partial[size_t(row+r)*D+rank*64+c];});}
  constexpr int NSTEPS=SIDE==0?20:16;
  for(int step=0;step<NSTEPS;++step){int slot=step%2,j=step-(SIDE==0?4:0);
   mark(p,it,3,8+step*4);
   if(j==8){mark(p,it,3,100);if(tid==0){mbar_arrive_expect_tx(b+2,65536);for(int peer=0;peer<8;++peer)remote_arrive(b+3,peer);}mbar_wait(b+2,1);named_bar_sync(3,128);mark(p,it,3,101);}
   mbar_wait(b+4+slot,(step/2)&1);named_bar_sync(3,128);
   mark(p,it,3,9+step*4);
   uint64_t desc=smem_desc(smem_u32(sm+DXB+slot*8192),16,1024,1);
   if(SIDE==0&&step<4){
    uint32_t a[4][4];static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;int r=row+warp*16+lane/4,c=step*64+q*16+(lane%4)*2;auto ptr=reinterpret_cast<const uint32_t*>(p.dg);a[q][0]=ptr[(size_t(r)*D+c)/2];a[q][1]=ptr[(size_t(r+8)*D+c)/2];a[q][2]=ptr[(size_t(r)*D+c+8)/2];a[q][3]=ptr[(size_t(r+8)*D+c+8)/2];fence_regs(a[q]);});
    fence_regs(v);wgmma_fence();static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;mma_rs_trans<q*2048>(v,a[q],uint32_t(desc),uint32_t(desc>>32),step>0||q>0);});wgmma_commit();wgmma_wait<0>();static_for<4>([&](auto qq){fence_regs(a[decltype(qq)::value]);});
   }else{
    fence_regs(v);wgmma_fence();uint64_t a=smem_desc(smem_u32(sm+FULL+(j%8)*8192),16,1024,1);
    static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;mma64_off<q*32,q*2048,0,1>(v,a,desc,1);});wgmma_commit();wgmma_wait<0>();
   }
   fence_regs(v);named_bar_sync(3,128);mark(p,it,3,10+step*4);if(tid==0)mbar_arrive(b+6+slot);mark(p,it,3,11+step*4);if(step==0)mark(p,it,3,2);
  }
  mark(p,it,3,3);
  static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);size_t off=size_t(row+r)*D+rank*64+c;if constexpr(SIDE==0)p.partial[off]=v[j];else p.dxn[off]=__float2bfloat16_rn(v[j]);});
  mark(p,it,3,4);cooperative_groups::this_cluster().sync();mark(p,it,3,5);
 }
}
template<int SIDE> TMN_DEVI void body(const Params& p){
 extern __shared__ __align__(1024) uint8_t sm[];auto b=reinterpret_cast<uint64_t*>(sm+BARS);
 if(threadIdx.x==0){for(int i=0;i<8;++i)mbar_init(b+i,i==3?4:1);fence_barrier_init();}
 __syncthreads();cooperative_groups::this_cluster().sync();int wg=threadIdx.x/128;
 if(wg==0){setmaxnreg_dec<32>();producer<SIDE>(p,sm,b);}
 else if(wg==3){setmaxnreg_dec<CLUSTER_DX_REGS>();dx<SIDE>(p,sm,b);}
 else{setmaxnreg_inc<(480-CLUSTER_DX_REGS)/2>();if(wg==1)source<SIDE,0>(p,sm,b);else source<SIDE,1>(p,sm,b);}
 cooperative_groups::this_cluster().sync();
}
extern "C" __global__ __launch_bounds__(512,1) void mw_d256_cluster_left(__grid_constant__ const Params p){body<0>(p);}
extern "C" __global__ __launch_bounds__(512,1) void mw_d256_cluster_right(__grid_constant__ const Params p){body<1>(p);}
