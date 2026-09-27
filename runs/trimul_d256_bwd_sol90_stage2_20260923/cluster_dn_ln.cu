#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
constexpr int D=WIDTH,H=2*D,ACTIVE=H/128,CLSIZE=D==256?4:8,STAGE=24576,SLOTS=3;
constexpr int TRI=STAGE*SLOTS,CHAIN=16384,FINAL=32768,BAR=TRI+16384;
struct Params{CUtensorMap dp,wp,tri,dt,dn;const float* mu;const float* rs;const float* gamma;float* partial;float* dg;float* db;int M;};
TMN_DEVI uint32_t remote_addr(const void* p,int rank){uint32_t r;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(r):"r"(smem_u32(p)),"r"(rank));return r;}
TMN_DEVI float remote_float(const void* p,int rank){float x;asm volatile("ld.shared::cluster.f32 %0,[%1];":"=f"(x):"r"(remote_addr(p,rank)):"memory");return x;}
TMN_DEVI void remote_arrive(const void* p,int rank){asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote_addr(p,rank)):"memory");}
TMN_DEVI void cluster_wait(const void* p){uint32_t done;do{asm volatile("{.reg .pred P;mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P,[%1],0;selp.u32 %0,1,0,P;}":"=r"(done):"r"(smem_u32(p)):"memory");}while(!done);}
TMN_DEVI float sumwarp(float x){for(int k=16;k;k>>=1)x+=__shfl_xor_sync(0xffffffff,x,k);return x;}
TMN_DEVI int pos(int r,int c){return (c/64)*4096+swz128(r,(c%64)*2)/2;}
TMN_DEVI void put(const CUtensorMap* map,uint8_t* buf,int a,int b){asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(buf)),"r"(a),"r"(b):"memory");}
// Two 64x64 tiles, both source and destination use 128-byte swizzling.
template<bool INVERSE> TMN_DEVI void transpose(uint8_t* sm){
 int lane=threadIdx.x%32,warp=threadIdx.x/32,mat=lane/8,r8=lane%8;
 for(int tile=0;tile<2;++tile){uint32_t f[4][4],base=smem_u32(sm+tile*8192);
  #pragma unroll
  for(int q=0;q<4;++q){
   uint32_t raw=swz128(16*q+r8+((mat&2)?8:0),(16*warp+((mat&1)?8:0))*2);
   uint32_t row=swz128(16*warp+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
   ldsm_x4_t(f[q],base+(INVERSE?row:raw));
  }
  __syncthreads();
  #pragma unroll
  for(int q=0;q<4;++q){
   uint32_t raw=swz128(16*q+r8+((mat&2)?8:0),(16*warp+((mat&1)?8:0))*2);
   uint32_t row=swz128(16*warp+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2);
   stsm_x4(base+(INVERSE?raw:row),f[q][0],f[q][1],f[q][2],f[q][3]);
  }
  __syncthreads();
 }
}
TMN_DEVI void load_gemm(const Params& p,uint8_t* sm,uint64_t* bar,int row,int col,int step){
 int slot=step%SLOTS;uint8_t* dst=sm+slot*STAGE;
 mbar_arrive_expect_tx(bar+slot,STAGE);
 tma_load_2d(dst,&p.dp,bar+slot,step*64,row);
 for(int c=0;c<2;++c)tma_load_2d(dst+8192+c*8192,&p.wp,bar+slot,col+c*64,step*64);
}
TMN_DEVI void gemm(const Params& p,uint8_t* sm,uint64_t* bar,int row,int col){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;float v[64]={};
 if(tid==0){load_gemm(p,sm,bar,row,col,0);load_gemm(p,sm,bar,row,col,1);
  mbar_arrive_expect_tx(bar+3,16384);
  for(int c=0;c<2;++c)tma_load_2d(sm+TRI+c*8192,&p.tri,bar+3,row,col+c*64);
 }
 for(int step=0;step<D/64;++step){int slot=step%SLOTS;uint8_t* src=sm+slot*STAGE;
  mbar_wait(bar+slot,(step/SLOTS)&1);__syncthreads();
  if(tid==0 && step+2<D/64)load_gemm(p,sm,bar,row,col,step+2);
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma128<0,1>(v,smem_desc(smem_u32(src+q*32),16,1024,1),smem_desc(smem_u32(src+8192+q*2048),8192,1024,1),step>0||q>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
 }
 static_for<32>([&](auto qq){constexpr int j=2*decltype(qq)::value;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  *reinterpret_cast<uint32_t*>(sm+2*pos(r,c))=pack_bf16(v[j],v[j+1]);
 });
 __syncthreads();
 if constexpr(EMIT_DN){fence_proxy_async();__syncthreads();if(tid==0){for(int c=0;c<2;++c)put(&p.dn,sm+c*8192,col+c*64,row);tma_store_commit();tma_store_wait_all();}__syncthreads();}
 mbar_wait(bar+3,0);__syncthreads();transpose<false>(sm+TRI);
}
TMN_DEVI void normalize(const Params& p,uint8_t* sm,uint64_t* bar,int row,int rank){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,col=rank*128;
 float gg[4]={},bb[4]={},gamma[4];
 #pragma unroll
 for(int q=0;q<4;++q)gamma[q]=p.gamma[col+lane+q*32];
 float* sums=reinterpret_cast<float*>(sm+CHAIN);float* result=reinterpret_cast<float*>(sm+FINAL);
 if(rank)cluster_wait(bar+4);__syncthreads();
 for(int r=warp;r<64;r+=4){float mu=p.mu[row+r],rs=p.rs[row+r];
  float s0=rank?remote_float(sums+r*32+lane,rank-1):0.f;
  float s1=rank?remote_float(sums+2048+r*32+lane,rank-1):0.f;
  #pragma unroll
  for(int q=0;q<4;++q){int c=lane+q*32;
   float z=(__bfloat162float(reinterpret_cast<bf*>(sm+TRI)[pos(r,c)])-mu)*rs;
   float dy=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)]),v=dy*gamma[q];
   s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;
  }
  sums[r*32+lane]=s0;sums[2048+r*32+lane]=s1;
  if(rank==ACTIVE-1){s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;if(lane==0){result[r]=s0;result[64+r]=s1;}}
 }
 __syncthreads();
 if(tid==0){if(rank<ACTIVE-1)remote_arrive(bar+4,rank+1);else for(int peer=0;peer<ACTIVE;++peer)remote_arrive(bar+5,peer);}
 cluster_wait(bar+5);__syncthreads();
 for(int r=warp;r<64;r+=4){float mu=p.mu[row+r],rs=p.rs[row+r],s0=0,s1=0;
  if(lane==0){s0=remote_float(result+r,ACTIVE-1);s1=remote_float(result+64+r,ACTIVE-1);}
  s0=__shfl_sync(0xffffffff,s0,0);s1=__shfl_sync(0xffffffff,s1,0);
  #pragma unroll
  for(int q=0;q<4;++q){int c=lane+q*32;
   float z=(__bfloat162float(reinterpret_cast<bf*>(sm+TRI)[pos(r,c)])-mu)*rs;
   float dy=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)]);
   reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,fmaf(dy,gamma[q],-s0))*rs);
  }
 }
 __syncthreads();transpose<true>(sm);fence_proxy_async();__syncthreads();
 if(tid==0){for(int c=0;c<2;++c)put(&p.dt,sm+c*8192,row,col+c*64);tma_store_commit();tma_store_wait_all();}
 __syncthreads();float* parts=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<4;++q){int c=lane+q*32;parts[warp*128+c]=gg[q];parts[(4+warp)*128+c]=bb[q];}
 __syncthreads();float g=0,b=0;
 #pragma unroll
 for(int w=0;w<4;++w){g+=parts[w*128+tid];b+=parts[(4+w)*128+tid];}
 p.partial[(size_t(row/64)*2)*H+col+tid]=g;p.partial[(size_t(row/64)*2+1)*H+col+tid]=b;
}
extern "C" __global__ __launch_bounds__(128,2)
void mw_cluster_dn_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 auto cluster=cooperative_groups::this_cluster();int rank=cluster.block_rank(),tid=threadIdx.x;
 for(int c=blockIdx.x*128+tid;c<H;c+=gridDim.x*128){p.dg[c]=0;p.db[c]=0;}
 if(tid==0){for(int i=0;i<6;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();cluster.sync();
 if(rank<ACTIVE){int row=(blockIdx.x/CLSIZE)*64;gemm(p,sm,bar,row,rank*128);normalize(p,sm,bar,row,rank);}
 cluster.sync();
}
extern "C" __global__ __launch_bounds__(256,4)
void mw_cluster_dn_ln_finish(__grid_constant__ const Params p){
 __shared__ double values[512];int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 int group=blockIdx.x/8,part=blockIdx.x%8,c=group*32+lane;double g=0,b=0;
 for(int tile=part*8+warp;tile<p.M/64;tile+=64){g+=p.partial[(size_t(tile)*2)*H+c];b+=p.partial[(size_t(tile)*2+1)*H+c];}
 values[warp*32+lane]=g;values[256+warp*32+lane]=b;__syncthreads();
 if(warp==0){g=0;b=0;
  #pragma unroll
  for(int w=0;w<8;++w){g+=values[w*32+lane];b+=values[256+w*32+lane];}
  atomicAdd(p.dg+c,float(g));atomicAdd(p.db+c,float(b));
 }
}
