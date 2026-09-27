// SPDX-License-Identifier: Apache-2.0
// Split output projection dW, exact BF16 operands and FP32 partial sums.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPER
constexpr int D=WIDTH,H=2*D,STAGE=32768,SLOTS=3,BAR=STAGE*SLOTS;
struct Params{CUtensorMap a,b,out;const float* partial;bf* result;int M;};
TMN_DEVI void load(const Params& p,uint8_t* sm,uint64_t* bar,int rm,int cn,int ki,int it){
 int slot=it%SLOTS;mbar_arrive_expect_tx(bar+slot,STAGE);
 tma_load_3d(sm+slot*STAGE,&p.a,bar+slot,0,ki,rm*2);
 tma_load_3d(sm+slot*STAGE+16384,&p.b,bar+slot,0,ki,cn*2);
}
extern "C" __global__ __launch_bounds__(256,2)
void mw_wide_native_projection_dw(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,wg=threadIdx.x/128;
 int tiles=(D/128)*(H/128),split=blockIdx.x/tiles,tile=blockIdx.x%tiles;
 int rm=tile/(H/128),cn=tile%(H/128),steps=p.M/64/DW_SPLITS,begin=split*steps*64;
 if(threadIdx.x==0){for(int i=0;i<SLOTS;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();
 if(threadIdx.x==0){load(p,sm,bar,rm,cn,begin,0);load(p,sm,bar,rm,cn,begin+64,1);}
 float v[64]={};
 for(int it=0;it<steps;++it){
  int slot=it%SLOTS;mbar_wait(bar+slot,(it/SLOTS)&1);__syncthreads();
  if(threadIdx.x==0 && it+2<steps)load(p,sm,bar,rm,cn,begin+(it+2)*64,it+2);
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   mma128_off<k*2048,k*2048,1,1>(v,smem_desc(smem_u32(sm+slot*STAGE+wg*8192),16,1024,1),smem_desc(smem_u32(sm+slot*STAGE+16384),8192,1024,1),it>0||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);
 }
 __syncthreads();uint8_t* out=sm+wg*32768;
 static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  *reinterpret_cast<float2*>(out+(c/32)*8192+swz128(r,(c%32)*4))=make_float2(v[j],v[j+1]);
 });
 fence_proxy_async();__syncthreads();
 if(tid==0){
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{0,%2,%3}],[%1];"::"l"(&p.out),"r"(smem_u32(out)),"r"(split*D+rm*128+wg*64),"r"(cn*4):"memory");
  tma_store_commit();tma_store_wait_all();
 }
}
extern "C" __global__ void mw_wide_native_projection_dw_reduce(__grid_constant__ const Params p){
 int i=(blockIdx.x*blockDim.x+threadIdx.x)*2;if(i>=D*H)return;
 float a=0,b=0;
 #pragma unroll
 for(int s=0;s<DW_SPLITS;++s){float2 q=*reinterpret_cast<const float2*>(p.partial+size_t(s)*D*H+i);a+=q.x;b+=q.y;}
 *reinterpret_cast<uint32_t*>(p.result+i)=pack_bf16(a,b);
}
