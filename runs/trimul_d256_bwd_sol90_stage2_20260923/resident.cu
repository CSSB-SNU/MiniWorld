// SPDX-License-Identifier: Apache-2.0
// MiniWorld D256 B1 prepare: resident output weights, register LN operand.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int D=256,H=512,WGBASE=131072,RAW=196608,BARS=229376;
struct Params{CUtensorMap tri,xn,wp,wg;const bf *dy,*ds;bf *norm,*dp,*dg;const float *gamma,*beta;float* mu;float* rs;int M,L;};
TMN_DEVI unsigned sw64(unsigned r,unsigned c){return r*64+((((c>>4)^((r>>1)&3))<<4)|(c&15));}
TMN_DEVI float sumwarp(float v){for(int s=16;s;s>>=1)v+=__shfl_xor_sync(0xffffffff,v,s);return v;}
extern "C" __global__ __launch_bounds__(256,1) void mw_d256_prepare_resident(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+BARS);uint8_t* raw=sm+RAW;
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,wg=tid/128,wiw=warp%4;
 int blockcol=(blockIdx.x%2)*128,col=blockcol+wg*64,cid=blockIdx.x/2,groups=gridDim.x/2;
 if(tid==0){mbar_init(bar,1);mbar_init(bar+1,1);fence_barrier_init();mbar_arrive_expect_tx(bar,RAW);
  for(int g=0;g<2;++g){for(int k=0;k<8;++k)tma_load_2d(sm+g*65536+k*8192,&p.wp,bar,k*64,blockcol+g*64);
   for(int k=0;k<4;++k)tma_load_2d(sm+WGBASE+g*32768+k*8192,&p.wg,bar,k*64,blockcol+g*64);}
 }__syncthreads();mbar_wait(bar,0);__syncthreads();int phase=0;
 for(int row=cid*64;row<p.M;row+=groups*64){
  uint32_t fa[32][4]={};
  for(int half=0;half<2;++half){
   if(tid==0){mbar_arrive_expect_tx(bar+1,32768);for(int c=0;c<H;c+=64)tma_load_2d(raw+(c/64)*4096,&p.tri,bar+1,row+half*32,c);}
   mbar_wait(bar+1,phase);phase^=1;__syncthreads();
   for(int kc=wg;kc<8;kc+=2){
    uint32_t f[4][4],base=smem_u32(raw+kc*4096);int mat=lane/8,r8=lane%8;
    if(wiw<2){
     #pragma unroll
     for(int q=0;q<4;++q)ldsm_x4_t(f[q],base+sw64(q*16+r8+((mat&2)?8:0),(16*wiw+((mat&1)?8:0))*2));
    }
    named_bar_sync(1+wg,128);
    if(wiw<2){
     #pragma unroll
     for(int q=0;q<4;++q)stsm_x4(base+swz128(16*wiw+r8+((mat&1)?8:0),(16*q+((mat&2)?8:0))*2),f[q][0],f[q][1],f[q][2],f[q][3]);
    }
   }__syncthreads();
   for(int r=warp;r<32;r+=8){
    float v[16],s=0;
    #pragma unroll
    for(int q=0;q<16;++q){int c=lane+q*32;v[q]=__bfloat162float(reinterpret_cast<bf*>(raw+(c/64)*4096)[swz128(r,(c%64)*2)/2]);s+=v[q];}
    float mu=sumwarp(s)/H;s=0;
    #pragma unroll
    for(int q=0;q<16;++q){float z=v[q]-mu;s+=z*z;}
    float rs=rsqrtf(sumwarp(s)/H+1e-5f);
    if(lane==0&&blockcol==0){p.mu[row+half*32+r]=mu;p.rs[row+half*32+r]=rs;}
    #pragma unroll
    for(int q=0;q<16;++q){int c=lane+q*32;bf z=__float2bfloat16_rn(fmaf((v[q]-mu)*rs,p.gamma[c],p.beta[c]));reinterpret_cast<bf*>(raw+(c/64)*4096)[swz128(r,(c%64)*2)/2]=z;
     if(blockcol==0)p.norm[size_t(row+half*32+r)*H+c]=z;
    }
   }__syncthreads();
   #pragma unroll
   for(int k=0;k<32;++k){int mat=lane/8;uint32_t addr=smem_u32(raw)+(k/4)*4096+swz128((wiw%2)*16+lane%8+((mat&1)?8:0),((k%4)*16+((mat&2)?8:0))*2);
    asm volatile("{.reg .pred p;setp.eq.s32 p,%5,%6;@p ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];}" : "+r"(fa[k][0]),"+r"(fa[k][1]),"+r"(fa[k][2]),"+r"(fa[k][3]) : "r"(addr),"r"(wiw/2),"r"(half));
   }
   __syncthreads();
  }
  float acc[32]={};fence_regs(acc);wgmma_fence();
  uint64_t desc=smem_desc(smem_u32(sm+wg*65536),16,1024,1);uint32_t lo=desc,hi=desc>>32;
  static_for<32>([&](auto kk){constexpr int k=decltype(kk)::value;wgmma_m64n64k16_rs_off<(k/4)*8192+(k%4)*32>(acc,fa[k],lo,hi,k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);
  uint32_t proj[16];
  #pragma unroll
  for(int j=0;j<16;++j)proj[j]=pack_bf16(acc[j*2],acc[j*2+1]);
  if(tid==0){mbar_arrive_expect_tx(bar+1,32768);for(int c=0;c<D;c+=64)tma_load_2d(raw+(c/64)*8192,&p.xn,bar+1,c,row);}
  mbar_wait(bar+1,phase);phase^=1;__syncthreads();
  auto& xn=*reinterpret_cast<uint32_t(*)[16][4]>(fa);load_frag_bf16<16,8192>(xn,smem_u32(raw),wiw*16,lane);
  desc=smem_desc(smem_u32(sm+WGBASE+wg*32768),16,1024,1);lo=desc;hi=desc>>32;
  fence_regs(acc);wgmma_fence();static_for<16>([&](auto kk){constexpr int k=decltype(kk)::value;wgmma_m64n64k16_rs_off<(k/4)*8192+(k%4)*32>(acc,xn[k],lo,hi,k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);
  #pragma unroll
  for(int j=0;j<32;++j){int r=wiw*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2)+col;size_t ix=size_t(row+r)*D+c;
   float v=(j%2)?bf16hi(proj[j/2]):bf16lo(proj[j/2]),g=math::sigmoid(math::round_bf16(acc[j]));float dy=math::round_bf16(__bfloat162float(p.dy[ix])*__bfloat162float(p.ds[((row+r)%p.L)*D+c]));
   p.dp[ix]=__float2bfloat16_rn(dy*g);p.dg[ix]=__float2bfloat16_rn(((dy*v)*g)*(1-g));
  }__syncthreads();
 }
}
