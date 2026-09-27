// SPDX-License-Identifier: Apache-2.0
// MiniWorld D256 B1: LN/projection/gate and local dW accumulation.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
#include "mma.cuh"
constexpr int D=256,H=512,SX=65536,XN=65536,WB=98304,DP=114688,DG=122880,BARS=131072;
struct Params {CUtensorMap tri,xn,wp,wg;const bf* dy;const bf* ds;const float* gamma;const float* beta;bf* dp;bf* dg;float *mu,*rs,*part;int M,L;};
TMN_DEVI float rd(const bf* p,size_t i){return __bfloat162float(p[i]);}
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));return v;}
template<bool GATE> TMN_DEVI void product(const Params& p,float (&v)[32],int col,uint8_t* sm,uint64_t* bars,int& phase){
 constexpr int K=GATE?D:H;
 auto load=[&](int k,int slot){mbar_arrive_expect_tx(bars+slot,8192);tma_load_2d(sm+WB+slot*8192,GATE?&p.wg:&p.wp,bars+slot,k,col);};
 if(threadIdx.x==0)load(0,0);
 for(int k=0,it=0;k<K;k+=64,++it){int slot=it%2;
  if(threadIdx.x==0&&k+64<K)load(k+64,1-slot);
  mbar_wait(bars+slot,(phase>>slot)&1);phase^=1<<slot;__syncthreads();
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma64<0,0>(v,smem_desc(smem_u32(sm+(GATE?XN:0)+(k/64)*8192+q*32),16,1024,1),smem_desc(smem_u32(sm+WB+slot*8192+q*32),16,1024,1),k>0||q>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
 }
}
extern "C" __global__ __launch_bounds__(256,1) void mw_d256_b1_source(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+BARS);uint8_t* sx=sm;
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,wg=tid/128;
 int col=(blockIdx.x%4)*64,split=blockIdx.x/4,tiles=p.M/64;
 int begin=tiles*split/WEIGHT_SPLITS,end=tiles*(split+1)/WEIGHT_SPLITS;
 if(tid==0){for(int i=0;i<3;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();int phase=0,xphase=0;
 float dwp0[64]={},dwp1[64]={},dwg[64]={};
 for(int tile=begin;tile<end;++tile){int row=tile*64;
  if(tid==0){mbar_arrive_expect_tx(bar+2,SX+32768);
   for(int c=0;c<H;c+=64)tma_load_2d(sx+c*128,&p.tri,bar+2,row,c);
   for(int c=0;c<D;c+=64)tma_load_2d(sm+XN+c*128,&p.xn,bar+2,c,row);
  }mbar_wait(bar+2,xphase);xphase^=1;__syncthreads();
#include "b1_ln.inc"
  fence_proxy_async();__syncthreads();
  float pre[32]={};product<false>(p,pre,col,sm,bar,phase);
  if(wg==0)for(int j=0;j<32;++j){int r=(warp%4)*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);reinterpret_cast<bf*>(sm+DP)[swz128(c,r*2)/2]=__float2bfloat16_rn(pre[j]);}
  __syncthreads();product<true>(p,pre,col,sm,bar,phase);
  if(wg==0)for(int j=0;j<32;++j){int r=(warp%4)*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);size_t ix=size_t(row+r)*D+col+c;
   float v=rd(reinterpret_cast<bf*>(sm+DP),swz128(c,r*2)/2),g=math::sigmoid(math::round_bf16(pre[j]));
   float dy=math::round_bf16(rd(p.dy,ix)*rd(p.ds,((row+r)%p.L)*D+col+c));
   bf dp=__float2bfloat16_rn(dy*g),dg=__float2bfloat16_rn(((dy*v)*g)*(1-g));
   p.dp[ix]=dp;p.dg[ix]=dg;reinterpret_cast<bf*>(sm+DP)[swz128(c,r*2)/2]=dp;reinterpret_cast<bf*>(sm+DG)[swz128(c,r*2)/2]=dg;
  }__syncthreads();fence_proxy_async();__syncthreads();
  fence_regs(dwp0);fence_regs(dwp1);fence_regs(dwg);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
   uint64_t a=smem_desc(smem_u32(sm+DP+k*32),16,1024,1),g=smem_desc(smem_u32(sm+DG+k*32),16,1024,1);
   mma128<0,1>(dwp0,a,smem_desc(smem_u32(sm+wg*32768+k*2048),8192,1024,1),tile>begin||k>0);
   mma128<0,1>(dwp1,a,smem_desc(smem_u32(sm+wg*32768+16384+k*2048),8192,1024,1),tile>begin||k>0);
   mma128<0,1>(dwg,g,smem_desc(smem_u32(sm+XN+wg*16384+k*2048),8192,1024,1),tile>begin||k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(dwp0);fence_regs(dwp1);fence_regs(dwg);__syncthreads();
 }
 for(int j=0;j<64;++j){int r=(warp%4)*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
  size_t base=size_t(split)*11*D*D;
  p.part[base+(col+r)*H+wg*256+c]=dwp0[j];p.part[base+(col+r)*H+wg*256+c+128]=dwp1[j];
  p.part[base+2*D*D+(col+r)*D+wg*128+c]=dwg[j];
 }
}
