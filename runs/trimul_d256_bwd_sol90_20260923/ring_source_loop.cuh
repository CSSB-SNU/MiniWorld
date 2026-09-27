// SPDX-License-Identifier: Apache-2.0
// MiniWorld D256 B7: recompute derivatives and immediately accumulate dW.
// Adapted from our D128 producer-local dW schedule; D256 uses shared operands.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;
using bf=__nv_bfloat16;
#include "mma.cuh"
constexpr int D=256,H=512,CH=32768,INPUT=36864,WEIGHT=73728,DERIV=106496;
struct Params {CUtensorMap xn,w,dl,dr;const bf* mask;bf *gp[4];float* part;int M;CUtensorMap ringmap,dg,wgate,weights[4];bf* dxn;bf* ring;unsigned* flags;int cohorts;};
constexpr int SOURCES=32,CONSUMERS=8,GROUP=40,RINGS=8;
TMN_DEVI void wait_flag(unsigned* ptr,unsigned want){unsigned got;do{asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(got):"l"(ptr):"memory");if(got<want)__nanosleep(32);}while(got<want);}
TMN_DEVI void signal_flag(unsigned* ptr,unsigned val){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(ptr),"r"(val):"memory");}

TMN_DEVI float rd(const bf* x,size_t i){return __bfloat162float(x[i]);}
TMN_DEVI void load_input(const Params& p,uint8_t* sm,uint64_t* bar,int row,int rank,int slot){
 mbar_arrive_expect_tx(bar+slot,INPUT);
 for(int c=0;c<4;++c)tma_load_2d(sm+slot*INPUT+c*8192,&p.xn,bar+slot,c*64,row);
 tma_load_2d(sm+slot*INPUT+CH,rank<16?&p.dl:&p.dr,bar+slot,row,(rank%16)*32);
}
TMN_DEVI void ring_source(const Params& p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+114688);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,rank=blockIdx.x%GROUP,cid=blockIdx.x/GROUP,split=cid*4;
 int tiles=p.M/64,begin=0,end=(tiles-1-cid)/p.cohorts+1,segment=end/4;
 if(tid==0){
  for(int b=0;b<3;++b)mbar_init(bar+b,1);fence_barrier_init();
  mbar_arrive_expect_tx(bar+2,CH);
  for(int c=0;c<4;++c)tma_load_2d(sm+WEIGHT+c*8192,&p.w,bar+2,c*64,rank*64);
  load_input(p,sm,bar,cid*64,rank,0);
 }
 __syncthreads();mbar_wait(bar+2,0);__syncthreads();
 float dw0[64]={},dw1[64]={};
 for(int tile=begin,it=0;tile<end;++tile,++it){
  int slot=it%2,row=(cid+tile*p.cohorts)*64;unsigned* flags=p.flags+(cid*RINGS+it%RINGS)*33;
  if(it>=RINGS&&tid==0)wait_flag(flags+32,it-RINGS+1);__syncthreads();uint8_t* xn=sm+slot*INPUT;
  if(tid==0&&tile+1<end)load_input(p,sm,bar,(cid+(tile+1)*p.cohorts)*64,rank,1-slot);
  mbar_wait(bar+slot,(it/2)&1);__syncthreads();
  float pre[32]={};fence_regs(pre);wgmma_fence();
  #pragma unroll 1
  for(int k=0;k<16;++k){
   mma64<0,0>(pre,smem_desc(smem_u32(xn+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+WEIGHT+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  }wgmma_commit();wgmma_wait<0>();fence_regs(pre);
  // Shared derivative matrix [gate32,projection32] x token64, swizzle128.
  for(int q=0;q<2;++q)for(int j=0;j<8;++j){
   int index=q*8+j,r=warp*16+lane/4+8*((index/2)&1),c=q*16+2*(lane%4)+8*((index/2)%4/2)+(index%2);
   float da=math::round_bf16(rd(reinterpret_cast<bf*>(xn+CH),swz128(c,r*2)/2)*rd(p.mask,row+r));
   float g=math::sigmoid(math::round_bf16(pre[index])),v=math::round_bf16(pre[index+16]);
   bf dg=__float2bfloat16_rn(((da*v)*g)*(1-g)),dp=__float2bfloat16_rn(da*g);
   reinterpret_cast<bf*>(sm+DERIV)[swz128(c,r*2)/2]=dg;
   reinterpret_cast<bf*>(sm+DERIV)[swz128(c+32,r*2)/2]=dp;
   int side=rank/16,col=(rank%16)*32+c;
   p.ring[((size_t(cid)*RINGS+it%RINGS)*4*H+side*2*H+col)*64+r]=dp;
   p.ring[((size_t(cid)*RINGS+it%RINGS)*4*H+(side*2+1)*H+col)*64+r]=dg;
  }
  __threadfence();__syncthreads();if(tid==0)signal_flag(flags+rank,it+1);fence_proxy_async();__syncthreads();
  fence_regs(dw0);fence_regs(dw1);wgmma_fence();
  #pragma unroll 1
  for(int k=0;k<4;++k){
   uint64_t a=smem_desc(smem_u32(sm+DERIV+k*32),16,1024,1);
   mma128<0,1>(dw0,a,smem_desc(smem_u32(xn+k*2048),8192,1024,1),(it%segment)>0||k>0);
   mma128<0,1>(dw1,a,smem_desc(smem_u32(xn+16384+k*2048),8192,1024,1),(it%segment)>0||k>0);
  }wgmma_commit();wgmma_wait<0>();fence_regs(dw0);fence_regs(dw1);__syncthreads();
 if((it+1)%segment==0){
 for(int j=0;j<64;++j){
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
  int which=(rank/16)*2+(r<32?1:0),outrow=(rank%16)*32+r%32;
  size_t ix=size_t(split+it/segment)*11*D*D+(3+2*which)*D*D+outrow*D+c;
  p.part[ix]=dw0[j];p.part[ix+128]=dw1[j];
 }
 for(int j=0;j<64;++j){dw0[j]=0;dw1[j]=0;}
 }
 }
}
