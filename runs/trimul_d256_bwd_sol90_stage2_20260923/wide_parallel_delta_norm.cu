// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=1024,NT=128,ROWS=16,SB=ROWS*H*2,GAM=SB+128,STATS=GAM+H*8;
struct Params{CUtensorMap tri,norm;uint64_t* patches;unsigned* count;unsigned capacity;unsigned* changed;const float *gamma,*beta;float *mu,*rs;int M;};
// TRANSPOSE_HELPERS
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));return v;}
TMN_DEVI int pos(int r,int c){return(c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
TMN_DEVI void packed(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;const float* gamma=reinterpret_cast<float*>(sm+GAM);const float* beta=gamma+H;const float* stats=reinterpret_cast<float*>(sm+STATS);
 for(int row=blockIdx.x*ROWS,it=0;row<p.M;row+=gridDim.x*ROWS,++it){
  if(tid==0){mbar_arrive_expect_tx(bar,SB);for(int c=0;c<H;c+=64)tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar,row,c);}
  mbar_wait(bar,it&1);named_bar_sync(1,128);transpose32<false>(sm);
  if(tid==0)mbar_arrive(bar+1);
  float means[4],scales[4];
  #pragma unroll
  for(int rr=0;rr<4;++rr){int r=warp+rr*4;float s=0;
   #pragma unroll
   for(int c=2*lane;c<H;c+=64){uint32_t v=*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));s+=bf16lo(v)+bf16hi(v);}
   float mu=sumwarp(s)/H;s=0;
   #pragma unroll
   for(int c=2*lane;c<H;c+=64){uint32_t v=*reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));float a=bf16lo(v)-mu,b=bf16hi(v)-mu;s+=a*a+b*b;}
   means[rr]=mu;scales[rr]=rsqrtf(sumwarp(s)/H+1e-5f);
  }
  mbar_wait(bar+2,it&1);named_bar_sync(1,128);
  #pragma unroll
  for(int rr=0;rr<4;++rr){int r=warp+rr*4;float mu=means[rr],rs=scales[rr],smu=stats[r],srs=stats[ROWS+r];
   if(__float_as_uint(mu)==__float_as_uint(smu)&&__float_as_uint(rs)==__float_as_uint(srs)){
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){auto ptr=reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));uint32_t v=*ptr;
     *ptr=pack_bf16(fmaf((bf16lo(v)-mu)*rs,gamma[c],beta[c]),fmaf((bf16hi(v)-mu)*rs,gamma[c+1],beta[c+1]));
    }
   }else{
    #pragma unroll
    for(int c=2*lane;c<H;c+=64){auto ptr=reinterpret_cast<uint32_t*>(reinterpret_cast<bf*>(sm)+pos(r,c));uint32_t v=*ptr;
     float g0=gamma[c],g1=gamma[c+1],b0=beta[c],b1=beta[c+1];
     uint32_t a=pack_bf16(fmaf((bf16lo(v)-mu)*rs,g0,b0),fmaf((bf16hi(v)-mu)*rs,g1,b1));
     uint32_t b=pack_bf16(fmaf((bf16lo(v)-smu)*srs,g0,b0),fmaf((bf16hi(v)-smu)*srs,g1,b1));*ptr=a;
     if(a!=b){unsigned at=atomicAdd(p.count,1u);if(at<p.capacity)p.patches[at]=(uint64_t(b)<<32)|uint32_t(((row+r)*H+c)/2);atomicExch(p.changed+(row+r)/64,1u);}
    }
   }
  }
  fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){for(int c=0;c<H;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.norm),"r"(smem_u32(sm+(c/64)*(ROWS*128))),"r"(c),"r"(row):"memory");
  }tma_store_commit();tma_store_wait_all();}named_bar_sync(1,128);
 }
}
TMN_DEVI void scalar(const Params& p,uint8_t* sm,uint64_t* bar){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;float* stats=reinterpret_cast<float*>(sm+STATS);
 for(int row=blockIdx.x*ROWS,it=0;row<p.M;row+=gridDim.x*ROWS,++it){
  mbar_wait(bar+1,it&1);named_bar_sync(2,128);
  for(int r=warp;r<ROWS;r+=4){float v[H/32],s=0;
   #pragma unroll
   for(int q=0;q<H/32;++q){v[q]=__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,lane+q*32)]);s+=v[q];}
   float mu=sumwarp(s)/H;s=0;
   #pragma unroll
   for(int q=0;q<H/32;++q){float z=v[q]-mu;s+=z*z;}
   float rs=rsqrtf(sumwarp(s)/H+1e-5f);
   if(lane==0){stats[r]=mu;stats[ROWS+r]=rs;p.mu[row+r]=mu;p.rs[row+r]=rs;}
  }
  named_bar_sync(2,128);if(tid==0)mbar_arrive(bar+2);
 }
}
extern "C" __global__ __launch_bounds__(256,DELTA_MINBLOCKS)
void mw_wide_parallel_delta_norm(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+SB);int tid=threadIdx.x;
 for(int c=tid;c<H;c+=256){reinterpret_cast<float*>(sm+GAM)[c]=p.gamma[c];reinterpret_cast<float*>(sm+GAM+H*4)[c]=p.beta[c];}
 if(tid==0){for(int i=0;i<3;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 if(tid<128){setmaxnreg_inc<DELTA_PACKED_REGS>();packed(p,sm,bar);}
 else{setmaxnreg_dec<DELTA_SCALAR_REGS>();scalar(p,sm,bar);}
}
