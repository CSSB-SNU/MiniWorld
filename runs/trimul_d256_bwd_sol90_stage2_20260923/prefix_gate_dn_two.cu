// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
constexpr int D=WIDTH,H=2*D,DPBASE=40960,BAR=DPBASE+64*D*2;
struct Params{CUtensorMap proj,gate,dy,ds,dp,dg,wp,dn;int M,N;};
// MMA_HELPERS
TMN_DEVI void load_weight(const Params& p,uint8_t* sm,uint64_t* bar,int col,int ki,int slot){
 mbar_arrive_expect_tx(bar+1+slot,16384);
 for(int g=0;g<4;++g)tma_load_2d(sm+slot*16384+g*4096,&p.wp,bar+1+slot,col+64*g,ki);
}
extern "C" __global__ __launch_bounds__(256,2)
void mw_prefix_gate_dn_two(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x,lane=tid%32,warp=(tid/32)%4,wg=tid/128;
 if(tid==0){for(int i=0;i<3;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();
 for(int row=blockIdx.x*64;row<p.M;row+=gridDim.x*64){
  for(int col=0;col<D;col+=64){
   if(tid==0){
    mbar_arrive_expect_tx(bar,32768);
    tma_load_2d(sm,&p.proj,bar,col,row);tma_load_2d(sm+8192,&p.gate,bar,col,row);
    tma_load_2d(sm+16384,&p.dy,bar,col,row);tma_load_2d(sm+24576,&p.ds,bar,col,row%p.N);
   }
   mbar_wait(bar,(col/64)&1);__syncthreads();
   #pragma unroll
   for(int q=wg*2;q<wg*2+2;++q){uint32_t dg[4];
    #pragma unroll
    for(int j=0;j<4;++j){
     int r=warp*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j/2);uint32_t off=swz128(r,c*2);
     uint32_t pr=*reinterpret_cast<uint32_t*>(sm+off),ga=*reinterpret_cast<uint32_t*>(sm+8192+off);
     uint32_t dy=*reinterpret_cast<uint32_t*>(sm+16384+off),ds=*reinterpret_cast<uint32_t*>(sm+24576+off);
     float a=math::round_bf16(bf16lo(dy)*bf16lo(ds)),b=math::round_bf16(bf16hi(dy)*bf16hi(ds));
     float g0=math::sigmoid(bf16lo(ga)),g1=math::sigmoid(bf16hi(ga));
     *reinterpret_cast<uint32_t*>(sm+DPBASE+(col/64)*8192+off)=pack_bf16(a*g0,b*g1);
     dg[j]=pack_bf16(((a*bf16lo(pr))*g0)*(1-g0),((b*bf16hi(pr))*g1)*(1-g1));
    }
    int mat=lane/8;uint32_t dst=smem_u32(sm+32768)+swz128(q*16+lane%8+8*(mat>>1),(warp*16+8*(mat&1))*2);
    stsm_x4_t(dst,dg[0],dg[1],dg[2],dg[3]);
   }
   fence_proxy_async();__syncthreads();
   if(tid==0){
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dp),"r"(smem_u32(sm+DPBASE+(col/64)*8192)),"r"(col),"r"(row):"memory");
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dg),"r"(smem_u32(sm+32768)),"r"(row),"r"(col):"memory");
    tma_store_commit();tma_store_wait_read<0>();
   }__syncthreads();
  }
  for(int col=0;col<H;col+=256){
   float v[64]={};if(tid==0)load_weight(p,sm,bar,col,0,0);
   for(int ki=0,it=0;ki<D;ki+=32,++it){
    int slot=it%2;mbar_wait(bar+1+slot,((col/128)*(D/128)+it/2)&1);__syncthreads();
    if(tid==0 && ki+32<D)load_weight(p,sm,bar,col,ki+32,1-slot);
    fence_regs(v);wgmma_fence();
    static_for<2>([&](auto qq){constexpr int q=decltype(qq)::value;
     mma128_off<q*32,q*2048,0,1>(v,smem_desc(smem_u32(sm+DPBASE+(ki/64)*8192+(ki%64)*2),16,1024,1),smem_desc(smem_u32(sm+slot*16384+wg*8192),4096,1024,1),it>0||q>0);
    });wgmma_commit();wgmma_wait<0>();fence_regs(v);
   }
   __syncthreads();
   static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value*2;
    int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
    *reinterpret_cast<uint32_t*>(sm+wg*16384+(c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(v[j],v[j+1]);
   });fence_proxy_async();__syncthreads();
   if(tid==0){for(int c=0;c<256;c+=64){
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dn),"r"(smem_u32(sm+(c/64)*8192)),"r"(col+c),"r"(row):"memory");
   }tma_store_commit();tma_store_wait_read<0>();}__syncthreads();
  }
 }
 if(tid==0)tma_store_wait_all();
}
