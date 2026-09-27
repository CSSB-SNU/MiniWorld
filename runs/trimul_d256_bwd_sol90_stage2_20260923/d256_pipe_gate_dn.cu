// Exact gate derivatives feed two dNorm WGMMA groups through two shared slots.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
// MMA256
constexpr int D=256,H=512,SLOT=106496,BAR=2*SLOT;
struct Params{CUtensorMap proj,gate,dy,ds,dp,dg,wp,dn;int M,N;float *og,*ob,*ig,*ib;};
TMN_DEVI void load_stage(const Params& p,uint8_t* sm,uint64_t* b,int row,int step){
 int slot=step%2,col=step*64;uint8_t* buf=sm+slot*SLOT;
 mbar_arrive_expect_tx(b+slot,98304);
 tma_load_2d(buf,&p.proj,b+slot,col,row);tma_load_2d(buf+8192,&p.gate,b+slot,col,row);
 tma_load_2d(buf+16384,&p.dy,b+slot,col,row);tma_load_2d(buf+24576,&p.ds,b+slot,col,row%p.N);
 tma_load_3d(buf+32768,&p.wp,b+slot,0,col,0);
}
TMN_DEVI void gate_stage(const Params& p,uint8_t* sm,uint64_t* b,int row){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 if(tid==0){load_stage(p,sm,b,row,0);load_stage(p,sm,b,row,1);}
 for(int step=0;step<4;++step){
  int slot=step%2,col=step*64;uint8_t* buf=sm+slot*SLOT;
  mbar_wait(b+slot,(step/2)&1);named_bar_sync(1,128);
  #pragma unroll
  for(int q=0;q<4;++q){uint32_t dg[4];
   #pragma unroll
   for(int j=0;j<4;++j){
    int r=warp*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j/2);uint32_t off=swz128(r,c*2);
    uint32_t pr=*reinterpret_cast<uint32_t*>(buf+off),ga=*reinterpret_cast<uint32_t*>(buf+8192+off);
    uint32_t dy=*reinterpret_cast<uint32_t*>(buf+16384+off),ds=*reinterpret_cast<uint32_t*>(buf+24576+off);
    float a=math::round_bf16(bf16lo(dy)*bf16lo(ds)),z=math::round_bf16(bf16hi(dy)*bf16hi(ds));
    float g0=math::sigmoid(bf16lo(ga)),g1=math::sigmoid(bf16hi(ga));
    *reinterpret_cast<uint32_t*>(buf+off)=pack_bf16(a*g0,z*g1);
    dg[j]=pack_bf16(((a*bf16lo(pr))*g0)*(1-g0),((z*bf16hi(pr))*g1)*(1-g1));
   }
   int mat=lane/8;uint32_t dst=smem_u32(buf+98304)+swz128(q*16+lane%8+8*(mat>>1),(warp*16+8*(mat&1))*2);
   stsm_x4_t(dst,dg[0],dg[1],dg[2],dg[3]);
  }
  fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dp),"r"(smem_u32(buf)),"r"(col),"r"(row):"memory");
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dg),"r"(smem_u32(buf+98304)),"r"(row),"r"(col):"memory");
   tma_store_commit();mbar_arrive(b+2+slot);tma_store_wait_all();
   if(step>=1 && step+1<4){int previous=1-slot;mbar_wait(b+4+previous,((step-1)/2)&1);load_stage(p,sm,b,row,step+1);}
  }
  named_bar_sync(1,128);
 }
}
template<int WG> TMN_DEVI void dnorm(const Params& p,uint8_t* sm,uint64_t* b,int row){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;float acc[128]={};
 fence_regs(acc);wgmma_fence();
 for(int step=0;step<4;++step){
  int slot=step%2;uint8_t* buf=sm+slot*SLOT;
  mbar_wait(b+2+slot,(step/2)&1);named_bar_sync(2+WG,128);
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   mma256<0,1>(acc,smem_desc(smem_u32(buf+q*32),16,1024,1),smem_desc(smem_u32(buf+32768+WG*32768+q*2048),8192,1024,1),step>0||q>0);
  });wgmma_commit();wgmma_wait<0>();
  named_bar_sync(2+WG,128);if(tid==0)mbar_arrive(b+4+slot);
 }
 fence_regs(acc);named_bar_sync(4,256);
 uint8_t* out=sm+WG*32768;
 static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value*2;
  int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
  *reinterpret_cast<uint32_t*>(out+(c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(acc[j],acc[j+1]);
 });
 fence_proxy_async();named_bar_sync(2+WG,128);
 if(tid==0){
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{0,%2,%3}],[%1];"::"l"(&p.dn),"r"(smem_u32(out)),"r"(row),"r"(WG*4):"memory");
  tma_store_commit();tma_store_wait_all();
 }
}
extern "C" __global__ __maxnreg__(CONSUMER_REGS)
void mw_d256_pipe_gate_dn(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto b=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x,row=blockIdx.x*64;
 for(int c=blockIdx.x*384+tid;c<H;c+=gridDim.x*384){p.og[c]=0;p.ob[c]=0;}
 for(int c=blockIdx.x*384+tid;c<D;c+=gridDim.x*384){p.ig[c]=0;p.ib[c]=0;}
 if(tid==0){for(int i=0;i<6;++i)mbar_init(b+i,i>=4?2:1);fence_barrier_init();}__syncthreads();
 if(tid<128){setmaxnreg_dec<PRODUCER_REGS>();gate_stage(p,sm,b,row);}
 else{setmaxnreg_inc<CONSUMER_REGS>();if(tid<256)dnorm<0>(p,sm,b,row);else dnorm<1>(p,sm,b,row);}
}
