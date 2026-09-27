// Retain all DP in shared memory, streaming the first half of dNorm with GP.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
// MMA256
constexpr int D=512,H=1024,WP=32768,DP=163840,BAR=229376;
struct Params{CUtensorMap proj,gate,dy,ds,dp,dg,wp,dn;int M,N;float *og,*ob,*ig,*ib;};
TMN_DEVI void load_gate(const Params& p,uint8_t* sm,uint64_t* b,int row,int step){
 int col=step*64;mbar_arrive_expect_tx(b,32768);
 tma_load_2d(sm,&p.proj,b,col,row);tma_load_2d(sm+8192,&p.gate,b,col,row);
 tma_load_2d(sm+16384,&p.dy,b,col,row);tma_load_2d(sm+24576,&p.ds,b,col,row%p.N);
}
TMN_DEVI void load_wp(const Params& p,uint8_t* sm,uint64_t* b,int half,int step){
 int slot=step%2;mbar_arrive_expect_tx(b+1+slot,65536);
 tma_load_3d(sm+WP+slot*65536,&p.wp,b+1+slot,0,step*64,half*8);
}
TMN_DEVI void producer(const Params& p,uint8_t* sm,uint64_t* b,int row){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 if(tid==0){load_wp(p,sm,b,0,0);load_wp(p,sm,b,0,1);load_gate(p,sm,b,row,0);}
 for(int step=0;step<8;++step){
  mbar_wait(b,step&1);named_bar_sync(1,128);
  uint32_t saved_ds[16];
  #pragma unroll
  for(int q=0;q<4;++q)for(int j=0;j<4;++j){
   int r=warp*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j/2);
   saved_ds[q*4+j]=*reinterpret_cast<uint32_t*>(sm+24576+swz128(r,c*2));
  }
  // DS is dead after all producer threads have retained their own values.
  named_bar_sync(1,128);
  #pragma unroll
  for(int q=0;q<4;++q){uint32_t dg[4];
   #pragma unroll
   for(int j=0;j<4;++j){
    int r=warp*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j/2);uint32_t off=swz128(r,c*2);
    uint32_t pr=*reinterpret_cast<uint32_t*>(sm+off),ga=*reinterpret_cast<uint32_t*>(sm+8192+off);
    uint32_t dy=*reinterpret_cast<uint32_t*>(sm+16384+off),ds=saved_ds[q*4+j];
    float a=math::round_bf16(bf16lo(dy)*bf16lo(ds)),z=math::round_bf16(bf16hi(dy)*bf16hi(ds));
    float g0=math::sigmoid(bf16lo(ga)),g1=math::sigmoid(bf16hi(ga));
    *reinterpret_cast<uint32_t*>(sm+DP+step*8192+off)=pack_bf16(a*g0,z*g1);
    dg[j]=pack_bf16(((a*bf16lo(pr))*g0)*(1-g0),((z*bf16hi(pr))*g1)*(1-g1));
   }
   int mat=lane/8;uint32_t dst=smem_u32(sm+24576)+swz128(q*16+lane%8+8*(mat>>1),(warp*16+8*(mat&1))*2);
   stsm_x4_t(dst,dg[0],dg[1],dg[2],dg[3]);
  }
  fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dp),"r"(smem_u32(sm+DP+step*8192)),"r"(step*64),"r"(row):"memory");
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dg),"r"(smem_u32(sm+24576)),"r"(row),"r"(step*64):"memory");
   tma_store_commit();mbar_arrive(b+3+step%2);tma_store_wait_all();
   if(step+1<8)load_gate(p,sm,b,row,step+1);
   if(step>=1 && step+1<8){int prev=(step+1)%2;mbar_wait(b+5+prev,((step-1)/2)&1);load_wp(p,sm,b,0,step+1);}
  }
  named_bar_sync(1,128);
 }
 if(tid==0){
  mbar_arrive(b+7);mbar_wait(b+8,0);
  for(int step=0;step<8;++step){
   if(step>=2)mbar_wait(b+5+step%2,((8+step)/2-1)&1);
   load_wp(p,sm,b,1,step);
  }
 }
}
template<int WG> TMN_DEVI void dnorm(const Params& p,uint8_t* sm,uint64_t* b,int row){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32;
 for(int half=0;half<2;++half){
  float acc[128]={};fence_regs(acc);wgmma_fence();
  for(int step=0;step<8;++step){
   int slot=step%2;mbar_wait(b+1+slot,((half*8+step)/2)&1);
   if(half==0)mbar_wait(b+3+slot,(step/2)&1);
   named_bar_sync(2+WG,128);
   static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
    mma256<0,1>(acc,smem_desc(smem_u32(sm+DP+step*8192+q*32),16,1024,1),smem_desc(smem_u32(sm+WP+slot*65536+WG*32768+q*2048),8192,1024,1),step>0||q>0);
   });wgmma_commit();wgmma_wait<0>();
   named_bar_sync(2+WG,128);if(tid==0)mbar_arrive(b+5+slot);
  }
  fence_regs(acc);if(half==0)mbar_wait(b+7,0);named_bar_sync(4,256);
  uint8_t* out=sm+WP+WG*32768;
  static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value*2;
   int r=warp*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2);
   *reinterpret_cast<uint32_t*>(out+(c/64)*8192+swz128(r,(c%64)*2))=pack_bf16(acc[j],acc[j+1]);
  });
  fence_proxy_async();named_bar_sync(2+WG,128);
  if(tid==0){
   asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0,{0,%2,%3}],[%1];"::"l"(&p.dn),"r"(smem_u32(out)),"r"(row),"r"(half*8+WG*4):"memory");
   tma_store_commit();tma_store_wait_all();
  }
  named_bar_sync(2+WG,128);if(tid==0 && half==0)mbar_arrive(b+8);
 }
}
extern "C" __global__ __maxnreg__(CONSUMER_REGS)
void mw_d512_retained_gate_dn(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto b=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x,row=blockIdx.x*64;
 for(int c=blockIdx.x*384+tid;c<H;c+=gridDim.x*384){p.og[c]=0;p.ob[c]=0;}
 for(int c=blockIdx.x*384+tid;c<D;c+=gridDim.x*384){p.ig[c]=0;p.ib[c]=0;}
 if(tid==0){for(int i=0;i<9;++i)mbar_init(b+i,(i==5||i==6||i==8)?2:1);fence_barrier_init();}__syncthreads();
 if(tid<128){setmaxnreg_dec<PRODUCER_REGS>();producer(p,sm,b,row);}
 else{setmaxnreg_inc<CONSUMER_REGS>();if(tid<256)dnorm<0>(p,sm,b,row);else dnorm<1>(p,sm,b,row);}
}
