// Ordered prefix dX: two compute warpgroups, one row tile per CTA.
// GEMM accumulators die before LN affine accumulators become live.
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
// MMA_HELPERS
constexpr int D=WIDTH,NHALF=D/2,AC=NHALF/2,SLOTS=DX_SLOTS;
constexpr int STAGE=(1+D/64)*8192,DN_BYTES=64*D*2,XS=16*D*2;
constexpr int BAR=STAGE*SLOTS>DN_BYTES+2*XS?STAGE*SLOTS:DN_BYTES+2*XS;
struct Params{CUtensorMap gp,w,x,res,out;const float* gamma;float* affine;float* dg;float* db;
 const float* weights;bf* dw[4];bf* dwg;int M;};
TMN_DEVI int pos64(int r,int c){return (c/64)*4096+swz128(r,(c%64)*2)/2;}
TMN_DEVI int pos16(int r,int c){return (c/64)*1024+swz128(r,(c%64)*2)/2;}
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
TMN_DEVI void load_gemm(const Params& p,uint8_t* sm,uint64_t* bar,int row,int step){
 int slot=step%SLOTS;uint8_t* buf=sm+slot*STAGE;
 mbar_arrive_expect_tx(bar+slot,STAGE);tma_load_2d(buf,&p.gp,bar+slot,row,step*64);
 for(int c=0;c<D;c+=64)tma_load_2d(buf+8192+(c/64)*8192,&p.w,bar+slot,c,step*64);
}
TMN_DEVI void gemm(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 int tid=threadIdx.x%128,lane=tid%32,warp=tid/32,wg=threadIdx.x/128;
 float v[AC]={};constexpr int STEPS=9*D/64;
 if(threadIdx.x==0){load_gemm(p,sm,bar,row,0);load_gemm(p,sm,bar,row,1);}
 for(int step=0;step<STEPS;++step){int slot=step%SLOTS;uint8_t* buf=sm+slot*STAGE;
  mbar_wait(bar+slot,(step/SLOTS)&1);__syncthreads();
  if constexpr(SLOTS==3){if(threadIdx.x==0 && step+2<STEPS)load_gemm(p,sm,bar,row,step+2);}
  fence_regs(v);wgmma_fence();
  static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;
   uint64_t a=smem_desc(smem_u32(buf+q*2048),16,1024,1);
   uint64_t b=smem_desc(smem_u32(buf+8192+wg*(NHALF/64)*8192+q*2048),8192,1024,1);
   // WIDE_MMA
  });wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
  if constexpr(SLOTS==2){if(threadIdx.x==0 && step+2<STEPS)load_gemm(p,sm,bar,row,step+2);}
 }
 static_for<AC>([&](auto jj){constexpr int j=decltype(jj)::value;
  int r=warp*16+lane/4+8*((j/2)&1),c=wg*NHALF+(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);
  reinterpret_cast<bf*>(sm)[pos64(r,c)]=__float2bfloat16_rn(v[j]);
 });__syncthreads();
}
TMN_DEVI void input_ln(const Params& p,uint8_t* sm,uint64_t* bar,int row){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 float gg[D/32]={},bb[D/32]={},gamma[D/32];
 #pragma unroll
 for(int q=0;q<D/32;++q)gamma[q]=p.gamma[lane+q*32];
 for(int tile=0;tile<4;++tile){
  if(tid==0){mbar_arrive_expect_tx(bar+SLOTS,2*XS);
   for(int c=0;c<D;c+=64){
    tma_load_2d(sm+DN_BYTES+(c/64)*2048,&p.x,bar+SLOTS,c,row+tile*16);
    tma_load_2d(sm+DN_BYTES+XS+(c/64)*2048,&p.res,bar+SLOTS,c,row+tile*16);
   }
  }
  mbar_wait(bar+SLOTS,tile&1);__syncthreads();
  for(int r=warp;r<16;r+=8){float xv[D/32],s=0;
   #pragma unroll
   for(int q=0;q<D/32;++q){xv[q]=__bfloat162float(reinterpret_cast<bf*>(sm+DN_BYTES)[pos16(r,lane+q*32)]);s+=xv[q];}
   float mu=sumwarp(s)/D;s=0;
   #pragma unroll
   for(int q=0;q<D/32;++q){float z=xv[q]-mu;s+=z*z;}
   float rs=rsqrtf(sumwarp(s)/D+1e-5f),s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<D/32;++q){int c=lane+q*32;float z=(xv[q]-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm)[pos64(tile*16+r,c)]),v=dy*gamma[q];
    s0+=v;s1+=v*z;gg[q]+=dy*z;bb[q]+=dy;
   }
   s0=sumwarp(s0)/D;s1=sumwarp(s1)/D;
   #pragma unroll
   for(int q=0;q<D/32;++q){int c=lane+q*32;float z=(xv[q]-mu)*rs;
    float v=__bfloat162float(reinterpret_cast<bf*>(sm)[pos64(tile*16+r,c)])*gamma[q];
    float dx=(v-s0-z*s1)*rs;
    float residual=__bfloat162float(reinterpret_cast<bf*>(sm+DN_BYTES+XS)[pos16(r,c)]);
    reinterpret_cast<bf*>(sm+DN_BYTES)[pos16(r,c)]=__float2bfloat16_rn(dx+residual);
   }
  }
  __syncthreads();fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<D;c+=64){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.out),"r"(smem_u32(sm+DN_BYTES+(c/64)*2048)),"r"(c),"r"(row+tile*16):"memory");
  }tma_store_commit();tma_store_wait_all();}
  __syncthreads();
 }
 float* tmp=reinterpret_cast<float*>(sm);
 #pragma unroll
 for(int q=0;q<D/32;++q){int c=lane+q*32;tmp[warp*D+c]=gg[q];tmp[(8+warp)*D+c]=bb[q];}
 __syncthreads();
 for(int c=tid;c<D;c+=256){float g=0,b=0;
  #pragma unroll
  for(int w=0;w<8;++w){g+=tmp[w*D+c];b+=tmp[(8+w)*D+c];}
  p.affine[(size_t(blockIdx.x)*2)*D+c]=g;p.affine[(size_t(blockIdx.x)*2+1)*D+c]=b;
 }
}
extern "C" __global__ __launch_bounds__(256,DX_MINBLOCKS)
void mw_prefix_tile_dx_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x;
 for(int c=blockIdx.x*256+tid;c<D;c+=gridDim.x*256){p.dg[c]=0;p.db[c]=0;}
 if(tid==0){for(int i=0;i<=SLOTS;++i)mbar_init(bar+i,1);fence_barrier_init();}
 __syncthreads();int row=blockIdx.x*64;gemm(p,sm,bar,row);input_ln(p,sm,bar,row);
}
extern "C" __global__ __launch_bounds__(256,4)
void mw_prefix_tile_dx_finish(__grid_constant__ const Params p){
 __shared__ double values[512];int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 int group=blockIdx.x/8,part=blockIdx.x%8,c=group*32+lane;double g=0,b=0;
 for(int tile=part*8+warp;tile<p.M/64;tile+=64){g+=p.affine[(size_t(tile)*2)*D+c];b+=p.affine[(size_t(tile)*2+1)*D+c];}
 values[warp*32+lane]=g;values[256+warp*32+lane]=b;__syncthreads();
 if(warp==0){g=0;b=0;
  #pragma unroll
  for(int w=0;w<8;++w){g+=values[w*32+lane];b+=values[256+w*32+lane];}
  atomicAdd(p.dg+c,float(g));atomicAdd(p.db+c,float(b));
 }
 if constexpr(JOINT_INPUT){for(int i=blockIdx.x*256+tid;i<D*D;i+=gridDim.x*256){float v=0;
  #pragma unroll
  for(int s=0;s<WEIGHT_SPLITS;++s)v+=p.weights[size_t(s)*11*D*D+2*D*D+i];
  p.dwg[i]=__float2bfloat16_rn(v);
 }}
 for(int which=0;which<4;++which){for(int i=blockIdx.x*256+tid;i<2*D*D;i+=gridDim.x*256){float v=0;
  #pragma unroll
  for(int s=0;s<WEIGHT_SPLITS;++s)v+=p.weights[size_t(s)*11*D*D+(3+2*which)*D*D+i];
  p.dw[which][i]=__float2bfloat16_rn(v);
 }}
}
