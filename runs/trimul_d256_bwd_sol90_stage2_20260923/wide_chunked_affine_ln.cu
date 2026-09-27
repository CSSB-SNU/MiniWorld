#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=2*WIDTH,ROWS=16,NT=128,SB=ROWS*H*2,Q=AFFINE_CHUNK,CH=32*Q;
struct Params {CUtensorMap tri,dt,dn;const float *mu,*rs,*gamma;float* partial;int M;};
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
// TRANSPOSE_HELPERS
TMN_DEVI int pos(int r,int c){return (c/64)*(ROWS*64)+swz128(r,(c%64)*2)/2;}
extern "C" __global__ __launch_bounds__(NT,MIN_BLOCKS)
void mw_wide_chunked_affine_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+2*SB);
 float* gamma=reinterpret_cast<float*>(sm+2*SB+128);
 float* stats=gamma+H;float* partial=p.partial+blockIdx.x*2*H;
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 for(int c=tid;c<H;c+=NT)gamma[c]=p.gamma[c];
 for(int c=tid;c<2*H;c+=NT)partial[c]=0;
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();
 for(int row=blockIdx.x*ROWS,it=0;row<p.M;row+=gridDim.x*ROWS,++it){
  if(tid==0){mbar_arrive_expect_tx(bar,2*SB);
   for(int c=0;c<H;c+=64){
    tma_load_2d(sm+(c/64)*(ROWS*128),&p.tri,bar,row,c);
    tma_load_2d(sm+SB+(c/64)*(ROWS*128),&p.dn,bar,c,row);
   }
  }
  mbar_wait(bar,it&1);__syncthreads();transpose32<false>(sm);
  for(int r=warp;r<ROWS;r+=4){
   float mu=p.mu[row+r],rs=p.rs[row+r],s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<H/32;++q){int c=lane+q*32;
    float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]),v=dy*gamma[c];
    s0+=v;s1+=v*z;
   }
   s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
   if(lane==0){stats[r]=s0;stats[ROWS+r]=s1;}
  }
  __syncthreads();
  // The consumed dNorm channel slab becomes affine-reduction scratch.
  // ROWS=16 makes its BF16 footprint exactly 2*4*CH FP32 values.
  #pragma unroll 1
  for(int base=0;base<H/32;base+=Q){
   float gg[Q]={},bb[Q]={};
   for(int r=warp;r<ROWS;r+=4){
    float mu=p.mu[row+r],rs=p.rs[row+r],s0=stats[r],s1=stats[ROWS+r];
    #pragma unroll
    for(int q=0;q<Q;++q){int c=lane+(base+q)*32;
     float z=(__bfloat162float(reinterpret_cast<bf*>(sm)[pos(r,c)])-mu)*rs;
     float dy=__bfloat162float(reinterpret_cast<bf*>(sm+SB)[pos(r,c)]);
     gg[q]+=dy*z;bb[q]+=dy;
     reinterpret_cast<bf*>(sm)[pos(r,c)]=__float2bfloat16_rn(fmaf(-z,s1,fmaf(dy,gamma[c],-s0))*rs);
    }
   }
   __syncthreads();
   float* scratch=reinterpret_cast<float*>(sm+SB+(base/2)*(ROWS*128));
   #pragma unroll
   for(int q=0;q<Q;++q){int c=lane+q*32;scratch[warp*CH+c]=gg[q];scratch[(4+warp)*CH+c]=bb[q];}
   __syncthreads();
   for(int c=tid;c<CH;c+=NT){float g=0,b=0;
    #pragma unroll
    for(int w=0;w<4;++w){g+=scratch[w*CH+c];b+=scratch[(4+w)*CH+c];}
    partial[base*32+c]+=g;partial[H+base*32+c]+=b;
   }
   __syncthreads();
  }
  transpose32<true>(sm);fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<H;c+=64)put_tile(&p.dt,sm+(c/64)*(ROWS*128),row,c);tma_store_commit();tma_store_wait_all();}
  __syncthreads();
 }
}
struct Reduce {const float* partial;float *dg,*db;int blocks;};
extern "C" __global__ __launch_bounds__(128,4)
void mw_wide_chunked_affine_reduce(__grid_constant__ const Reduce p){
 int c=blockIdx.x*128+threadIdx.x;if(c>=H)return;float g=0,b=0;
 for(int i=0;i<p.blocks;++i){g+=p.partial[i*2*H+c];b+=p.partial[i*2*H+H+c];}
 p.dg[c]=g;p.db[c]=b;
}
