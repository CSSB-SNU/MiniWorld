// SPDX-License-Identifier: Apache-2.0
// MiniWorld D256 output-LN gradient: TMA load/store and warpgroup transposes.
#include "tmn_kernels.cuh"
#include <cooperative_groups.h>
using namespace tmn;using namespace tmn::sm90;using bf=__nv_bfloat16;
constexpr int H=512,NT=256,SB=65536,DN=65536,WB=131072,BARS=163840;
struct Params {CUtensorMap tri,dt,dp,wp;const bf* dn;const float *mu,*rs,*gamma;float* dg;float* db;int M;};
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v+=__shfl_xor_sync(0xffffffff,v,q);return v;}
// This operation is its own inverse: [c,r] <-> [r,c] within each 64x64 tile.
TMN_DEVI void transpose(uint8_t* sm){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 for(int kc=tid/128;kc<8;kc+=2){
  uint32_t f[4][4],base=smem_u32(sm+kc*8192);int mat=lane/8,r8=lane%8,wiw=warp%4;
  #pragma unroll
  for(int q=0;q<4;++q)ldsm_x4_t(f[q],base+swz128(16*q+r8+((mat&2)?8:0),(16*wiw+((mat&1)?8:0))*2));
  named_bar_sync(1+tid/128,128);
  #pragma unroll
  for(int q=0;q<4;++q){int r=16*wiw+r8+((mat&1)?8:0),c=16*q+((mat&2)?8:0);stsm_x4(base+swz128(r,c*2),f[q][0],f[q][1],f[q][2],f[q][3]);}
 }
 __syncthreads();
}
TMN_DEVI void store_tile(const CUtensorMap* map,uint8_t* sm,int row,int c){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(map),"r"(smem_u32(sm)),"r"(row),"r"(c):"memory");
}
template <int OFF_BYTES>
TMN_DEVI void mma_rs_trans(float (&d)[32], const uint32_t (&a)[4], uint32_t desc_lo, uint32_t desc_hi, int scale_d) {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    ".reg .b32 lo;\n"
    ".reg .b64 dsc;\n"
    "setp.ne.b32 p, %38, 0;\n"
    "add.u32 lo, %36, %39;\n"
    "mov.b64 dsc, {lo, %37};\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31},"
    "{%32, %33, %34, %35}, dsc, p, 1, 1, 1;\n"
    "}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(desc_lo), "r"(desc_hi), "r"(scale_d), "n"(OFF_BYTES >> 4));
}

TMN_DEVI void dnorm(const Params& p,int row,uint8_t* sm,uint64_t* bars,int& phase){
 int tid=threadIdx.x,wg=tid/128,lane=tid%32,warp=tid/32;
 if(tid==0){mbar_arrive_expect_tx(bars+1,32768);for(int k=0;k<4;++k)tma_load_2d(sm+WB+k*8192,&p.dp,bars+1,k*64,row);}
 mbar_wait(bars+1,(row/(gridDim.x*64))&1);__syncthreads();
 uint32_t fa[16][4];load_frag_bf16<16,8192>(fa,smem_u32(sm+WB),(warp%4)*16,lane);__syncthreads();
 for(int col=0;col<H;col+=128){
  auto load=[&](int k,int slot){mbar_arrive_expect_tx(bars+2+slot,16384);for(int g=0;g<2;++g)tma_load_2d(sm+WB+slot*16384+g*8192,&p.wp,bars+2+slot,col+g*64,k);};
  if(tid==0)load(0,0);float v[32]={};
  #pragma unroll
  for(int k=0;k<4;++k){int slot=k%2;
   if(tid==0&&k<3)load((k+1)*64,1-slot);
   mbar_wait(bars+2+slot,(phase>>slot)&1);phase^=1<<slot;__syncthreads();
   uint64_t desc=smem_desc(smem_u32(sm+WB+slot*16384+wg*8192),16,1024,1);uint32_t lo=desc,hi=desc>>32;
   fence_regs(v);wgmma_fence();
   static_for<4>([&](auto qq){constexpr int q=decltype(qq)::value;mma_rs_trans<q*2048>(v,fa[k*4+q],lo,hi,k>0||q>0);});wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
  }
  for(int j=0;j<32;++j){int r=(warp%4)*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2);reinterpret_cast<bf*>(sm+DN+(col/64+wg)*8192)[swz128(r,c*2)/2]=__float2bfloat16_rn(v[j]);}
  __syncthreads();
 }
}
extern "C" __global__ __launch_bounds__(NT,1) void mw_d256_dn_ln(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto bar=reinterpret_cast<uint64_t*>(sm+BARS);int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 for(int c=blockIdx.x*NT+tid;c<H;c+=gridDim.x*NT){p.dg[c]=0;p.db[c]=0;}
 if(tid==0){for(int i=0;i<4;++i)mbar_init(bar+i,1);fence_barrier_init();}__syncthreads();cooperative_groups::this_grid().sync();
 float gg[16]={},bb[16]={};int phase=0,wp_phase=0;
 for(int row=blockIdx.x*64;row<p.M;row+=gridDim.x*64){
  if(tid==0){mbar_arrive_expect_tx(bar,SB);for(int c=0;c<H;c+=64)tma_load_2d(sm+c*128,&p.tri,bar,row,c);}
  mbar_wait(bar,phase);phase^=1;__syncthreads();
  dnorm(p,row,sm,bar,wp_phase);transpose(sm);
  for(int r=warp;r<64;r+=8){
   float mu=p.mu[row+r],rs=p.rs[row+r],z[16],v[16],s0=0,s1=0;
   #pragma unroll
   for(int q=0;q<16;++q){int c=lane+q*32;
    z[q]=(__bfloat162float(reinterpret_cast<bf*>(sm+(c/64)*8192)[swz128(r,(c%64)*2)/2])-mu)*rs;
    float dy=__bfloat162float(reinterpret_cast<bf*>(sm+DN+(c/64)*8192)[swz128(r,(c%64)*2)/2]);v[q]=dy*p.gamma[c];s0+=v[q];s1+=v[q]*z[q];gg[q]+=dy*z[q];bb[q]+=dy;
   }
   s0=sumwarp(s0)/H;s1=sumwarp(s1)/H;
   #pragma unroll
   for(int q=0;q<16;++q){int c=lane+q*32;float dy=__bfloat162float(reinterpret_cast<bf*>(sm+DN+(c/64)*8192)[swz128(r,(c%64)*2)/2]);float centered=fmaf(dy,p.gamma[c],-s0);reinterpret_cast<bf*>(sm+(c/64)*8192)[swz128(r,(c%64)*2)/2]=__float2bfloat16_rn(fmaf(-z[q],s1,centered)*rs);}
  }
  __syncthreads();transpose(sm);fence_proxy_async();__syncthreads();
  if(tid==0){for(int c=0;c<H;c+=64)store_tile(&p.dt,sm+c*128,row,c);tma_store_commit();tma_store_wait_all();}
  __syncthreads();
 }
 for(int q=0;q<16;++q){atomicAdd(p.dg+lane+q*32,gg[q]);atomicAdd(p.db+lane+q*32,bb[q]);}
}
