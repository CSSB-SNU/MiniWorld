// SPDX-License-Identifier: Apache-2.0
// MiniWorld wide training forward. Uses the engine's TMA/WGMMA primitives.
// Keep the output-LN tile on chip; stream weight K64 chunks instead of
// allocating whole gate/projection blocks. No global normalized-triangle buffer.
#include "tmn_kernels.cuh"
using namespace tmn;
using namespace tmn::sm90;
using bf = __nv_bfloat16;
constexpr int D=WIDTH,H=2*D,G=GROUPS,NT=128*G;
constexpr int XBYTES=H*128,STAGE=8192*(G+1);
struct Params {CUtensorMap tri,xn,wp,wg; const bf *x,*ds; bf *y; const float *gamma,*beta; int M,L;};
TMN_DEVI float rd(const bf* p,int i){return __bfloat162float(p[i]);}
TMN_DEVI float sumwarp(float v){for(int q=16;q;q>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,q));return v;}
TMN_DEVI void mma(float (&v)[32],uint64_t a,uint64_t b,int ac){
 asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]) : "l"(a),"l"(b),"r"(ac));
}
template<bool GATE> TMN_DEVI void product(const Params& p,float (&v)[32],int row,int col,uint8_t* sx,uint8_t* buf,uint64_t* bars,int& phase){
 constexpr int K=GATE?D:H;
 auto load=[&](int k,int slot){
  mbar_arrive_expect_tx(bars+slot,8192*(G+(GATE?1:0)));
  for(int g=0;g<G;++g)tma_load_2d(buf+slot*STAGE+8192*g,GATE?&p.wg:&p.wp,bars+slot,k,col+64*g);
  if constexpr(GATE)tma_load_2d(buf+slot*STAGE+8192*G,&p.xn,bars+slot,k,row);
 };
 for(int j=0;j<32;++j)v[j]=0;
 if(threadIdx.x==0)load(0,0);
 for(int k=0,it=0;k<K;k+=64,++it){
  int slot=it&1;
  if(k+64<K&&threadIdx.x==0)load(k+64,1-slot);
  mbar_wait(bars+slot,(phase>>slot)&1);phase^=1<<slot;__syncthreads();
  uint8_t* a=GATE?buf+slot*STAGE+8192*G:sx+(k/64)*8192;
  uint8_t* b=buf+slot*STAGE+8192*(threadIdx.x/128);
  fence_regs(v);wgmma_fence();
  #pragma unroll
  for(int q=0;q<4;++q)mma(v,smem_desc(smem_u32(a+32*q),16,1024,1),smem_desc(smem_u32(b+32*q),16,1024,1),k>0||q>0);
  wgmma_commit();wgmma_wait<0>();fence_regs(v);__syncthreads();
 }
}
extern "C" __global__ __launch_bounds__(NT,1) void mw_wide_output(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 uint8_t* sx=sm;uint8_t* buf=sm+XBYTES;
 float* gb=reinterpret_cast<float*>(buf+2*STAGE);
 float* stats=gb+2*H;uint64_t* bar=reinterpret_cast<uint64_t*>(stats+128);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 for(int c=tid;c<H;c+=NT){gb[c]=p.gamma[c];gb[H+c]=p.beta[c];}
 if(tid==0){mbar_init(bar,1);mbar_init(bar+1,1);mbar_init(bar+2,1);fence_barrier_init();}
 __syncthreads();int phase=0,xphase=0;
 for(int row=blockIdx.x*64;row<p.M;row+=gridDim.x*64){
  if(tid==0){mbar_arrive_expect_tx(bar+2,XBYTES);for(int c=0;c<H;c+=64)tma_load_2d(sx+c*128,&p.tri,bar+2,row,c);}
  mbar_wait(bar+2,xphase);xphase^=1;__syncthreads();
  // Read channel-major raw triangle; all rows' statistics precede in-place transpose.
  for(int r=warp;r<64;r+=NT/32){
   float s=0;for(int c=lane;c<H;c+=32)s+=rd(reinterpret_cast<bf*>(sx),swz128(c,r*2)/2);
   float mu=sumwarp(s)/H;s=0;
   for(int c=lane;c<H;c+=32){float z=rd(reinterpret_cast<bf*>(sx),swz128(c,r*2)/2)-mu;s+=z*z;}
   float rs=rsqrtf(sumwarp(s)/H+1e-5f);if(lane==0){stats[r]=mu;stats[64+r]=rs;}
  }
  __syncthreads();
  for(int c0=0;c0<H;c0+=64){
   bf* tmp=reinterpret_cast<bf*>(buf);
   for(int i=tid;i<4096;i+=NT){int c=i/64,r=i%64;float z=rd(reinterpret_cast<bf*>(sx),swz128(c0+c,r*2)/2);tmp[swz128(r,c*2)/2]=__float2bfloat16_rn(fmaf((z-stats[r])*stats[64+r],gb[c0+c],gb[H+c0+c]));}
   __syncthreads();for(int i=tid;i<4096;i+=NT)reinterpret_cast<bf*>(sx+c0*128)[i]=tmp[i];__syncthreads();
  }
  fence_proxy_async();__syncthreads();
  for(int col=0;col<D;col+=64*G){
   float proj[32],gate[32];product<false>(p,proj,row,col,sx,buf,bar,phase);product<true>(p,gate,row,col,sx,buf,bar,phase);
   #pragma unroll
   for(int j=0;j<32;++j){int r=((tid/32)%4)*16+lane/4+8*((j/2)&1),c=(j/8)*16+2*(lane%4)+8*((j/2)%4/2)+(j%2)+col+64*(tid/128);if(c<D){size_t ix=size_t(row+r)*D+c;float v=math::round_bf16(proj[j])*math::sigmoid(math::round_bf16(gate[j]));p.y[ix]=__float2bfloat16_rn(fmaf(v,rd(p.ds,((row+r)%p.L)*D+c),__bfloat162float(p.x[ix])));}}
  }
  __syncthreads();
 }
}
