#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>
#include <cuda_bf16.h>
#include "../core_tiles/r1/csrc/fa3_utils.h"

namespace SOL_NAMESPACE {
using namespace cute;
using Element=cutlass::bfloat16_t;
constexpr int M=64,N=SOL_N,LN=64,Ratio=LN/N,D=32,Rows=SOL_ROWS,Stages=2;
constexpr int Consumers=Rows*128,Threads=Consumers+SOL_PRODUCER_THREADS;
using SQ=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));
using SK=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<Int<LN>,_32>{}));
using SV=decltype(composition(SK{},make_ordered_layout(Shape<_32,Int<LN>>{},Step<_2,_1>{})));
using SB=Layout<Shape<_256,Int<N/4>>,Stride<_1,_256>>;
using GB=Shape<int,int,int,int,int>;
using DB=Stride<_1,_256,int64_t,int64_t,int64_t>;
using S1=decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<Element>{},Shape<_8,Int<N>>{}));
using G3=Shape<int,int,int>;
using ST=Stride<int64_t,_1,int64_t>;
using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),G3{},ST{}),SQ{},Shape<_64,_32>{},_1{}));
using TK=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),G3{},ST{}),SK{},Shape<Int<LN>,_32>{},_1{}));
using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((float const*)nullptr),GB{},DB{}),SB{},Shape<_256,Int<N/4>>{},_1{}));
using QK=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,Int<N>,_32>>()));
using PV=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));
using LS=decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>{}));
struct Params { TQ q;TK k,v;TB bias;Element* out;int* fix;int* valid;int L;float scale; };
struct Shared {
 alignas(128) Element q[Rows][M*D],k[Stages][Rows][LN*D],v[Stages][Rows][LN*D],ones[8*N];
 alignas(128) float bias[4][M*N];
 cutlass::arch::ClusterTransactionBarrier qr,full[Stages],bf[4];
 cutlass::arch::ClusterBarrier empty[Stages],be[4];int bad;
};
__device__ __forceinline__ float ex2(float x) { float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y; }

template<bool Safe> __global__ __launch_bounds__(Threads,SOL_RESIDENCY)
void attention(CUTE_GRID_CONSTANT Params const p) {
 extern __shared__ __align__(128) unsigned char memory[];
 auto& s=*reinterpret_cast<Shared*>(memory);
 int tid=threadIdx.x,nq=p.L/M,nk=p.L/N,nr=(p.L+Rows-1)/Rows;
 if(!*p.valid)return;
 int tile=blockIdx.x;
 if constexpr(Safe) { if(tile>=p.fix[0])return;tile=p.fix[1+tile]; }
 int qt=tile%nq,rg=tile/nq%nr,h=tile/(nq*nr),i0=rg*Rows;
 if(tid==0) {
  s.qr.init(1);for(int st=0;st<4;st++){s.bf[st].init(1);s.be[st].init(Rows*4);}s.bad=0;
  for(int st=0;st<Stages;st++){s.full[st].init(1);s.empty[st].init(Rows*4);}
  cutlass::arch::fence_barrier_init();
 }
 for(int x=tid;x<8*N;x+=Threads)s.ones[x]=Element(1.f);
 cutlass::arch::fence_view_async_shared();__syncthreads();
 if(tid>=Consumers) {
  if constexpr(SOL_REG_CONSUMER>0)cutlass::arch::warpgroup_reg_dealloc<SOL_REG_PRODUCER>();
  if(tid==Consumers) {
   s.qr.arrive_and_expect_tx(Rows*M*D*sizeof(Element));
   for(int r=0;r<Rows;r++) {
    auto src=local_tile(p.q.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,min(i0+r,p.L-1)*4+h),Shape<_64,_32>{},make_coord(qt,0));
    auto dst=make_tensor(make_smem_ptr(s.q[r]),SQ{});auto sl=p.q.get_slice(_0{});
    copy(p.q.with(reinterpret_cast<uint64_t&>(s.qr)),sl.partition_S(src),sl.partition_D(dst));
   }
   for(int seq=0;seq<p.L/LN;seq++) {
    int st=seq%Stages;
    if(seq>=Stages){s.empty[st].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
    s.full[st].arrive_and_expect_tx(2*Rows*LN*D*sizeof(Element));
    for(int r=0;r<Rows;r++) {
     int ih=min(i0+r,p.L-1)*4+h;
     auto ks=local_tile(p.k.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto vs=local_tile(p.v.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{}),vd=make_tensor(make_smem_ptr(s.v[st][r]),SK{});
     auto kl=p.k.get_slice(_0{}),vl=p.v.get_slice(_0{});
     copy(p.k.with(reinterpret_cast<uint64_t&>(s.full[st])),kl.partition_S(ks),kl.partition_D(kd));
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[st])),vl.partition_S(vs),vl.partition_D(vd));
    }

   }
  }
  if(tid==Consumers+32) {
   for(int seq=0;seq<p.L/N;seq++) {
    int st=seq%4;
    if(seq>=4){s.be[st].wait(((seq/4)-1)&1);asm volatile("":::"memory");}
    s.bf[st].arrive_and_expect_tx(M*N*sizeof(float));
    auto bs=p.bias.get_tma_tensor(make_shape(256,N/4,p.L/N,nq,4))(_,_,seq,qt,h);
    auto bd=make_tensor(make_smem_ptr(s.bias[st]),SB{});auto bl=p.bias.get_slice(_0{});
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bf[st])),bl.partition_S(bs),bl.partition_D(bd));
   }
  }
  return;
 }
 if constexpr(SOL_REG_CONSUMER>0)cutlass::arch::warpgroup_reg_alloc<SOL_REG_CONSUMER>();
 int wg=tid/128,t=tid%128,lane=t%32,wm=t/32;
 float acc[4][4]={},sum[2]={},mx[2]={Safe?-INFINITY:0.f,Safe?-INFINITY:0.f};
 bool missing_seed=false;
 float c=p.scale*1.4426950408889634f;
 uint32_t qa[2][4];
 auto ld4=[](uint32_t (&r)[4],Element const* ptr) __attribute__((always_inline)) {
  uint32_t a=__cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
 };
 auto ld2=[](uint32_t (&r)[2],Element const* ptr) __attribute__((always_inline)) {
  uint32_t a=__cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];":"=r"(r[0]),"=r"(r[1]):"r"(a));
 };
 auto ldt2=[](uint32_t (&r)[2],Element const* ptr) __attribute__((always_inline)) {
  uint32_t a=__cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];":"=r"(r[0]),"=r"(r[1]):"r"(a));
 };
 auto mma=[](float (&o)[4],uint32_t const* a,uint32_t const* b) __attribute__((always_inline)) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(o[0]),"+f"(o[1]),"+f"(o[2]),"+f"(o[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
 };
 s.qr.wait(0);asm volatile("":::"memory");
 #pragma unroll
 for(int kk=0;kk<2;kk++)ld4(qa[kk],s.q[wg]+SQ{}(make_coord(wm*16+lane%16,16*kk+8*(lane/16))));
 #pragma unroll 1
 for(int seq=0;seq<nk;seq++) {
  int kt=seq/Ratio,st=kt%Stages,cx=seq%Ratio;
  if(cx==0){s.full[st].wait((kt/Stages)&1);asm volatile("":::"memory");}
  s.bf[seq%4].wait((seq/4)&1);asm volatile("":::"memory");
  float sc[4][4];
  #pragma unroll
  for(int n=0;n<4;n++){
   float4 b=*reinterpret_cast<float4 const*>(s.bias[seq%4]+n*512+t*4);
   sc[n][0]=b.x;sc[n][1]=b.y;sc[n][2]=b.z;sc[n][3]=b.w;
  }
  #pragma unroll
  for(int kk=0;kk<2;kk++) {
   #pragma unroll
   for(int n=0;n<4;n++){
    uint32_t kb[2];
    ld2(kb,s.k[st][wg]+SK{}(make_coord(cx*N+n*8+lane%8,kk*16+8*((lane/8)%2))));
    mma(sc[n],qa[kk],kb);
   }
  }
  __syncwarp();if(lane==0)s.be[seq%4].arrive();
  #pragma unroll
  for(int row=0;row<2;row++) {
   if(Safe || seq==0) {
    float m=sc[0][row*2];
    #pragma unroll
    for(int n=0;n<4;n++){m=fmaxf(m,sc[n][row*2]);m=fmaxf(m,sc[n][row*2+1]);}
    m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));m*=c;
    if constexpr(Safe){
     float next=fmaxf(mx[row],m),alpha=mx[row]==-INFINITY?0.f:ex2(mx[row]-next);
     #pragma unroll
     for(int n=0;n<4;n++){acc[n][row*2]*=alpha;acc[n][row*2+1]*=alpha;}
     sum[row]*=alpha;mx[row]=next;
    }else{if(m!=-INFINITY)mx[row]=m+64.f;else missing_seed=true;}
   }
   #pragma unroll
   for(int n=0;n<4;n++){
    sc[n][row*2]=ex2(fmaf(sc[n][row*2],c,-mx[row]));
    sc[n][row*2+1]=ex2(fmaf(sc[n][row*2+1],c,-mx[row]));
   }
  }
  uint32_t pp[8];
  #pragma unroll
  for(int n=0;n<4;n++){
   #pragma unroll
   for(int row=0;row<2;row++){
    auto packed=__floats2bfloat162_rn(sc[n][row*2],sc[n][row*2+1]);
    pp[n*2+row]=reinterpret_cast<uint32_t const&>(packed);
    float2 fp=__bfloat1622float2(packed);sum[row]+=fp.x+fp.y;
   }
  }
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   #pragma unroll
   for(int n=0;n<4;n++){
    uint32_t vb[2];
    ldt2(vb,s.v[st][wg]+SK{}(make_coord(cx*N+kk*16+lane%16,n*8)));
    mma(acc[n],pp+kk*4,vb);
   }
  }
  __syncwarp();if(cx==Ratio-1 && lane==0)s.empty[st].arrive();
 }
 #pragma unroll
 for(int row=0;row<2;row++){
  sum[row]+=__shfl_xor_sync(0xffffffff,sum[row],1);sum[row]+=__shfl_xor_sync(0xffffffff,sum[row],2);
 }
 if constexpr(!Safe){
  bool bad=missing_seed;
  #pragma unroll
  for(int row=0;row<2;row++)bad|=!(sum[row]>0x1p-84f && sum[row]<0x1p-44f);
  if(__any_sync(0xffffffff,bad) && lane==0 && atomicCAS(&s.bad,0,1)==0){int index=atomicAdd(p.fix,1);p.fix[index+1]=tile;}
 }
 #pragma unroll
 for(int row=0;row<2;row++){
  float inv=1.f/sum[row];int q=qt*M+wm*16+lane/4+row*8;
  #pragma unroll
  for(int n=0;n<4;n++){
   auto x=__floats2bfloat162_rn(acc[n][row*2]*inv,acc[n][row*2+1]*inv);
   int d=n*8+(lane%4)*2;
   if(i0+wg<p.L)*reinterpret_cast<uint32_t*>(p.out+((int64_t(i0+wg)*4+h)*p.L+q)*32+d)=reinterpret_cast<uint32_t const&>(x);
  }
 }
}
__global__ void prepare(float const* bias,bool const* mask,int64_t stride,float inv,int L,float* out,int* valid,int* fix){
 int kt=blockIdx.x,qt=blockIdx.y,h=blockIdx.z,nk=L/LN,nq=L/M;
 for(int idx=threadIdx.x;idx<M*LN;idx+=256){
  int e=idx&3,t=(idx>>2)&127,u=idx>>9;
  int q=qt*M+16*(t>>5)+(t&31)/4+8*(e/2),k=kt*LN+8*u+2*(t%4)+e%2;
  out[((int64_t(h)*nq+qt)*nk+kt)*M*LN+idx]=(!mask || mask[int64_t(k)*stride])?bias[(int64_t(h)*L+q)*L+k]*inv:-INFINITY;
 }
 if(kt==0 && qt==0 && h==0){
  bool any=!mask;for(int k=threadIdx.x;k<L;k+=256)any|=mask && mask[int64_t(k)*stride];
  int yes=__syncthreads_or(any);if(threadIdx.x==0){*valid=yes;*fix=0;}
 }
}
__global__ void uniform(Element const* v,Element* out,int L,int const* valid){
 if(*valid)return;int i=blockIdx.x,h=blockIdx.y,d=threadIdx.x%32,g=threadIdx.x/32;__shared__ float part[4][32];
 float sum=0;for(int k=g;k<L;k+=4)sum+=float(v[((int64_t(i)*4+h)*L+k)*32+d]);part[g][d]=sum;__syncthreads();
 Element mean=Element((part[0][d]+part[1][d]+part[2][d]+part[3][d])/float(L));
 for(int q=g;q<L;q+=4)out[((int64_t(i)*4+h)*L+q)*32+d]=mean;
}
}
at::Tensor sol_forward(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor bias,c10::optional<at::Tensor> mask,double scale){
 using namespace SOL_NAMESPACE;int L=q.size(-2);
 TORCH_CHECK((L==384 || L==768 || L==1024) && q.numel()==int64_t(L)*4*L*32 && q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous(),"C128/H4/D32 contiguous square BF16 operands required");
 for(auto const& x:{k,v})TORCH_CHECK(x.device()==q.device() && x.sizes()==q.sizes() && x.scalar_type()==q.scalar_type() && x.is_contiguous(),"unsupported K/V");
 TORCH_CHECK(bias.device()==q.device() && bias.scalar_type()==at::kFloat && bias.is_contiguous() && bias.numel()==4*L*L,"contiguous float bias required");
 bool const* mp=nullptr;int64_t ms=1;
 if(mask){auto m=*mask;TORCH_CHECK(m.device()==q.device() && m.scalar_type()==at::kBool && m.dim()==5 && m.size(0)==1 && m.size(1)==L && m.size(2)==1 && m.size(3)==1 && m.size(4)==L && m.stride(1)==0,"row-broadcast mask required");mp=m.data_ptr<bool>();ms=m.stride(4);}
 c10::cuda::CUDAGuard guard(q.device());
 auto output=at::empty_like(q),prepared=at::empty_like(bias),fix=at::empty({1+(L/M)*((L+Rows-1)/Rows)*4},q.options().dtype(at::kInt)),valid=at::empty({1},fix.options());
 auto stream=at::cuda::getCurrentCUDAStream();
 prepare<<<dim3(L/LN,L/M,4),256,0,stream>>>(bias.data_ptr<float>(),mp,ms,float(1./scale),L,prepared.data_ptr<float>(),valid.data_ptr<int>(),fix.data_ptr<int>());
 auto g=[&](at::Tensor x){return make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),make_shape(L,32,L*4),ST{32,_1{},int64_t(L)*32});};
 auto tq=make_tma_copy(SM90_TMA_LOAD{},g(q),SQ{},Shape<_64,_32>{},_1{});
 auto tk=make_tma_copy(SM90_TMA_LOAD{},g(k),SK{},Shape<Int<LN>,_32>{},_1{});
 auto tv=make_tma_copy(SM90_TMA_LOAD{},g(v),SK{},Shape<Int<LN>,_32>{},_1{});
 auto tb=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(prepared.data_ptr<float>()),make_shape(256,N/4,L/N,L/M,4),DB{_1{},_256{},M*N,int64_t(L/N)*M*N,int64_t(L/M)*(L/N)*M*N}),SB{},Shape<_256,Int<N/4>>{},_1{});
 Params p{tq,tk,tv,tb,(Element*)output.data_ptr(),fix.data_ptr<int>(),valid.data_ptr<int>(),L,float(scale)};
 int ct=(L/M)*((L+Rows-1)/Rows)*4;
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);
 attention<true><<<ct,Threads,sizeof(Shared),stream>>>(p);
 uniform<<<dim3(L,4),128,0,stream>>>((Element const*)v.data_ptr(),(Element*)output.data_ptr(),L,valid.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
