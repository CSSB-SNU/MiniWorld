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
constexpr int M=64,N=SOL_N,D=32,Rows=SOL_ROWS,Stages=3;
constexpr int Consumers=Rows*128,Threads=Consumers+SOL_PRODUCER_THREADS;
using SQ=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));
using SK=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<Int<N>,_32>{}));
using SV=decltype(composition(SK{},make_ordered_layout(Shape<_32,Int<N>>{},Step<_2,_1>{})));
using SB=Layout<Shape<_256,Int<N/4>>,Stride<_1,_256>>;
using GB=Shape<int,int,int,int,int>;
using DB=Stride<_1,_256,int64_t,int64_t,int64_t>;
using S1=decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<Element>{},Shape<_8,Int<N>>{}));
using G3=Shape<int,int,int>;
using ST=Stride<int64_t,_1,int64_t>;
using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),G3{},ST{}),SQ{},Shape<_64,_32>{},_1{}));
using TK=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),G3{},ST{}),SK{},Shape<Int<N>,_32>{},_1{}));
using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((float const*)nullptr),GB{},DB{}),SB{},Shape<_256,Int<N/4>>{},_1{}));
using QK=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,Int<N>,_32>>()));
using PV=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));
using LS=decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>{}));
struct Params { TQ q;TK k,v;TB bias;Element* out;int* fix;int* valid;int L;float scale; };
struct Shared {
 alignas(128) Element q[Rows][M*D],k[Stages][Rows][N*D],v[Stages][Rows][N*D],ones[8*N];
 alignas(128) float bias[Stages][M*N];
 cutlass::arch::ClusterTransactionBarrier qr,full[Stages];
 cutlass::arch::ClusterBarrier empty[Stages];int bad;
};
__device__ __forceinline__ float ex2(float x) { float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y; }

template<bool Safe> __global__ __launch_bounds__(Threads,SOL_RESIDENCY)
void attention(CUTE_GRID_CONSTANT Params const p) {
 extern __shared__ __align__(128) unsigned char memory[];
 auto& s=*reinterpret_cast<Shared*>(memory);
 int tid=threadIdx.x,nq=p.L/M,nr=(p.L+Rows-1)/Rows;constexpr int nk=768/N;
 if(!*p.valid)return;
 int tile=blockIdx.x;
 if constexpr(Safe) { if(tile>=p.fix[0])return;tile=p.fix[1+tile]; }
 int qt=tile%nq,rg=tile/nq%nr,h=tile/(nq*nr),i0=rg*Rows;
 if(tid==0) {
  s.qr.init(1);s.bad=0;
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
   for(int seq=0;seq<nk;seq++) {
    int st=seq%Stages;
    if(seq>=Stages){s.empty[st].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
    s.full[st].arrive_and_expect_tx(2*Rows*N*D*sizeof(Element)+M*N*sizeof(float));
    for(int r=0;r<Rows;r++) {
     int ih=min(i0+r,p.L-1)*4+h;
     auto ks=local_tile(p.k.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,ih),Shape<Int<N>,_32>{},make_coord(seq,0));
     auto vs=local_tile(p.v.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,ih),Shape<Int<N>,_32>{},make_coord(seq,0));
     auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{}),vd=make_tensor(make_smem_ptr(s.v[st][r]),SK{});
     auto kl=p.k.get_slice(_0{}),vl=p.v.get_slice(_0{});
     copy(p.k.with(reinterpret_cast<uint64_t&>(s.full[st])),kl.partition_S(ks),kl.partition_D(kd));
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[st])),vl.partition_S(vs),vl.partition_D(vd));
    }
    auto bs=p.bias.get_tma_tensor(make_shape(256,N/4,nk,nq,4))(_,_,seq,qt,h);
    auto bd=make_tensor(make_smem_ptr(s.bias[st]),SB{});auto bl=p.bias.get_slice(_0{});
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.full[st])),bl.partition_S(bs),bl.partition_D(bd));
   }
  }
  return;
 }
 if constexpr(SOL_REG_CONSUMER>0)cutlass::arch::warpgroup_reg_alloc<SOL_REG_CONSUMER>();
 int wg=tid/128,t=tid%128;
 QK qk;PV pv;LS ls;
 auto tq=qk.get_slice(t);auto tp=pv.get_slice(t);auto tl=ls.get_slice(t);
 auto sq=make_tensor(make_smem_ptr(s.q[wg]),SQ{});auto qa=tq.partition_fragment_A(sq);
 auto ones=make_tensor(make_smem_ptr(s.ones),S1{});auto lb=tl.partition_fragment_B(ones);
 auto score=partition_fragment_C(qk,Shape<_64,Int<N>>{});
 auto next_score=partition_fragment_C(qk,Shape<_64,Int<N>>{});
 auto acc=partition_fragment_C(pv,Shape<_64,_32>{});auto sum=partition_fragment_C(ls,Shape<_64,_8>{});clear(acc);clear(sum);
 auto ar=make_tensor(acc.data(),flash::convert_layout_acc_rowcol(acc.layout()));
 auto lr=make_tensor(sum.data(),flash::convert_layout_acc_rowcol(sum.layout()));
 auto pp=make_tensor_like<Element>(make_tensor(score.data(),flash::convert_layout_acc_Aregs<PV>(score.layout())));
 float mx[2]={-INFINITY,-INFINITY};float c=p.scale*1.4426950408889634f;
 warpgroup_fence_operand(acc);warpgroup_fence_operand(sum);
 s.qr.wait(0);asm volatile("":::"memory");
 auto issue_qk=[&](auto& sc,int seq) __attribute__((always_inline)) {
  int st=seq%Stages;if(seq<nk){s.full[st].wait((seq/Stages)&1);asm volatile("":::"memory");}
  #pragma unroll
  for(int u=0;u<N/8;u++) {
   float4 b=*reinterpret_cast<float4 const*>(s.bias[st]+u*512+t*4);
   sc(4*u)=b.x;sc(4*u+1)=b.y;sc(4*u+2)=b.z;sc(4*u+3)=b.w;
  }
  auto sk=make_tensor(make_smem_ptr(s.k[st][wg]),SK{});auto kb=tq.partition_fragment_B(sk);
  warpgroup_fence_operand(sc);warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<2;kk++)gemm(qk,qa(_,_,kk),kb(_,_,kk),sc);
  warpgroup_commit_batch();
 };
 auto exponentiate=[&](auto& sc) __attribute__((always_inline)) {
  auto sr=make_tensor(sc.data(),flash::convert_layout_acc_rowcol(sc.layout()));
  #pragma unroll
  for(int row=0;row<2;row++) {
   if(Safe || mx[row]==-INFINITY) {
    float m=sr(row,0);
    #pragma unroll
    for(int col=1;col<N/4;col++)m=fmaxf(m,sr(row,col));
    m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));m*=c;
    if constexpr(Safe) {
     float next=fmaxf(mx[row],m);float alpha=mx[row]==-INFINITY?0.f:ex2(mx[row]-next);
     #pragma unroll
     for(int col=0;col<8;col++)ar(row,col)*=alpha;
     lr(row,0)*=alpha;lr(row,1)*=alpha;mx[row]=next;
    } else { if(m!=-INFINITY)mx[row]=m+64.f; }
   }
   #pragma unroll
   for(int col=0;col<N/4;col++) {
    float x=sr(row,col);sr(row,col)=x==-INFINITY?0.f:ex2(fmaf(x,c,-mx[row]));
   }
  }
 };
 auto issue_pv=[&](auto& sc,int seq) __attribute__((always_inline)) {
  auto packed=recast<uint32_t>(pp);
  #pragma unroll
  for(int n=0;n<N/4;n++) { auto x=__floats2bfloat162_rn(sc(2*n),sc(2*n+1));packed(n)=reinterpret_cast<uint32_t const&>(x); }
  auto sv=make_tensor(make_smem_ptr(s.v[seq%Stages][wg]),SV{});auto vb=tp.partition_fragment_B(sv);
  warpgroup_fence_operand(pp);
  if constexpr(Safe){warpgroup_fence_operand(acc);warpgroup_fence_operand(sum);}
  warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<N/16;kk++)gemm(pv,pp(_,_,kk),vb(_,_,kk),acc);
  #pragma unroll
  for(int kk=0;kk<N/16;kk++)gemm(ls,pp(_,_,kk),lb(_,_,kk),sum);
  warpgroup_commit_batch();
 };
 if constexpr(Safe) {
  for(int seq=0;seq<nk;seq++) {
   issue_qk(score,seq);warpgroup_wait<0>();warpgroup_fence_operand(score);
   exponentiate(score);issue_pv(score,seq);
   warpgroup_wait<0>();warpgroup_fence_operand(acc);warpgroup_fence_operand(sum);
   if(t%32==0)s.empty[seq%Stages].arrive();
  }
 } else {
  // Prologue establishes exactly two outstanding groups: QK(1), PV(0).
  // The steady body always issues QK and PV, including an unused in-bounds
  // shared-memory QK past the stream end. A fixed group sequence lets ptxas
  // preserve asynchronous overlap across control-flow joins.
  issue_qk(score,0);warpgroup_wait<0>();warpgroup_fence_operand(score);
  issue_qk(next_score,1);exponentiate(score);issue_pv(score,0);
  auto step=[&](auto parity,int seq) __attribute__((always_inline)) {
   constexpr int cur=decltype(parity)::value;
   auto& sc=cur==0?score:next_score;auto& ns=cur==0?next_score:score;
   warpgroup_wait<1>();warpgroup_fence_operand(sc);
   issue_qk(ns,seq+1);
   exponentiate(sc);
   warpgroup_wait<1>();warpgroup_fence_operand(pp);
   if(t%32==0)s.empty[(seq-1)%Stages].arrive();
   issue_pv(sc,seq);
  };
  #pragma unroll
  for(int seq=1;seq<nk-1;seq+=2){step(_1{},seq);step(_0{},seq+1);}
  step(_1{},nk-1);
  warpgroup_wait<0>();warpgroup_fence_operand(acc);warpgroup_fence_operand(sum);
  if(t%32==0)s.empty[(nk-1)%Stages].arrive();
  bool bad=false;
  #pragma unroll
  for(int row=0;row<2;row++)bad|=!(lr(row,0)>0x1p-84f && lr(row,0)<0x1p-44f);
  if(__any_sync(0xffffffff,bad) && t%32==0 && atomicCAS(&s.bad,0,1)==0){int index=atomicAdd(p.fix,1);p.fix[index+1]=tile;}
 }
 auto id=tp.partition_C(make_identity_tensor(Shape<_64,_32>{}));
 #pragma unroll
 for(int row=0;row<2;row++){
  float inv=1.f/lr(row,0);
  #pragma unroll
  for(int col=0;col<8;col++)ar(row,col)*=inv;
 }
 #pragma unroll
 for(int n=0;n<size(acc);n+=2){
  auto x=__floats2bfloat162_rn(acc(n),acc(n+1));int q=qt*M+get<0>(id(n)),d=get<1>(id(n));
  if(i0+wg<p.L)*reinterpret_cast<uint32_t*>(p.out+((int64_t(i0+wg)*4+h)*p.L+q)*32+d)=reinterpret_cast<uint32_t const&>(x);
 }
}
__global__ void prepare(float const* bias,bool const* mask,int64_t stride,float inv,int L,float* out,int* valid,int* fix){
 int kt=blockIdx.x,qt=blockIdx.y,h=blockIdx.z,nk=L/N,nq=L/M;
 for(int idx=threadIdx.x;idx<M*N;idx+=256){
  int e=idx&3,t=(idx>>2)&127,u=idx>>9;
  int q=qt*M+16*(t>>5)+(t&31)/4+8*(e/2),k=kt*N+8*u+2*(t%4)+e%2;
  out[((int64_t(h)*nq+qt)*nk+kt)*M*N+idx]=(!mask || mask[int64_t(k)*stride])?bias[(int64_t(h)*L+q)*L+k]*inv:-INFINITY;
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
 using namespace SOL_NAMESPACE;int L=q.size(-2);TORCH_CHECK(L==768,"fixed L768 kernel");
 TORCH_CHECK((L==384 || L==768 || L==1024) && q.numel()==int64_t(L)*4*L*32 && q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous(),"C128/H4/D32 contiguous square BF16 operands required");
 for(auto const& x:{k,v})TORCH_CHECK(x.device()==q.device() && x.sizes()==q.sizes() && x.scalar_type()==q.scalar_type() && x.is_contiguous(),"unsupported K/V");
 TORCH_CHECK(bias.device()==q.device() && bias.scalar_type()==at::kFloat && bias.is_contiguous() && bias.numel()==4*L*L,"contiguous float bias required");
 bool const* mp=nullptr;int64_t ms=1;
 if(mask){auto m=*mask;TORCH_CHECK(m.device()==q.device() && m.scalar_type()==at::kBool && m.dim()==5 && m.size(0)==1 && m.size(1)==L && m.size(2)==1 && m.size(3)==1 && m.size(4)==L && m.stride(1)==0,"row-broadcast mask required");mp=m.data_ptr<bool>();ms=m.stride(4);}
 c10::cuda::CUDAGuard guard(q.device());
 auto output=at::empty_like(q),prepared=at::empty_like(bias),fix=at::empty({1+(L/M)*((L+Rows-1)/Rows)*4},q.options().dtype(at::kInt)),valid=at::empty({1},fix.options());
 auto stream=at::cuda::getCurrentCUDAStream();
 prepare<<<dim3(L/N,L/M,4),256,0,stream>>>(bias.data_ptr<float>(),mp,ms,float(1./scale),L,prepared.data_ptr<float>(),valid.data_ptr<int>(),fix.data_ptr<int>());
 auto g=[&](at::Tensor x){return make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),make_shape(L,32,L*4),ST{32,_1{},int64_t(L)*32});};
 auto tq=make_tma_copy(SM90_TMA_LOAD{},g(q),SQ{},Shape<_64,_32>{},_1{});
 auto tk=make_tma_copy(SM90_TMA_LOAD{},g(k),SK{},Shape<Int<N>,_32>{},_1{});
 auto tv=make_tma_copy(SM90_TMA_LOAD{},g(v),SK{},Shape<Int<N>,_32>{},_1{});
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
