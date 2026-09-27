#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>

namespace triattn_projected_proof {
using namespace cute;
using Element=cutlass::bfloat16_t;
constexpr int M=128,C=128,D=32,R=2,Stages=2;
using SX=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_128,_128>{}));
using SWQ=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_32,_128>{}));
using SWKV=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));
using SO=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_128,_32>{}));
using GX=Shape<int,int,int>;using DX=Stride<int64_t,_1,int64_t>;
using GW=Shape<int,int>;using DW=Stride<int64_t,_1>;
using GO=Shape<int,int,int,int>;using DO=Stride<int64_t,_1,int64_t,int64_t>;
using TX=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GX{},DX{}),SX{},Shape<_128,_128>{},_1{}));
using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GW{},DW{}),SWQ{},Shape<_32,_128>{},_1{}));
using TWKV=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GW{},DW{}),SWKV{},Shape<_64,_128>{},_1{}));
using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),GO{},DO{}),SO{},Shape<_128,_32>{},_1{}));
using MQ=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_32,_128>>()));
using MKV=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_64,_128>>()));
struct Params {TX x;TW w;TWKV wkv;TO q,k,v;GX xs;GW ws,wkvs;GO os;int L;};
// Use precisely the future R2/two-stage attention Q/K/V layouts. The proof
// uses stage0; stage1 remains reserved to validate the intended address gaps.
struct Shared {
 alignas(128) Element q[R][M*D];
 alignas(128) Element k[Stages][R][M*D];
 alignas(128) Element v[Stages][R][M*D];
 alignas(128) Element x[M*C];
 alignas(128) Element w[2*D*C];
 cutlass::arch::ClusterTransactionBarrier xr,wr;
};
static_assert(sizeof(Shared)<=232448);

template<class MMA,class WTensor>
__device__ __forceinline__ void project(Shared& s,WTensor const& weight,int r,int hh,int tid) {
 MMA mma;auto mt=mma.get_slice(tid);
 auto x=local_tile(make_tensor(make_smem_ptr(s.x),SX{}),Shape<_64,_128>{},make_coord(hh,0));
 auto a=mt.partition_fragment_A(x);auto b=mt.partition_fragment_B(weight);
 constexpr bool Q=std::is_same<MMA,MQ>::value;
 using NN=Int<Q?32:64>;
 auto acc=partition_fragment_C(mma,Shape<_64,NN>{});clear(acc);
 warpgroup_fence_operand(acc);warpgroup_arrive();
 #pragma unroll
 for(int kk=0;kk<size<2>(a);kk++)gemm(mma,a(_,_,kk),b(_,_,kk),acc);
 warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(acc);
 auto id=mt.partition_C(make_identity_tensor(Shape<_64,NN>{}));
 auto oq=make_tensor(make_smem_ptr(s.q[r]),SO{});
 auto ok=make_tensor(make_smem_ptr(s.k[0][r]),SO{});
 auto ov=make_tensor(make_smem_ptr(s.v[0][r]),SO{});
 // Initial correctness proof uses ordinary stores through exact swizzled
 // CuTe layouts. STSM is a separately verified optimization, not assumed here.
 #pragma unroll
 for(int n=0;n<size(acc);n++){
  int m=get<0>(id(n))+hh*64,d=get<1>(id(n));
  Element value=Element(acc(n));
  if constexpr(Q)oq(m,d)=value;
  else {if(d<32)ok(m,d)=value;else ov(m,d-32)=value;}
 }
}

__global__ __launch_bounds__(128,8) void producer_projection(CUTE_GRID_CONSTANT Params const p) {
 extern __shared__ __align__(128) unsigned char storage[];
 Shared& s=*reinterpret_cast<Shared*>(storage);
 int tid=threadIdx.x,h=blockIdx.y,i0=blockIdx.z*R,j=blockIdx.x*M;
 if(tid==0){s.xr.init(1);s.wr.init(1);cutlass::arch::fence_barrier_init();}
 __syncthreads();
 auto x=make_tensor(make_smem_ptr(s.x),SX{});
 auto load_w=[&](int tile,Element* ptr){
  auto dst=make_tensor(make_smem_ptr(ptr),SWQ{});
  auto src=local_tile(p.w.get_tma_tensor(p.ws),Shape<_32,_128>{},make_coord(tile,0));
  auto slice=p.w.get_slice(_0{});
  copy(p.w.with(reinterpret_cast<uint64_t&>(s.wr)),slice.partition_S(src),slice.partition_D(dst));
 };
 #pragma unroll 1
 for(int r=0;r<R;r++){
  int i=i0+r;
  if(tid==0){
   auto src=local_tile(p.x.get_tma_tensor(p.xs)(_,_,i),Shape<_128,_128>{},make_coord(j/M,0));
   auto slice=p.x.get_slice(_0{});
   s.xr.arrive_and_expect_tx(M*C*sizeof(Element));
   copy(p.x.with(reinterpret_cast<uint64_t&>(s.xr)),slice.partition_S(src),slice.partition_D(x));
   s.wr.arrive_and_expect_tx(D*C*sizeof(Element));load_w(h,s.w);
  }
  s.xr.wait(r&1);s.wr.wait(0);asm volatile("":::"memory");
  auto wq=make_tensor(make_smem_ptr(s.w),SWQ{});
  project<MQ>(s,wq,r,0,tid);project<MQ>(s,wq,r,1,tid);
  __syncthreads();
  if(tid==0){
   s.wr.arrive_and_expect_tx(2*D*C*sizeof(Element));
   auto dst=make_tensor(make_smem_ptr(s.w),SWKV{});
   auto src=local_tile(p.wkv.get_tma_tensor(p.wkvs),Shape<_64,_128>{},make_coord(h,0));
   auto slice=p.wkv.get_slice(_0{});
   copy(p.wkv.with(reinterpret_cast<uint64_t&>(s.wr)),slice.partition_S(src),slice.partition_D(dst));
  }
  s.wr.wait(1);asm volatile("":::"memory");
  auto wkv=make_tensor(make_smem_ptr(s.w),SWKV{});
  project<MKV>(s,wkv,r,0,tid);project<MKV>(s,wkv,r,1,tid);
  // All producer lanes finish shared stores before async-proxy readers.
  cutlass::arch::fence_view_async_shared();__syncthreads();
  if(tid==0){
   auto store=[&](TO const& t,Element* ptr){
    auto src=make_tensor(make_smem_ptr(ptr),SO{});
    auto dst=local_tile(t.get_tma_tensor(p.os)(_,_,h,i),Shape<_128,_32>{},make_coord(j/M,0));
    auto slice=t.get_slice(_0{});copy(t,slice.partition_S(src),slice.partition_D(dst));
   };
   store(p.q,s.q[r]);store(p.k,s.k[0][r]);store(p.v,s.v[0][r]);tma_store_arrive();
  }
  __syncthreads(); // Every lane drained its X/weight MMA before their next fill.
 }
 if(tid==0)tma_store_wait<0>();
}
}

void projected_qkv_cuda(at::Tensor x,at::Tensor w,at::Tensor wkv,at::Tensor q,at::Tensor k,at::Tensor v){
 using namespace triattn_projected_proof;
 int L=check_pair(x);TORCH_CHECK(L%128==0 && L%2==0,"invalid projected shape");
 check_bf16(w,x,512*128,"projection weight");
 check_bf16(wkv,x,256*128,"per-head packed K/V weight");
 for(auto const& t:{q,k,v})check_bf16(t,x,int64_t(L)*L*128,"projection output");
 c10::cuda::CUDAGuard guard(x.device());
 GX xs=make_shape(L,128,L);GW ws=make_shape(512,128),wkvs=make_shape(256,128);GO os=make_shape(L,32,4,L);
 auto tx=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),xs,DX{128,_1{},int64_t(L)*128}),SX{},Shape<_128,_128>{},_1{});
 auto tw=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)w.data_ptr()),ws,DW{128,_1{}}),SWQ{},Shape<_32,_128>{},_1{});
 auto twkv=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)wkv.data_ptr()),wkvs,DW{128,_1{}}),SWKV{},Shape<_64,_128>{},_1{});
 auto to=[&](at::Tensor t){return make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)t.data_ptr()),os,DO{32,_1{},int64_t(L)*32,int64_t(4)*L*32}),SO{},Shape<_128,_32>{},_1{});};
 Params p{tx,tw,twkv,to(q),to(k),to(v),xs,ws,wkvs,os,L};
 C10_CUDA_CHECK(cudaFuncSetAttribute(producer_projection,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 producer_projection<<<dim3(L/128,4,L/2),128,sizeof(Shared),at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
