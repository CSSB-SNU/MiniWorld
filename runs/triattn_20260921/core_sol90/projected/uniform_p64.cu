#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>

namespace triattn_projected_uniform_p64 {
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
struct Params {TX x;TWKV wkv;TO out;GX xs;GW wkvs;GO os;int L;int const* valid;};
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
 auto converted=make_tensor<Element>(shape(acc));
 #pragma unroll
 for(int n=0;n<size(acc);n++)converted(n)=Element(acc(n));
 auto copyop=make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N,Element>{},mma);
 auto thread_copy=copyop.get_slice(tid);
 auto src=thread_copy.retile_S(converted);
 if constexpr(Q){
  auto out=local_tile(make_tensor(make_smem_ptr(s.q[r]),SO{}),Shape<_64,_32>{},make_coord(hh,0));
  copy(copyop,src,thread_copy.partition_D(out));
 }else{
  using Full=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_128,_32,Int<Stages*R*2>>{}));
  // Logical N[0:32] maps into K, N[32:64] into V at the whole K-array gap.
  auto mapping=make_layout(Shape<_128,Shape<_32,_2>>{},Stride<_1,Stride<_128,Int<Stages*R*M*D>>>{});
  auto both=make_tensor(make_smem_ptr(s.k[0][r]),composition(Full{},mapping));
  auto out=local_tile(both,Shape<_64,_64>{},make_coord(hh,0));
  copy(copyop,src,thread_copy.partition_D(out));
 }

}

// Rare all-masked fallback: project rounded V into output, then average it.
// All CTAs return immediately for any-valid-key inputs.
__global__ __launch_bounds__(128,4) void project_uniform_v(CUTE_GRID_CONSTANT Params const p){
 if(*p.valid)return;
 extern __shared__ __align__(128) unsigned char storage[];
 auto& s=*reinterpret_cast<Shared*>(storage);
 int tid=threadIdx.x,h=blockIdx.y,i0=blockIdx.z*R,j=blockIdx.x*M;
 if(tid==0){
  s.xr.init(1);s.wr.init(1);cutlass::arch::fence_barrier_init();
 }
 __syncthreads();
 if(tid==0){
  auto dst=make_tensor(make_smem_ptr(s.w),SWKV{});
  auto src=local_tile(p.wkv.get_tma_tensor(p.wkvs),Shape<_64,_128>{},make_coord(h,0));
  auto slice=p.wkv.get_slice(_0{});
  s.wr.arrive_and_expect_tx(64*128*sizeof(Element));
  copy(p.wkv.with(reinterpret_cast<uint64_t&>(s.wr)),slice.partition_S(src),slice.partition_D(dst));
 }
 s.wr.wait(0);asm volatile("":::"memory");
 auto wv=local_tile(make_tensor(make_smem_ptr(s.w),SWKV{}),Shape<_32,_128>{},make_coord(1,0));
 #pragma unroll 1
 for(int r=0;r<R;++r){
  if(tid==0){
   auto dst=make_tensor(make_smem_ptr(s.x),SX{});
   auto src=local_tile(p.x.get_tma_tensor(p.xs)(_,_,i0+r),Shape<_128,_128>{},make_coord(j/M,0));
   auto slice=p.x.get_slice(_0{});
   s.xr.arrive_and_expect_tx(M*C*sizeof(Element));
   copy(p.x.with(reinterpret_cast<uint64_t&>(s.xr)),slice.partition_S(src),slice.partition_D(dst));
  }
  s.xr.wait(r&1);asm volatile("":::"memory");
  // MQ selects M64/N32 and writes the q shared tile. The operand here is W_V.
  project<MQ>(s,wv,r,0,tid);project<MQ>(s,wv,r,1,tid);
  cutlass::arch::fence_view_async_shared();__syncthreads();
  if(tid==0){
   auto src=make_tensor(make_smem_ptr(s.q[r]),SO{});
   auto dst=local_tile(p.out.get_tma_tensor(p.os)(_,_,h,i0+r),Shape<_128,_32>{},make_coord(j/M,0));
   auto slice=p.out.get_slice(_0{});
   copy(p.out,slice.partition_S(src),slice.partition_D(dst));tma_store_arrive();
  }
  __syncthreads();
 }
 if(tid==0)tma_store_wait<0>();
}
}
void projected_p64_uniform_cuda(at::Tensor x,at::Tensor wkv,at::Tensor out,at::Tensor valid){
 using namespace triattn_projected_uniform_p64;
 int L=check_pair(x);
 check_bf16(wkv,x,256*128,"packed KV weight");check_bf16(out,x,int64_t(L)*L*128,"output");
 TORCH_CHECK(valid.device()==x.device() && valid.scalar_type()==at::kInt && valid.numel()==1,"invalid mask flag");
 c10::cuda::CUDAGuard guard(x.device());
 GX xs=make_shape(L,128,L);GW wkvs=make_shape(256,128);GO os=make_shape(L,32,4,L);
 auto tx=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),xs,DX{128,_1{},int64_t(L)*128}),SX{},Shape<_128,_128>{},_1{});
 auto twkv=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)wkv.data_ptr()),wkvs,DW{128,_1{}}),SWKV{},Shape<_64,_128>{},_1{});
 auto to=make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)out.data_ptr()),os,DO{32,_1{},int64_t(L)*32,int64_t(4)*L*32}),SO{},Shape<_128,_32>{},_1{});
 Params p{tx,twkv,to,xs,wkvs,os,L,valid.data_ptr<int>()};
 C10_CUDA_CHECK(cudaFuncSetAttribute(project_uniform_v,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 project_uniform_v<<<dim3(L/128,4,L/2),128,sizeof(Shared),at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
