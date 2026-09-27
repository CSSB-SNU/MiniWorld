#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>
namespace ta_epi_pipeline {
using namespace cute;
constexpr int M=64;
using Elem=cutlass::bfloat16_t;
using SA=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<Int<M>,_128>{}));
using SW=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<_128,_128>{}));
using SO=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Elem>{},Shape<Int<M>,_32>{}));
using GZ=Shape<int,int,int>;using DZ=Stride<int64_t,_1,int64_t>;
using G2=Shape<int,int>;using D2=Stride<int64_t,_1>;
using G4=Shape<int,int,int,int>;using D4=Stride<int64_t,_1,int64_t,int64_t>;
using TA=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G2{},D2{}),SA{},Shape<Int<M>,_128>{},_1{}));
using TZ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),GZ{},DZ{}),SA{},Shape<Int<M>,_128>{},_1{}));
using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G2{},D2{}),SW{},Shape<_128,_128>{},_1{}));
using TO=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G4{},D4{}),SO{},Shape<Int<M>,_32>{},_1{}));
using TS=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)nullptr),GZ{},DZ{}),SA{},Shape<Int<M>,_128>{},_1{}));
using MMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Elem,Elem,float,Shape<Int<M>,_128,_128>>(),Layout<Shape<_1,_1,_1>>{}));
struct Params{TA gate;TZ z;TW weight;TO o;TS out;uint16_t const* lookup;};
struct Shared {
 alignas(128) Elem a[2][M*128],oz[2][M*128],w[128*128];
 cutlass::arch::ClusterTransactionBarrier inputs[2],residual[2],weight;
 cutlass::arch::ClusterBarrier gated[2],output[2],empty[2];
};
template<int L> __global__ __launch_bounds__(192,2) void epilogue_tma(CUTE_GRID_CONSTANT Params const p){
 extern __shared__ __align__(128) unsigned char storage[];
 auto& s=*reinterpret_cast<Shared*>(storage);
 int tid=threadIdx.x;
 int count=(L*L/M-1-int(blockIdx.x))/int(gridDim.x)+1;
 auto w=make_tensor(make_smem_ptr(s.w),SW{});
 if(tid==0){
  s.weight.init(1);
  for(int st=0;st<2;st++){s.inputs[st].init(1);s.residual[st].init(1);s.gated[st].init(1);s.output[st].init(1);s.empty[st].init(1);}
  cutlass::arch::fence_barrier_init();
 }
 __syncthreads();
 if(tid>=128 && tid<160){
  if(tid==128){
   auto load_inputs=[&](int seq){
    int st=seq&1,tile=blockIdx.x+seq*gridDim.x,i=tile*M/L,j=tile*M%L;
    auto a=make_tensor(make_smem_ptr(s.a[st]),SA{});
    s.inputs[st].arrive_and_expect_tx(2*M*128*sizeof(Elem));
    auto src=local_tile(p.gate.get_tma_tensor(make_shape(Int<L*L>{},_128{})),Shape<Int<M>,_128>{},make_coord(tile,0));
    auto gs=p.gate.get_slice(_0{});
    copy(p.gate.with(reinterpret_cast<uint64_t&>(s.inputs[st])),gs.partition_S(src),gs.partition_D(a));
    #pragma unroll
    for(int h=0;h<4;h++){
     auto src=local_tile(p.o.get_tma_tensor(make_shape(Int<L>{},_32{},_4{},Int<L>{}))(_,_,h,i),Shape<Int<M>,_32>{},make_coord(j/M,0));
     auto dst=make_tensor(make_smem_ptr(s.oz[st]+h*M*32),SO{});auto os=p.o.get_slice(_0{});
     copy(p.o.with(reinterpret_cast<uint64_t&>(s.inputs[st])),os.partition_S(src),os.partition_D(dst));
    }
   };
   s.weight.arrive_and_expect_tx(128*128*sizeof(Elem));
   auto src=p.weight.get_tma_tensor(Shape<_128,_128>{});auto ws=p.weight.get_slice(_0{});
   copy(p.weight.with(reinterpret_cast<uint64_t&>(s.weight)),ws.partition_S(src),ws.partition_D(w));
   load_inputs(0);if(count>1)load_inputs(1);
   for(int seq=0;seq<count;seq++){
    int st=seq&1,tile=blockIdx.x+seq*gridDim.x,i=tile*M/L,j=tile*M%L;
    s.gated[st].wait((seq/2)&1);asm volatile("":::"memory");
    s.residual[st].arrive_and_expect_tx(M*128*sizeof(Elem));
    auto src=local_tile(p.z.get_tma_tensor(make_shape(Int<L>{},_128{},Int<L>{}))(_,_,i),Shape<Int<M>,_128>{},make_coord(j/M,0));
    auto dst=make_tensor(make_smem_ptr(s.oz[st]),SA{});auto zs=p.z.get_slice(_0{});
    copy(p.z.with(reinterpret_cast<uint64_t&>(s.residual[st])),zs.partition_S(src),zs.partition_D(dst));
    if(seq>=1 && seq+1<count){
     int previous=(seq-1)&1;
     s.empty[previous].wait(((seq-1)/2)&1);asm volatile("":::"memory");
     load_inputs(seq+1);
    }
   }
  }
  return;
 }
 if(tid>=160){
  if(tid==160){
   for(int seq=0;seq<count;seq++){
    int st=seq&1,tile=blockIdx.x+seq*gridDim.x,i=tile*M/L,j=tile*M%L;
    s.output[st].wait((seq/2)&1);asm volatile("":::"memory");
    auto src=make_tensor(make_smem_ptr(s.a[st]),SA{});
    auto dst=local_tile(p.out.get_tma_tensor(make_shape(Int<L>{},_128{},Int<L>{}))(_,_,i),Shape<Int<M>,_128>{},make_coord(j/M,0));auto os=p.out.get_slice(_0{});
    copy(p.out,os.partition_S(src),os.partition_D(dst));tma_store_arrive();tma_store_wait<0>();
    s.empty[st].arrive();
   }
  }
  return;
 }
 s.weight.wait(0);asm volatile("":::"memory");
 for(int seq=0;seq<count;seq++){
  int st=seq&1;
  s.inputs[st].wait((seq/2)&1);asm volatile("":::"memory");
  auto a=make_tensor(make_smem_ptr(s.a[st]),SA{}),z=make_tensor(make_smem_ptr(s.oz[st]),SA{});
  #pragma unroll 8
  for(int n=tid;n<M*64;n+=128){
   int row=n/64,c=(n%64)*2;
   auto o=make_tensor(make_smem_ptr(s.oz[st]+(c/32)*M*32),SO{});
   uint32_t gates=*reinterpret_cast<uint32_t*>(&a(row,c));
   uint32_t oval=*reinterpret_cast<uint32_t*>(&o(row,c%32));
   uint32_t sigmoid=uint32_t(__ldg(p.lookup+(gates&65535))) | (uint32_t(__ldg(p.lookup+(gates>>16)))<<16);
   uint32_t gated;asm volatile("mul.rn.bf16x2 %0, %1, %2;":"=r"(gated):"r"(oval),"r"(sigmoid));
   *reinterpret_cast<uint32_t*>(&a(row,c))=gated;
  }
  cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,1);
  if(tid==0)s.gated[st].arrive();
  MMA mma;auto tm=mma.get_slice(tid);auto ma=tm.partition_fragment_A(a);auto mb=tm.partition_fragment_B(w);
  auto acc=partition_fragment_C(mma,Shape<Int<M>,_128>{});clear(acc);
  warpgroup_fence_operand(acc);warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<size<2>(ma);kk++)gemm(mma,ma(_,_,kk),mb(_,_,kk),acc);
  warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(acc);
  s.residual[st].wait((seq/2)&1);asm volatile("":::"memory");
  auto converted=make_tensor<Elem>(shape(acc));
  auto s2r=make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N,Elem>{},mma);auto sr=s2r.get_slice(tid);
  copy(s2r,sr.partition_S(z),sr.retile_D(converted));
  #pragma unroll
  for(int n=0;n<size(acc);n++)converted(n)=Elem(__fadd_rn(float(Elem(acc(n))),float(converted(n))));
  auto r2s=make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N,Elem>{},mma);auto rs=r2s.get_slice(tid);
  copy(r2s,rs.retile_S(converted),rs.partition_D(a));
  cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,1);
  if(tid==0)s.output[st].arrive();
 }
}
}
void epilogue_pipeline_cuda(at::Tensor o,at::Tensor g,at::Tensor w,at::Tensor z,at::Tensor out,at::Tensor lookup,bool ending){
 using namespace ta_epi_pipeline;
 int L=check_pair(z);
 check_bf16(g,z,int64_t(L)*L*128,"gate");check_bf16(out,z,int64_t(L)*L*128,"output");
 check_bf16(w,z,128*128,"output weight");check_bf16(lookup,z,65536,"sigmoid table");
 check_bf16(o,z,int64_t(L)*L*128,"attention output",false);
 TORCH_CHECK(o.dim()>=4 && o.size(-4)==L && o.size(-3)==4 && o.size(-2)==L && o.size(-1)==32 && o.stride(-1)==1,"attention output must have IHJD layout");
 c10::cuda::CUDAGuard guard(z.device());
 auto shape=make_shape(L*L,128);auto stride=D2{128,_1{}};
 auto load=[&](at::Tensor t){return make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)t.data_ptr()),shape,stride),SA{},Shape<Int<M>,_128>{},_1{});};
 auto tw=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)w.data_ptr()),make_shape(128,128),stride),SW{},Shape<_128,_128>{},_1{});
 auto to=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)o.data_ptr()),make_shape(L,32,4,L),D4{o.stride(-2),_1{},o.stride(-3),o.stride(-4)}),SO{},Shape<Int<M>,_32>{},_1{});
 auto zshape=make_shape(L,128,L);auto zstride=DZ{ending?int64_t(L)*128:128,_1{},ending?128:int64_t(L)*128};
 auto tz=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)z.data_ptr()),zshape,zstride),SA{},Shape<Int<M>,_128>{},_1{});
 auto ts=make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)out.data_ptr()),zshape,zstride),SA{},Shape<Int<M>,_128>{},_1{});
 Params p{load(g),tz,tw,to,ts,reinterpret_cast<uint16_t const*>(lookup.data_ptr())};
 #define LAUNCH(LEN) do { \
 C10_CUDA_CHECK(cudaFuncSetAttribute(epilogue_tma<LEN>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared))); \
 epilogue_tma<LEN><<<std::min(L*L/M,2*at::cuda::getCurrentDeviceProperties()->multiProcessorCount),192,sizeof(Shared),at::cuda::getCurrentCUDAStream()>>>(p); \
 } while(0)
 if(L==384){LAUNCH(384);}else if(L==768){LAUNCH(768);}else if(L==1024){LAUNCH(1024);}else{TORCH_CHECK(false,"unsupported L");}
 #undef LAUNCH
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
