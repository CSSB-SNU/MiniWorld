#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>
namespace ta_epi_packed128 {
using namespace cute;
constexpr int M=128;
using Elem=cutlass::bfloat16_t;
using SA=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<Int<M>,_128>{}));
using SW=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<_128,_128>{}));
using SO=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Elem>{},Shape<Int<M>,_32>{}));
using G2=Shape<int,int>;using D2=Stride<int64_t,_1>;
using G4=Shape<int,int,int,int>;using D4=Stride<int64_t,_1,int64_t,int64_t>;
using TA=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G2{},D2{}),SA{},Shape<Int<M>,_128>{},_1{}));
using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G2{},D2{}),SW{},Shape<_128,_128>{},_1{}));
using TO=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G4{},D4{}),SO{},Shape<Int<M>,_32>{},_1{}));
using TS=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)nullptr),G2{},D2{}),SA{},Shape<Int<M>,_128>{},_1{}));
using MMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Elem,Elem,float,Shape<Int<M>,_128,_128>>(),Layout<Shape<_2,_1,_1>>{}));
struct Params{TA gate,z;TW weight;TO o;TS out;uint16_t const* lookup;};
struct Shared{
 alignas(128) Elem a[M*128],oz[M*128],w[128*128];
 cutlass::arch::ClusterTransactionBarrier inputs,residual;
};
template<int L> __global__ __launch_bounds__(256,2) void epilogue_tma(CUTE_GRID_CONSTANT Params const p){
 extern __shared__ __align__(128) unsigned char storage[];
 auto& s=*reinterpret_cast<Shared*>(storage);
 int tid=threadIdx.x,tile=blockIdx.x,i=tile*M/L,j=tile*M%L;
 auto a=make_tensor(make_smem_ptr(s.a),SA{}),z=make_tensor(make_smem_ptr(s.oz),SA{});
 auto w=make_tensor(make_smem_ptr(s.w),SW{});
 if(tid==0){s.inputs.init(1);s.residual.init(1);cutlass::arch::fence_barrier_init();}
 __syncthreads();
 if(tid==0){
  s.inputs.arrive_and_expect_tx((M*128*2+128*128)*sizeof(Elem));
  auto gsrc=local_tile(p.gate.get_tma_tensor(make_shape(Int<L*L>{},_128{})),Shape<Int<M>,_128>{},make_coord(tile,0));
  auto gs=p.gate.get_slice(_0{});
  copy(p.gate.with(reinterpret_cast<uint64_t&>(s.inputs)),gs.partition_S(gsrc),gs.partition_D(a));
  auto wsrc=p.weight.get_tma_tensor(Shape<_128,_128>{});auto ws=p.weight.get_slice(_0{});
  copy(p.weight.with(reinterpret_cast<uint64_t&>(s.inputs)),ws.partition_S(wsrc),ws.partition_D(w));
  #pragma unroll
  for(int h=0;h<4;h++){
   auto osrc=local_tile(p.o.get_tma_tensor(make_shape(Int<L>{},_32{},_4{},Int<L>{}))(_,_,h,i),Shape<Int<M>,_32>{},make_coord(j/M,0));
   auto dst=make_tensor(make_smem_ptr(s.oz+h*M*32),SO{});auto os=p.o.get_slice(_0{});
   copy(p.o.with(reinterpret_cast<uint64_t&>(s.inputs)),os.partition_S(osrc),os.partition_D(dst));
  }
 }
 s.inputs.wait(0);asm volatile("":::"memory");
 #pragma unroll 8
 for(int n=tid;n<M*64;n+=M*2){
  int row=n/64,c=(n%64)*2;
  auto o=make_tensor(make_smem_ptr(s.oz+(c/32)*M*32),SO{});
  uint32_t gates=*reinterpret_cast<uint32_t*>(&a(row,c));
  uint32_t oval=*reinterpret_cast<uint32_t*>(&o(row,c%32));
  uint32_t sigmoid=uint32_t(__ldg(p.lookup+(gates&65535))) | (uint32_t(__ldg(p.lookup+(gates>>16)))<<16);
  uint32_t gated;
  asm volatile("mul.rn.bf16x2 %0, %1, %2;":"=r"(gated):"r"(oval),"r"(sigmoid));
  *reinterpret_cast<uint32_t*>(&a(row,c))=gated;
 }
 cutlass::arch::fence_view_async_shared();__syncthreads();
 if(tid==0){
  s.residual.arrive_and_expect_tx(M*128*sizeof(Elem));
  auto src=local_tile(p.z.get_tma_tensor(make_shape(Int<L*L>{},_128{})),Shape<Int<M>,_128>{},make_coord(tile,0));
  auto zs=p.z.get_slice(_0{});
  copy(p.z.with(reinterpret_cast<uint64_t&>(s.residual)),zs.partition_S(src),zs.partition_D(z));
 }
 MMA mma;auto tm=mma.get_slice(tid);auto ma=tm.partition_fragment_A(a);auto mb=tm.partition_fragment_B(w);
 auto acc=partition_fragment_C(mma,Shape<Int<M>,_128>{});clear(acc);
 warpgroup_fence_operand(acc);warpgroup_arrive();
 #pragma unroll
 for(int kk=0;kk<size<2>(ma);kk++)gemm(mma,ma(_,_,kk),mb(_,_,kk),acc);
 warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(acc);
 s.residual.wait(0);asm volatile("":::"memory");
 auto converted=make_tensor<Elem>(shape(acc));
 auto s2r=make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N,Elem>{},mma);
 auto st=s2r.get_slice(tid);
 copy(s2r,st.partition_S(z),st.retile_D(converted));
 #pragma unroll
 for(int n=0;n<size(acc);n++)converted(n)=Elem(__fadd_rn(float(Elem(acc(n))),float(converted(n))));
 auto r2s=make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N,Elem>{},mma);
 auto rt=r2s.get_slice(tid);
 copy(r2s,rt.retile_S(converted),rt.partition_D(a));
 cutlass::arch::fence_view_async_shared();__syncthreads();
 if(tid==0){
  auto dst=local_tile(p.out.get_tma_tensor(make_shape(Int<L*L>{},_128{})),Shape<Int<M>,_128>{},make_coord(tile,0));auto os=p.out.get_slice(_0{});
  copy(p.out,os.partition_S(a),os.partition_D(dst));tma_store_arrive();tma_store_wait<0>();
 }
}
}
void epilogue_packed128_cuda(at::Tensor o,at::Tensor g,at::Tensor w,at::Tensor z,at::Tensor out,at::Tensor lookup){
 using namespace ta_epi_packed128;
 c10::cuda::CUDAGuard guard(z.device());int L=z.size(-2);
 TORCH_CHECK(z.is_contiguous() && g.is_contiguous() && w.is_contiguous() && out.is_contiguous(),"contiguous tensors required");
 auto shape=make_shape(L*L,128);auto stride=D2{128,_1{}};
 auto load=[&](at::Tensor t){return make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)t.data_ptr()),shape,stride),SA{},Shape<Int<M>,_128>{},_1{});};
 auto tw=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)w.data_ptr()),make_shape(128,128),stride),SW{},Shape<_128,_128>{},_1{});
 auto to=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)o.data_ptr()),make_shape(L,32,4,L),D4{o.stride(-2),_1{},o.stride(-3),o.stride(-4)}),SO{},Shape<Int<M>,_32>{},_1{});
 auto ts=make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)out.data_ptr()),shape,stride),SA{},Shape<Int<M>,_128>{},_1{});
 Params p{load(g),load(z),tw,to,ts,reinterpret_cast<uint16_t const*>(lookup.data_ptr())};
 #define LAUNCH(LEN) do { \
 C10_CUDA_CHECK(cudaFuncSetAttribute(epilogue_tma<LEN>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared))); \
 epilogue_tma<LEN><<<L*L/M,M*2,sizeof(Shared),at::cuda::getCurrentCUDAStream()>>>(p); \
 } while(0)
 if(L==384){LAUNCH(384);}else if(L==768){LAUNCH(768);}else if(L==1024){LAUNCH(1024);}else{TORCH_CHECK(false,"unsupported L");}
 #undef LAUNCH
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
