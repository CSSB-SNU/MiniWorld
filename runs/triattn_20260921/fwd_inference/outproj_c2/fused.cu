#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;
constexpr int ConsumerGroups=2;
using OutputLayout=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));
using WeightLayout=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_128,_128>{}));
using GS=Shape<int32_t,_128,int32_t>;using Str=Stride<int64_t,_1,int64_t>;
using WS=Shape<_128,_128>;using WStr=Stride<_128,_1>;
using Load=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GS{},Str{}),OutputLayout{},Shape<_64,_128>{},_1{}));
using WL=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),WS{},WStr{}),WeightLayout{},Shape<_128,_128>{},_1{}));
using Save=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),GS{},Str{}),OutputLayout{},Shape<_64,_128>{},_1{}));
using OutputMMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_128,_128>>()));
struct Shared {
  array_aligned<Element,8192,1024> input[ConsumerGroups],residual[ConsumerGroups];
  array_aligned<Element,16384,1024> weight;
  cutlass::arch::ClusterTransactionBarrier full,xfull;
};
struct Params {Load input,residual;WL weight;Save output;int L;};
template<class T,class G,class S>
__device__ __forceinline__ void fetch(T const& t,G const& g,S const& s,cutlass::arch::ClusterTransactionBarrier& bar){
  auto slice=t.get_slice(_0{});
  copy(t.with(reinterpret_cast<uint64_t&>(bar)),slice.partition_S(g),slice.partition_D(s));
}
__global__ __launch_bounds__(ConsumerGroups*128,1)
void inference_output_residual(CUTE_GRID_CONSTANT Params const p){
  extern __shared__ char storage[];auto& s=*reinterpret_cast<Shared*>(storage);
  int tid=threadIdx.x,c=tid/128,lane=tid%128,qt=blockIdx.x*ConsumerGroups+c,row=blockIdx.y;
  int L=p.L,nt=L/64;
  if(tid==0){s.full.init(1);s.xfull.init(1);
    prefetch_tma_descriptor(p.input.get_tma_descriptor());prefetch_tma_descriptor(p.residual.get_tma_descriptor());
    prefetch_tma_descriptor(p.weight.get_tma_descriptor());prefetch_tma_descriptor(p.output.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();}
  __syncthreads();
  if(tid==0){
    int count=min(ConsumerGroups,nt-int(blockIdx.x)*ConsumerGroups);
    s.full.arrive_and_expect_tx(32768+count*16384);s.xfull.arrive_and_expect_tx(count*16384);
    auto wg=p.weight.get_tma_tensor(make_shape(_128{},_128{}));
    fetch(p.weight,wg,make_tensor(make_smem_ptr(s.weight.data()),WeightLayout{}),s.full);
    auto ag=p.input.get_tma_tensor(make_shape(L,_128{},L));
    auto xg=p.residual.get_tma_tensor(make_shape(L,_128{},L));
    for(int cc=0;cc<count;++cc){
      auto coord=make_coord(int(blockIdx.x)*ConsumerGroups+cc,0);
      auto aa=local_tile(ag(_,_,row),Shape<_64,_128>{},coord);
      auto xx=local_tile(xg(_,_,row),Shape<_64,_128>{},coord);
      fetch(p.input,aa,make_tensor(make_smem_ptr(s.input[cc].data()),OutputLayout{}),s.full);
      fetch(p.residual,xx,make_tensor(make_smem_ptr(s.residual[cc].data()),OutputLayout{}),s.xfull);
    }
  }
  if(qt<nt){
    s.full.wait(0);
    OutputMMA mma;auto mt=mma.get_slice(lane);
    auto aa=mt.partition_fragment_A(make_tensor(make_smem_ptr(s.input[c].data()),OutputLayout{}));
    auto bb=mt.partition_fragment_B(make_tensor(make_smem_ptr(s.weight.data()),WeightLayout{}));
    auto acc=partition_fragment_C(mma,Shape<_64,_128>{});
    flash::gemm<true,0>(mma,aa,bb,acc);
    s.xfull.wait(0);
    // Complete every GMMA source read before input storage becomes the output tile.
    cutlass::arch::NamedBarrier::sync(128,c+1);
    auto coords=mt.partition_C(make_identity_tensor(Shape<_64,_128>{}));
    #pragma unroll
    for(int n=0;n<size(acc);n+=2){
      int qr=get<0>(coords(n)),ch=get<1>(coords(n));
      int off=as_position_independent_swizzle_layout(OutputLayout{})(make_coord(qr,ch));
      uint32_t xv=*reinterpret_cast<uint32_t const*>(s.residual[c].data()+off);
      // Match BF16(X + BF16(A @ W^T)), including the inner projection rounding.
      float u=float(Element(acc(n)))+float(Element::bitcast(uint16_t(xv)));
      float v=float(Element(acc(n+1)))+float(Element::bitcast(uint16_t(xv>>16)));
      uint32_t out;asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(out):"f"(v),"f"(u));
      *reinterpret_cast<uint32_t*>(s.input[c].data()+off)=out;
    }
    cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,c+1);
    if(lane==0){
      auto og=p.output.get_tma_tensor(make_shape(L,_128{},L));
      auto tile=local_tile(og(_,_,row),Shape<_64,_128>{},make_coord(qt,0));
      auto src=make_tensor(make_smem_ptr(s.input[c].data()),OutputLayout{});auto slice=p.output.get_slice(_0{});
      copy(p.output,slice.partition_S(src),slice.partition_D(tile));tma_store_arrive();tma_store_wait<0>();
    }
  }
}
torch::Tensor launch_output(torch::Tensor input,torch::Tensor weight,torch::Tensor x,bool ending){
  TORCH_CHECK(input.is_cuda() && input.scalar_type()==torch::kBFloat16 && input.is_contiguous() && input.dim()==4 && input.size(0)==1 && input.size(3)==128,"B1 L L C128 BF16 input required");
  c10::cuda::CUDAGuard guard(input.device());int L=input.size(1);
  TORCH_CHECK(input.size(2)==L && L>=64 && L%64==0,"square L divisible by64 required");
  TORCH_CHECK(weight.device()==input.device() && weight.scalar_type()==input.scalar_type() && weight.is_contiguous() && weight.sizes()==torch::IntArrayRef({128,128}),"invalid weight");
  TORCH_CHECK(x.device()==input.device() && x.scalar_type()==input.scalar_type() && x.is_contiguous() && x.sizes()==input.sizes(),"invalid residual");
  auto out=torch::empty_like(x);auto shape=make_shape(L,_128{},L);
  Str astr{128,_1{},int64_t(L)*128};Str xstr{ending?int64_t(L)*128:128,_1{},ending?128:int64_t(L)*128};
  auto ai=make_tensor(make_gmem_ptr((Element const*)input.data_ptr()),shape,astr);
  auto xi=make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),shape,xstr);
  auto wo=make_tensor(make_gmem_ptr((Element const*)weight.data_ptr()),WS{},WStr{});
  auto yo=make_tensor(make_gmem_ptr((Element*)out.data_ptr()),shape,xstr);
  Params p{make_tma_copy(SM90_TMA_LOAD{},ai,OutputLayout{},Shape<_64,_128>{},_1{}),
           make_tma_copy(SM90_TMA_LOAD{},xi,OutputLayout{},Shape<_64,_128>{},_1{}),
           make_tma_copy(SM90_TMA_LOAD{},wo,WeightLayout{},Shape<_128,_128>{},_1{}),
           make_tma_copy(SM90_TMA_STORE{},yo,OutputLayout{},Shape<_64,_128>{},_1{}),L};
  C10_CUDA_CHECK(cudaFuncSetAttribute(inference_output_residual,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
  inference_output_residual<<<dim3((L/64+ConsumerGroups-1)/ConsumerGroups,L),ConsumerGroups*128,sizeof(Shared),at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&launch_output);m.def("smem",[](){return sizeof(Shared);});}
