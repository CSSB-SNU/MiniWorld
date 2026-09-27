// Native wide projection dgrad: pipelined WGMMA and shared weights across WGs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;
using Width=Int<PAIR_DIM>;
using N=Int<TILE_N>;
using K=Int<TILE_K>;
struct Config {
  using A=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,K>{}));
  using B=decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{},Shape<N,K>{}));
  using MMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,N,K>,GMMA::Major::K,GMMA::Major::MN>()));
  using TA=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),make_shape(int32_t{},Width{}),make_stride(Width{},_1{})),A{},Shape<_64,K>{},_1{}));
  using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),Shape<Width,Width>{},Stride<_1,Width>{}),B{},Shape<N,K>{},_1{}));
  struct Shared {
    array_aligned<Element,cosize_v<A>,1024> a[STAGES][CONSUMERS];
    array_aligned<Element,cosize_v<B>,1024> b[STAGES];
    cutlass::arch::ClusterTransactionBarrier full[STAGES];
    cutlass::arch::ClusterBarrier empty[STAGES];
  };
  struct Params {TA a[4];TB b[4];Element const *db,*wb;Element* dx;int rows;};
};
template<class T,class G,class S>
__device__ __forceinline__ void load(T const& t,G const& g,S const& s,cutlass::arch::ClusterTransactionBarrier& bar){
  auto c=t.get_slice(_0{});copy(t.with(reinterpret_cast<uint64_t&>(bar)),c.partition_S(g),c.partition_D(s));
}
__global__ __launch_bounds__(128*(CONSUMERS+1),MIN_BLOCKS)
void projection_pipeline(CUTE_GRID_CONSTANT Config::Params const p){
  extern __shared__ char storage[];auto& s=*reinterpret_cast<Config::Shared*>(storage);
  int tid=threadIdx.x,wg=tid/128-1,lane=tid%128;
  constexpr int ITER=4*(PAIR_DIM/TILE_K);
  if(tid==0){
    for(int j=0;j<STAGES;++j){s.full[j].init(1);s.empty[j].init(CONSUMERS);}
    for(int j=0;j<4;++j){prefetch_tma_descriptor(p.a[j].get_tma_descriptor());prefetch_tma_descriptor(p.b[j].get_tma_descriptor());}
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if(tid<128){
#if DYNAMIC_REG
    cutlass::arch::warpgroup_reg_dealloc<32>();
#endif
    if(tid==0){
      for(int it=0;it<ITER;++it){
        int st=it%STAGES,phase=(it/STAGES)%2,proj=it/(PAIR_DIM/TILE_K),kt=it%(PAIR_DIM/TILE_K);
        s.empty[st].wait(phase^1);
        s.full[st].arrive_and_expect_tx((CONSUMERS*64+TILE_N)*TILE_K*sizeof(Element));
        for(int w=0;w<CONSUMERS;++w){
          auto a=local_tile(p.a[proj].get_tma_tensor(make_shape(p.rows,Width{})),Shape<_64,K>{},make_coord(blockIdx.x*CONSUMERS+w,kt));
          load(p.a[proj],a,make_tensor(make_smem_ptr(s.a[st][w].data()),Config::A{}),s.full[st]);
        }
        auto b=local_tile(p.b[proj].get_tma_tensor(Shape<Width,Width>{}),Shape<N,K>{},make_coord(blockIdx.y,kt));
        load(p.b[proj],b,make_tensor(make_smem_ptr(s.b[st].data()),Config::B{}),s.full[st]);
      }
    }
    return;
  }
#if DYNAMIC_REG
  cutlass::arch::warpgroup_reg_alloc<CONSUMER_REGS>();
#endif
  Config::MMA mma;auto thr=mma.get_slice(lane);
  auto acc=partition_fragment_C(mma,Shape<_64,N>{});clear(acc);
  for(int it=0;it<ITER;++it){
    int st=it%STAGES,phase=(it/STAGES)%2;s.full[st].wait(phase);
    auto a=make_tensor(make_smem_ptr(s.a[st][wg].data()),Config::A{});
    auto b=make_tensor(make_smem_ptr(s.b[st].data()),Config::B{});
    auto ra=thr.partition_fragment_A(a);auto rb=thr.partition_fragment_B(b);
    flash::gemm<false,-1>(mma,ra,rb,acc);
    if(it>0){
      warpgroup_wait<1>();
      // The previous MMA group is complete in every warp before TMA reuses it.
      cutlass::arch::NamedBarrier::sync(128,wg+1);
      if(lane==0)s.empty[(it-1)%STAGES].arrive();
    }
  }
  warpgroup_wait<0>();warpgroup_fence_operand(acc);
  auto coord=thr.partition_C(make_identity_tensor(Shape<_64,N>{}));
  #pragma unroll
  for(int i=0;i<size(acc);++i){
    int row=(blockIdx.x*CONSUMERS+wg)*64+get<0>(coord(i)),col=blockIdx.y*TILE_N+get<1>(coord(i));
    if(row<p.rows){
      float value=acc(i);
      #pragma unroll
      for(int h=0;h<4;++h)value=fmaf(float(p.db[row*4+h]),float(p.wb[h*PAIR_DIM+col]),value);
      p.dx[int64_t(row)*PAIR_DIM+col]=Element(value);
    }
  }
}
torch::Tensor dgrad(std::vector<torch::Tensor> dy,std::vector<torch::Tensor> weights,int tile){
  TORCH_CHECK(dy.size()==5 && weights.size()==5 && tile==64,"five gradients, five weights, tile64 required");
  c10::cuda::CUDAGuard guard(dy[0].device());int rows=dy[0].numel()/PAIR_DIM;
  for(int j=0;j<5;++j){
    auto w=weights[j];auto g=dy[j];
    TORCH_CHECK(g.is_cuda() && g.device()==dy[0].device() && w.device()==g.device(),"device mismatch");
    TORCH_CHECK(g.scalar_type()==torch::kBFloat16 && w.scalar_type()==g.scalar_type() && g.is_contiguous() && w.is_contiguous(),"contiguous BF16 inputs required");
    TORCH_CHECK(w.sizes()==torch::IntArrayRef({j<4?PAIR_DIM:4,PAIR_DIM}) && g.numel()==int64_t(rows)*(j<4?PAIR_DIM:4),"shape mismatch");
  }
  auto dx=torch::empty({rows,PAIR_DIM},dy[0].options());Config::Params p;
  for(int j=0;j<4;++j){
    auto a=make_tensor(make_gmem_ptr((Element const*)dy[j].data_ptr()),make_shape(rows,Width{}),make_stride(Width{},_1{}));
    auto b=make_tensor(make_gmem_ptr((Element const*)weights[j].data_ptr()),Shape<Width,Width>{},Stride<_1,Width>{});
    p.a[j]=make_tma_copy(SM90_TMA_LOAD{},a,Config::A{},Shape<_64,K>{},_1{});
    p.b[j]=make_tma_copy(SM90_TMA_LOAD{},b,Config::B{},Shape<N,K>{},_1{});
  }
  p.db=(Element const*)dy[4].data_ptr();p.wb=(Element const*)weights[4].data_ptr();p.dx=(Element*)dx.data_ptr();p.rows=rows;
  C10_CUDA_CHECK(cudaFuncSetAttribute(projection_pipeline,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Config::Shared)));
  projection_pipeline<<<dim3((rows+64*CONSUMERS-1)/(64*CONSUMERS),PAIR_DIM/TILE_N),128*(1+CONSUMERS),sizeof(Config::Shared),at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return dx;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("dgrad",&dgrad);m.def("smem",[](){return sizeof(Config::Shared);});}
