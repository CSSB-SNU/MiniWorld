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

namespace ta_tma {
using namespace cute;
using Elem=cutlass::bfloat16_t;
constexpr int M=64,K=128,N=64;
using SA=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<Int<M>,Int<K>>{}));
using SW=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<Int<N>,Int<K>>{}));
using SB=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<_16,Int<K>>{}));
using SOQ=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Elem>{},Shape<Int<M>,_32>{}));
using SOG=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<Int<M>,Int<N>>{}));
using G2=Shape<int,int>;using D2=Stride<int64_t,_1>;
using GQ=Shape<int,int,int,int>;using DQ=Stride<int64_t,_1,int64_t,int64_t>;
using GG=Shape<int,int,int>;using DG=Stride<int64_t,_1,int64_t>;
using TA=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G2{},D2{}),SA{},Shape<Int<M>,Int<K>>{},_1{}));
using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G2{},D2{}),SW{},Shape<Int<N>,Int<K>>{},_1{}));
using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G2{},D2{}),SB{},Shape<_16,Int<K>>{},_1{}));
using TQ=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)nullptr),GQ{},DQ{}),SOQ{},Shape<Int<M>,_32>{},_1{}));
using TG=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)nullptr),GG{},DG{}),SOG{},Shape<Int<M>,Int<N>>{},_1{}));
using MMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Elem,Elem,float,Shape<Int<M>,Int<N>,Int<K>>>(),Layout<Shape<_1,_1,_1>>{}));
using MMAB=decltype(make_tiled_mma(GMMA::ss_op_selector<Elem,Elem,float,Shape<Int<M>,_16,Int<K>>>(),Layout<Shape<_1,_1,_1>>{}));
struct Params {TA za;TW weights;TB wb;TQ qo,ko,vo;TG go;G2 zshape,wshape,bshape;GQ qshape;GG gshape;Elem const* lnw;Elem const* lnb;float* bias;float eps;int L;};
struct Shared {
 alignas(128) Elem a[M*K];
 alignas(128) Elem w[2][N*K];
 alignas(128) Elem b[16*K];
 alignas(128) Elem out[2][M*N];
 cutlass::arch::ClusterTransactionBarrier ready_a,ready_b,ready_w[2];
 cutlass::arch::ClusterBarrier normalized,empty_w[2],ready_out[2],empty_out[2];
};
__device__ __forceinline__ float sum8(float const (&x)[8]) {
 float sum=__fadd_rn(x[0],x[1]);
 #pragma unroll
 for(int n=2;n<8;n++)sum=__fadd_rn(sum,x[n]);
 return sum;
}
template<int L>
__global__ __launch_bounds__(256,2) void prologue_tma(CUTE_GRID_CONSTANT Params const p) {
 extern __shared__ __align__(128) unsigned char storage[];
 Shared& s=*reinterpret_cast<Shared*>(storage);
 int tid=threadIdx.x, tile=blockIdx.x;
 int lane_tid=tid%128;
 int i=(tile*M)/L,j=(tile*M)%L;
 auto a=make_tensor(make_smem_ptr(s.a),SA{});
 auto b=make_tensor(make_smem_ptr(s.b),SB{});
 auto wa=make_tensor(make_smem_ptr(s.w[0]),SW{});
 auto wb=make_tensor(make_smem_ptr(s.w[1]),SW{});
 if(tid==0){
  s.ready_a.init(1);s.ready_b.init(1);s.ready_w[0].init(1);s.ready_w[1].init(1);
  s.normalized.init(1);
  for(int st=0;st<2;st++){s.empty_w[st].init(1);s.ready_out[st].init(1);s.empty_out[st].init(1);}
  cutlass::arch::fence_barrier_init();
 }
 __syncthreads();
 auto load_weight=[&](int chunk,int stage){
  auto dest=make_tensor(make_smem_ptr(s.w[stage]),SW{});
  auto src=local_tile(p.weights.get_tma_tensor(Shape<_512,_128>{}),Shape<Int<N>,Int<K>>{},make_coord(chunk,0));
  auto slice=p.weights.get_slice(_0{});
  s.ready_w[stage].arrive_and_expect_tx(N*K*sizeof(Elem));
  copy(p.weights.with(reinterpret_cast<uint64_t&>(s.ready_w[stage])),slice.partition_S(src),slice.partition_D(dest));
 };
 if(tid<128){
 if(tid==0){
  auto src=local_tile(p.za.get_tma_tensor(make_shape(Int<L*L>{},_128{})),Shape<Int<M>,Int<K>>{},make_coord(tile,0));
  auto slice=p.za.get_slice(_0{});
  s.ready_a.arrive_and_expect_tx(M*K*sizeof(Elem));
  copy(p.za.with(reinterpret_cast<uint64_t&>(s.ready_a)),slice.partition_S(src),slice.partition_D(a));
  auto bsrc=local_tile(p.wb.get_tma_tensor(Shape<_16,_128>{}),Shape<_16,Int<K>>{},make_coord(0,0));
  auto bs=p.wb.get_slice(_0{});
  s.ready_b.arrive_and_expect_tx(16*K*sizeof(Elem));
  copy(p.wb.with(reinterpret_cast<uint64_t&>(s.ready_b)),bs.partition_S(bsrc),bs.partition_D(b));
  load_weight(0,0);load_weight(1,1);
 }
 s.ready_a.wait(0);asm volatile("":::"memory");
 // Sixteen lanes per row, eight adjacent channels per lane: match the established LN reduction layout.
 int lane=tid%16;
 #pragma unroll 1
 for(int m=tid/16;m<M;m+=8){
  float z[8],sq[8];
  #pragma unroll
  for(int c=0;c<8;c++)z[c]=float(a(m,lane*8+c));
  float mean=sum8(z);
  #pragma unroll
  for(int d=8;d>0;d>>=1)mean=__fadd_rn(mean,__shfl_xor_sync(0xffffffff,mean,d,16));
  mean=__fmul_rn(mean,1.f/128.f);
  #pragma unroll
  for(int c=0;c<8;c++){z[c]=__fsub_rn(z[c],mean);sq[c]=__fmul_rn(z[c],z[c]);}
  float var=sum8(sq);
  #pragma unroll
  for(int d=8;d>0;d>>=1)var=__fadd_rn(var,__shfl_xor_sync(0xffffffff,var,d,16));
  float inv=__fdividef(1.f,__fsqrt_rn(__fadd_rn(__fmul_rn(var,1.f/128.f),p.eps)));
  #pragma unroll
  for(int c=0;c<8;c++)a(m,lane*8+c)=Elem(__fmaf_rn(__fmul_rn(z[c],inv),float(p.lnw[lane*8+c]),float(p.lnb[lane*8+c])));
 }
 cutlass::arch::fence_view_async_shared();
 cutlass::arch::NamedBarrier::sync(128,1);
 if(tid==0)s.normalized.arrive();
 if(tid==0){
  #pragma unroll 1
  for(int chunk=0;chunk<8;chunk++){
   int stage=chunk&1;
   s.ready_out[stage].wait((chunk/2)&1);asm volatile("":::"memory");
   if(chunk<6){
    auto const& tma=chunk<2?p.qo:(chunk<4?p.ko:p.vo);
    #pragma unroll
    for(int h=0;h<2;h++){
     auto src=make_tensor(make_smem_ptr(s.out[stage]+h*(M*32)),SOQ{});
     auto global=tma.get_tma_tensor(make_shape(Int<L>{},_32{},_4{},Int<L>{}))(_,_,(chunk%2)*2+h,i);
     auto dst=local_tile(global,Shape<Int<M>,_32>{},make_coord(j/M,0));
     auto slice=tma.get_slice(_0{});
     copy(tma,slice.partition_S(src),slice.partition_D(dst));
    }
   }else{
    auto og=make_tensor(make_smem_ptr(s.out[stage]),SOG{});
    auto global=p.go.get_tma_tensor(make_shape(Int<L*L>{},_64{},_2{}))(_,_,chunk-6);
    auto dst=local_tile(global,Shape<Int<M>,Int<N>>{},make_coord(tile,0));
    auto slice=p.go.get_slice(_0{});
    copy(p.go,slice.partition_S(og),slice.partition_D(dst));
   }
   tma_store_arrive();
   if(chunk+2<8){
    s.empty_w[stage].wait((chunk/2)&1);asm volatile("":::"memory");
    load_weight(chunk+2,stage);
   }
   tma_store_wait<0>();
   s.empty_out[stage].arrive();
  }
 }
 return;
 }
 s.normalized.wait(0);asm volatile("":::"memory");
 s.ready_b.wait(0);asm volatile("":::"memory");
 MMAB mb;auto tb=mb.get_slice(lane_tid);
 auto ba=tb.partition_fragment_A(a);auto bb=tb.partition_fragment_B(b);
 auto bc=partition_fragment_C(mb,Shape<Int<M>,_16>{});clear(bc);
 warpgroup_fence_operand(bc);warpgroup_arrive();
 #pragma unroll
 for(int kk=0;kk<size<2>(ba);kk++)gemm(mb,ba(_,_,kk),bb(_,_,kk),bc);
 warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(bc);
 auto bid=tb.partition_C(make_identity_tensor(Shape<Int<M>,_16>{}));
 #pragma unroll
 for(int n=0;n<size(bc);n++){
  int m=get<0>(bid(n)),h=get<1>(bid(n));
  if(h<4)p.bias[(int64_t(h)*L+i)*L+j+m]=float(Elem(bc(n)));
 }
 MMA mma;auto tm=mma.get_slice(lane_tid);auto ma=tm.partition_fragment_A(a);
 auto id=tm.partition_C(make_identity_tensor(Shape<Int<M>,Int<N>>{}));
 #pragma unroll 1
 for(int chunk=0;chunk<8;chunk++){
  constexpr int unused=0;(void)unused;
  int stage=chunk&1;
  s.ready_w[stage].wait((chunk/2)&1);asm volatile("":::"memory");
  auto weight=make_tensor(make_smem_ptr(s.w[stage]),SW{});
  auto mw=tm.partition_fragment_B(weight);
  auto acc=partition_fragment_C(mma,Shape<Int<M>,Int<N>>{});clear(acc);
  warpgroup_fence_operand(acc);warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<size<2>(ma);kk++)gemm(mma,ma(_,_,kk),mw(_,_,kk),acc);
  warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(acc);
  cutlass::arch::NamedBarrier::sync(128,2);
  if(lane_tid==0)s.empty_w[stage].arrive();
  if(chunk>=2){s.empty_out[stage].wait(((chunk/2)-1)&1);asm volatile("":::"memory");}
  auto oq=make_tensor(make_smem_ptr(s.out[stage]),SOQ{});
  auto og=make_tensor(make_smem_ptr(s.out[stage]),SOG{});
  #pragma unroll
  for(int n=0;n<size(acc);n++){
   int m=get<0>(id(n)),col=get<1>(id(n));
   if(chunk<6){auto oh=make_tensor(make_smem_ptr(s.out[stage]+(col/32)*(M*32)),SOQ{});oh(m,col%32)=Elem(acc(n));}
   else og(m,col)=Elem(acc(n));
  }
  cutlass::arch::fence_view_async_shared();
  cutlass::arch::NamedBarrier::sync(128,2);
  if(lane_tid==0)s.ready_out[stage].arrive();
 }

}
} // namespace

void prologue_cuda(at::Tensor z,at::Tensor w,at::Tensor wb,at::Tensor lnw,at::Tensor lnb,
                   at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor g,at::Tensor bias,double eps){
 using namespace ta_tma;
 TORCH_CHECK(z.is_cuda() && z.scalar_type()==at::kBFloat16 && z.is_contiguous(),"contiguous CUDA bf16 input required");
 int L=z.size(-2);TORCH_CHECK(z.numel()==int64_t(L)*L*128 && L%64==0,"C128 square pair, L multiple of 64 required");
 c10::cuda::CUDAGuard guard(z.device());
 auto zp=(Elem const*)z.data_ptr();auto wp=(Elem const*)w.data_ptr();auto bp=(Elem const*)wb.data_ptr();
 G2 zs=make_shape(L*L,128),ws=make_shape(512,128),bs=make_shape(16,128);
 GQ qs=make_shape(L,32,4,L);GG gs=make_shape(L*L,64,2);
 auto tz=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(zp),zs,D2{128,_1{}}),SA{},Shape<Int<M>,Int<K>>{},_1{});
 auto tw=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(wp),ws,D2{128,_1{}}),SW{},Shape<Int<N>,Int<K>>{},_1{});
 auto tb=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(bp),bs,D2{128,_1{}}),SB{},Shape<_16,Int<K>>{},_1{});
 auto make_q=[&](at::Tensor t){return make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)t.data_ptr()),qs,DQ{32,_1{},int64_t(L)*32,int64_t(4)*L*32}),SOQ{},Shape<Int<M>,_32>{},_1{});};
 auto tg=make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)g.data_ptr()),gs,DG{128,_1{},64}),SOG{},Shape<Int<M>,Int<N>>{},_1{});
 Params p{tz,tw,tb,make_q(q),make_q(k),make_q(v),tg,zs,ws,bs,qs,gs,(Elem const*)lnw.data_ptr(),(Elem const*)lnb.data_ptr(),bias.data_ptr<float>(),float(eps),L};
 #define LAUNCH(LEN) do { \
 C10_CUDA_CHECK(cudaFuncSetAttribute(prologue_tma<LEN>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared))); \
 prologue_tma<LEN><<<L*L/M,256,sizeof(Shared),at::cuda::getCurrentCUDAStream()>>>(p); \
 } while(0)
 if(L==384){LAUNCH(384);}else if(L==768){LAUNCH(768);}else if(L==1024){LAUNCH(1024);}else{TORCH_CHECK(false,"unsupported L");}
 #undef LAUNCH
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
