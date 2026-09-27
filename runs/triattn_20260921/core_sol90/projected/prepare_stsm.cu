#include "checks.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>
#include <cuda_bf16.h>

namespace triattn_projected_stsm_prepare {
using namespace cute;
using Elem=cutlass::bfloat16_t;
constexpr int M=128,K=128,N=128;
using SA=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<Int<M>,Int<K>>{}));
using SW=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<Int<N>,Int<K>>{}));
using SB=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<_16,Int<K>>{}));
using SOQ=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Elem>{},Shape<Int<M>,_32>{}));
using SOQQ=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Elem>{},Shape<Int<M>,Int<N>>{},Step<_1,_2>{}));
using SOG=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Elem>{},Shape<Int<M>,Int<N>>{}));
using GZ=Shape<int,int,int>;using DZ=Stride<int64_t,_1,int64_t>;
using G2=Shape<int,int>;using D2=Stride<int64_t,_1>;
using GQ=Shape<int,int,int,int>;using DQ=Stride<int64_t,_1,int64_t,int64_t>;
using GG=Shape<int,int,int>;using DG=Stride<int64_t,_1,int64_t>;
using TA=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),GZ{},DZ{}),SA{},Shape<Int<M>,Int<K>>{},_1{}));
using TW=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G2{},D2{}),SW{},Shape<Int<N>,Int<K>>{},_1{}));
using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Elem const*)nullptr),G2{},D2{}),SB{},Shape<_16,Int<K>>{},_1{}));
using TQ=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)nullptr),GQ{},DQ{}),SOQ{},Shape<Int<M>,_32>{},_1{}));
using TG=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)nullptr),GG{},DG{}),SOG{},Shape<Int<M>,Int<N>>{},_1{}));
using MMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Elem,Elem,float,Shape<Int<M>,Int<N>,Int<K>>>(),Layout<Shape<_2,_1,_1>>{}));
using MMAB=decltype(make_tiled_mma(GMMA::ss_op_selector<Elem,Elem,float,Shape<Int<M>,_16,Int<K>>>(),Layout<Shape<_2,_1,_1>>{}));
struct Params {TA za;TW weights;TB wb;TQ qo,ko,vo;TG go;GZ zshape;G2 wshape,bshape;GQ qshape;GG gshape;Elem const* lnw;Elem const* lnb;float* bias;float eps;int L;Elem* norm;};
struct Shared {
 alignas(128) Elem a[M*K];
 alignas(128) Elem w[1][N*K];
 alignas(128) Elem b[16*K];
 alignas(128) Elem out[1][M*N];
 cutlass::arch::ClusterTransactionBarrier ready_a,ready_b,ready_w[2];
};
__device__ __forceinline__ float sum8(float const (&x)[8]) {
 float a0=__fadd_rn(x[0],x[4]),a1=__fadd_rn(x[1],x[5]),a2=__fadd_rn(x[2],x[6]),a3=__fadd_rn(x[3],x[7]);
 return __fadd_rn(__fadd_rn(a0,a2),__fadd_rn(a1,a3));
}
__global__ __launch_bounds__(256,2) void prologue_tma(CUTE_GRID_CONSTANT Params const p) {
 extern __shared__ __align__(128) unsigned char storage[];
 Shared& s=*reinterpret_cast<Shared*>(storage);
 int tid=threadIdx.x, tile=blockIdx.x;
 int i=(tile*M)/p.L,j=(tile*M)%p.L;
 auto a=make_tensor(make_smem_ptr(s.a),SA{});
 auto b=make_tensor(make_smem_ptr(s.b),SB{});
 if(tid==0){
  s.ready_a.init(1);s.ready_b.init(1);s.ready_w[0].init(1);s.ready_w[1].init(1);
  cutlass::arch::fence_barrier_init();
 }
 __syncthreads();
 auto load_weight=[&](int chunk,int stage){
  auto dest=make_tensor(make_smem_ptr(s.w[stage]),SW{});
  auto src=local_tile(p.weights.get_tma_tensor(p.wshape),Shape<Int<N>,Int<K>>{},make_coord(chunk,0));
  auto slice=p.weights.get_slice(_0{});
  s.ready_w[stage].arrive_and_expect_tx(N*K*sizeof(Elem));
  copy(p.weights.with(reinterpret_cast<uint64_t&>(s.ready_w[stage])),slice.partition_S(src),slice.partition_D(dest));
 };
 if(tid==0){
  auto src=local_tile(p.za.get_tma_tensor(p.zshape)(_,_,i),Shape<Int<M>,Int<K>>{},make_coord(j/M,0));
  auto slice=p.za.get_slice(_0{});
  s.ready_a.arrive_and_expect_tx(M*K*sizeof(Elem));
  copy(p.za.with(reinterpret_cast<uint64_t&>(s.ready_a)),slice.partition_S(src),slice.partition_D(a));
  auto bsrc=local_tile(p.wb.get_tma_tensor(p.bshape),Shape<_16,Int<K>>{},make_coord(0,0));
  auto bs=p.wb.get_slice(_0{});
  s.ready_b.arrive_and_expect_tx(16*K*sizeof(Elem));
  copy(p.wb.with(reinterpret_cast<uint64_t&>(s.ready_b)),bs.partition_S(bsrc),bs.partition_D(b));
  load_weight(3,0);
 }
 s.ready_a.wait(0);asm volatile("":::"memory");
 // Sixteen lanes per row, eight adjacent channels per lane: match the established LN reduction layout.
 int lane=tid%16;
 uint4 gamma_bits=*reinterpret_cast<uint4 const*>(p.lnw+lane*8);
 uint4 beta_bits=*reinterpret_cast<uint4 const*>(p.lnb+lane*8);
 float gamma[8],beta[8];
 #pragma unroll
 for(int c=0;c<8;c++){gamma[c]=float(reinterpret_cast<Elem*>(&gamma_bits)[c]);beta[c]=float(reinterpret_cast<Elem*>(&beta_bits)[c]);}
 #pragma unroll 2
 for(int m=tid/16;m<M;m+=16){
  float z[8],sq[8];
  uint4 input_bits=*reinterpret_cast<uint4*>(&a(m,lane*8));
  #pragma unroll
  for(int c=0;c<8;c++)z[c]=float(reinterpret_cast<Elem*>(&input_bits)[c]);
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
  uint4 output_bits;
  #pragma unroll
  for(int c=0;c<8;c++)reinterpret_cast<Elem*>(&output_bits)[c]=Elem(__fmaf_rn(__fmul_rn(z[c],inv),gamma[c],beta[c]));
  *reinterpret_cast<uint4*>(&a(m,lane*8))=output_bits;
  *reinterpret_cast<uint4*>(p.norm+((int64_t(i)*p.L+j+m)*128)+lane*8)=output_bits;
 }
 cutlass::arch::fence_view_async_shared();__syncthreads();
 s.ready_b.wait(0);asm volatile("":::"memory");
 MMAB mb;auto tb=mb.get_slice(tid);
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
  if(h<4)p.bias[(int64_t(h)*p.L+i)*p.L+j+m]=float(Elem(bc(n)));
 }
 MMA mma;auto tm=mma.get_slice(tid);auto ma=tm.partition_fragment_A(a);
 #pragma unroll
 for(int chunk=3;chunk<4;chunk++){
  int stage=0;
  s.ready_w[stage].wait(0);asm volatile("":::"memory");
  auto weight=make_tensor(make_smem_ptr(s.w[stage]),SW{});
  auto mw=tm.partition_fragment_B(weight);
  auto acc=partition_fragment_C(mma,Shape<Int<M>,Int<N>>{});clear(acc);
  warpgroup_fence_operand(acc);warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<size<2>(ma);kk++)gemm(mma,ma(_,_,kk),mw(_,_,kk),acc);
  warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(acc);
  __syncthreads(); // Every warp has finished reading this weight stage before refill.
  if(tid==0 && chunk+1<4)load_weight(chunk+1,stage);
  if(tid==0 && chunk>=1)tma_store_wait<0>();
  __syncthreads();
  auto og=make_tensor(make_smem_ptr(s.out[stage]),SOG{});
  auto converted=make_tensor<Elem>(shape(acc));
  #pragma unroll
  for(int n=0;n<size(acc);n++)converted(n)=Elem(acc(n));
  auto r2s=make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N,Elem>{},mma);
  auto thread_copy=r2s.get_slice(tid);
  auto src=thread_copy.retile_S(converted);
  if(chunk<3){
   auto dst=make_tensor(make_smem_ptr(s.out[stage]),SOQQ{});
   copy(r2s,src,thread_copy.partition_D(dst));
  }else{
   copy(r2s,src,thread_copy.partition_D(og));
  }
  cutlass::arch::fence_view_async_shared();__syncthreads();
  if(tid==0){
   if(chunk<3){
    auto const& tma=chunk==0?p.qo:(chunk==1?p.ko:p.vo);
    #pragma unroll
    for(int h=0;h<4;h++){
     auto src=make_tensor(make_smem_ptr(s.out[stage]+h*(M*32)),SOQ{});
     auto global=tma.get_tma_tensor(p.qshape)(_,_,h,i);
     auto dst=local_tile(global,Shape<Int<M>,_32>{},make_coord(j/M,0));
     auto slice=tma.get_slice(_0{});
     copy(tma,slice.partition_S(src),slice.partition_D(dst));
    }
   }else{
    auto global=p.go.get_tma_tensor(p.gshape)(_,_,chunk-3);
    auto dst=local_tile(global,Shape<Int<M>,Int<N>>{},make_coord(tile,0));
    auto slice=p.go.get_slice(_0{});
    copy(p.go,slice.partition_S(og),slice.partition_D(dst));
   }
   tma_store_arrive();
  }
 }
 if(tid==0)tma_store_wait<0>();
}
} // namespace

void projected_stsm_prepare_cuda(at::Tensor z,at::Tensor w,at::Tensor wb,at::Tensor lnw,at::Tensor lnb,
                   at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor g,at::Tensor bias,double eps,bool ending){
 using namespace triattn_projected_stsm_prepare;
 int L=check_pair(z);
 check_bf16(w,z,512*128,"projection weight");check_bf16(wb,z,16*128,"bias weight");
 check_bf16(lnw,z,128,"LN weight");check_bf16(lnb,z,128,"LN bias");
 for(auto const& t:{q,k,v,g})check_bf16(t,z,int64_t(L)*L*128,"projection output");
 TORCH_CHECK(bias.is_cuda() && bias.device()==z.device() && bias.scalar_type()==at::kFloat && bias.is_contiguous() && bias.numel()==int64_t(4)*L*L,"invalid bias output");
 c10::cuda::CUDAGuard guard(z.device());
 auto zp=(Elem const*)z.data_ptr();auto wp=(Elem const*)w.data_ptr();auto bp=(Elem const*)wb.data_ptr();
 GZ zs=make_shape(L,128,L);G2 ws=make_shape(512,128),bs=make_shape(16,128);
 GQ qs=make_shape(L,32,4,L);GG gs=make_shape(L*L,128,1);
 auto tz=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(zp),zs,DZ{ending?int64_t(L)*128:128,_1{},ending?128:int64_t(L)*128}),SA{},Shape<Int<M>,Int<K>>{},_1{});
 auto tw=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(wp),ws,D2{128,_1{}}),SW{},Shape<Int<N>,Int<K>>{},_1{});
 auto tb=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(bp),bs,D2{128,_1{}}),SB{},Shape<_16,Int<K>>{},_1{});
 auto make_q=[&](at::Tensor t){return make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)t.data_ptr()),qs,DQ{32,_1{},int64_t(L)*32,int64_t(4)*L*32}),SOQ{},Shape<Int<M>,_32>{},_1{});};
 auto tg=make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Elem*)g.data_ptr()),gs,DG{128,_1{},128}),SOG{},Shape<Int<M>,Int<N>>{},_1{});
 Params p{tz,tw,tb,make_q(q),make_q(k),make_q(v),tg,zs,ws,bs,qs,gs,(Elem const*)lnw.data_ptr(),(Elem const*)lnb.data_ptr(),bias.data_ptr<float>(),float(eps),L,(Elem*)q.data_ptr()};
 C10_CUDA_CHECK(cudaFuncSetAttribute(prologue_tma,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 prologue_tma<<<L*L/M,256,sizeof(Shared),at::cuda::getCurrentCUDAStream()>>>(p);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
