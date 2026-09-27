#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_types.h>
#include <cuda_bf16.h>
#include "r1/csrc/fa3_utils.h"
namespace ta_core64 {
using namespace cute;using Element=cutlass::bfloat16_t;
constexpr int M=64,N=64,D=32,Rows=2;
using SQ=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));
using SV=decltype(composition(SQ{},make_ordered_layout(Shape<_32,_64>{},Step<_2,_1>{})));
using SB=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<float>{},Shape<_64,_64>{}));
using S1=decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<Element>{},Shape<_8,_64>{}));
using G3=Shape<int,int,int>;using ST=Stride<int64_t,_1,int64_t>;
using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),G3{},ST{}),SQ{},Shape<_64,_32>{},_1{}));
using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((float const*)nullptr),G3{},ST{}),SB{},Shape<_64,_64>{},_1{}));
using QK=decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>{}));
using PV=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));
using LS=decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>{}));
struct Params{TQ q,k,v;TB bias;Element* out;int* fix;int* valid;int L;float scale;};
struct Shared{
 alignas(128) Element q[Rows][M*D],k[2][Rows][N*D],v[2][Rows][N*D],ones[8*N];
 alignas(128) float bias[2][M*N];
 cutlass::arch::ClusterTransactionBarrier qr,full[2];
 cutlass::arch::ClusterBarrier empty[2];int bad;
};
__device__ __forceinline__ float ex2(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
template<bool Safe> __global__ __launch_bounds__(288,2) void attention(CUTE_GRID_CONSTANT Params const p){
 extern __shared__ __align__(128) unsigned char memory[];auto& s=*reinterpret_cast<Shared*>(memory);
 int tid=threadIdx.x,nt=p.L/M,nr=(p.L+Rows-1)/Rows;
 if(!*p.valid)return;
 int tile=blockIdx.x;
 if constexpr(Safe){if(tile>=p.fix[0])return;tile=p.fix[1+tile];}
 int qt=tile%nt,rg=tile/nt%nr,h=tile/(nt*nr),i0=rg*Rows;
 if(tid==0){s.qr.init(1);s.bad=0;for(int st=0;st<2;st++){s.full[st].init(1);s.empty[st].init(8);}cutlass::arch::fence_barrier_init();}
 for(int x=tid;x<8*N;x+=288)s.ones[x]=Element(1.f);
 cutlass::arch::fence_view_async_shared();__syncthreads();
 if(tid>=256){
  if(tid==256){
   s.qr.arrive_and_expect_tx(Rows*M*D*sizeof(Element));
   for(int r=0;r<Rows;r++){
    auto src=local_tile(p.q.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,min(i0+r,p.L-1)*4+h),Shape<_64,_32>{},make_coord(qt,0));
    auto dst=make_tensor(make_smem_ptr(s.q[r]),SQ{});auto sl=p.q.get_slice(_0{});
    copy(p.q.with(reinterpret_cast<uint64_t&>(s.qr)),sl.partition_S(src),sl.partition_D(dst));
   }
   for(int seq=0;seq<nt;seq++){
    int st=seq&1;if(seq>=2){s.empty[st].wait(((seq/2)-1)&1);asm volatile("":::"memory");}
    s.full[st].arrive_and_expect_tx((2*Rows*N*D*sizeof(Element))+M*N*sizeof(float));
    for(int r=0;r<Rows;r++){
     int ih=min(i0+r,p.L-1)*4+h;
     auto ks=local_tile(p.k.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,ih),Shape<_64,_32>{},make_coord(seq,0));
     auto vs=local_tile(p.v.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,ih),Shape<_64,_32>{},make_coord(seq,0));
     auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SQ{}),vd=make_tensor(make_smem_ptr(s.v[st][r]),SQ{});auto kl=p.k.get_slice(_0{}),vl=p.v.get_slice(_0{});
     copy(p.k.with(reinterpret_cast<uint64_t&>(s.full[st])),kl.partition_S(ks),kl.partition_D(kd));
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[st])),vl.partition_S(vs),vl.partition_D(vd));
    }
    auto bs=local_tile(p.bias.get_tma_tensor(make_shape(p.L,p.L,4))(_,_,h),Shape<_64,_64>{},make_coord(qt,seq));
    auto bd=make_tensor(make_smem_ptr(s.bias[st]),SB{});auto bl=p.bias.get_slice(_0{});
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.full[st])),bl.partition_S(bs),bl.partition_D(bd));
   }
  }
  return;
 }
 int wg=tid/128,t=tid%128;QK qk;PV pv;LS ls;
 auto tq=qk.get_slice(t);auto tp=pv.get_slice(t);auto tl=ls.get_slice(t);
 auto sq=make_tensor(make_smem_ptr(s.q[wg]),SQ{});auto qa=tq.partition_fragment_A(sq);
 auto ones=make_tensor(make_smem_ptr(s.ones),S1{});auto lb=tl.partition_fragment_B(ones);
 auto score=partition_fragment_C(qk,Shape<_64,_64>{});
 auto acc=partition_fragment_C(pv,Shape<_64,_32>{});auto sum=partition_fragment_C(ls,Shape<_64,_8>{});clear(acc);clear(sum);
 auto sr=make_tensor(score.data(),flash::convert_layout_acc_rowcol(score.layout()));
 auto ar=make_tensor(acc.data(),flash::convert_layout_acc_rowcol(acc.layout()));auto lr=make_tensor(sum.data(),flash::convert_layout_acc_rowcol(sum.layout()));
 auto pp=make_tensor_like<Element>(make_tensor(score.data(),flash::convert_layout_acc_Aregs<PV>(score.layout())));
 float mx[2]={-INFINITY,-INFINITY};float c=p.scale*1.4426950408889634f;
 s.qr.wait(0);asm volatile("":::"memory");
 for(int seq=0;seq<nt;seq++){
  int st=seq&1;s.full[st].wait((seq/2)&1);asm volatile("":::"memory");
  auto bias=make_tensor(make_smem_ptr(s.bias[st]),SB{});copy(tq.partition_C(bias),score);
  auto sk=make_tensor(make_smem_ptr(s.k[st][wg]),SQ{});auto kb=tq.partition_fragment_B(sk);
  warpgroup_fence_operand(score);warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<2;kk++)gemm(qk,qa(_,_,kk),kb(_,_,kk),score);
  warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(score);
  #pragma unroll
  for(int row=0;row<2;row++){
   float m=sr(row,0);
   #pragma unroll
   for(int col=1;col<16;col++)m=fmaxf(m,sr(row,col));
   m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));m*=c;
   if constexpr(Safe){
    float next=fmaxf(mx[row],m);float alpha=mx[row]==-INFINITY?0.f:ex2(mx[row]-next);
    #pragma unroll
    for(int col=0;col<8;col++)ar(row,col)*=alpha;
    lr(row,0)*=alpha;lr(row,1)*=alpha;mx[row]=next;
   }else{if(mx[row]==-INFINITY && m!=-INFINITY)mx[row]=m+64.f;}
   #pragma unroll
   for(int col=0;col<16;col++){float x=sr(row,col);sr(row,col)=x==-INFINITY?0.f:ex2(fmaf(x,c,-mx[row]));}
  }
  auto packed=recast<uint32_t>(pp);
  #pragma unroll
  for(int n=0;n<16;n++){auto x=__floats2bfloat162_rn(score(2*n),score(2*n+1));packed(n)=reinterpret_cast<uint32_t const&>(x);}
  auto sv=make_tensor(make_smem_ptr(s.v[st][wg]),SV{});auto vb=tp.partition_fragment_B(sv);
  warpgroup_fence_operand(pp);warpgroup_fence_operand(acc);warpgroup_fence_operand(sum);warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<4;kk++)gemm(pv,pp(_,_,kk),vb(_,_,kk),acc);
  #pragma unroll
  for(int kk=0;kk<4;kk++)gemm(ls,pp(_,_,kk),lb(_,_,kk),sum);
  warpgroup_commit_batch();warpgroup_wait<0>();warpgroup_fence_operand(acc);warpgroup_fence_operand(sum);
  // Four arrivals per WG: every warp has completed all operand reads.
  if(t%32==0)s.empty[st].arrive();
 }
 if constexpr(!Safe){
  bool bad=false;
  #pragma unroll
  for(int row=0;row<2;row++)bad|=!(lr(row,0)>0x1p-84f && lr(row,0)<0x1p-44f);
  if(__any_sync(0xffffffff,bad) && t%32==0 && atomicCAS(&s.bad,0,1)==0){int index=atomicAdd(p.fix,1);p.fix[index+1]=tile;}
 }
 auto id=tp.partition_C(make_identity_tensor(Shape<_64,_32>{}));
 auto out=make_tensor_like<Element>(acc);
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
 int x=blockIdx.x*256+threadIdx.x,total=4*L*L;
 if(x<total){int k=x%L;out[x]=(!mask || mask[int64_t(k)*stride])?bias[x]*inv:-INFINITY;}
 if(blockIdx.x==0){bool any=!mask;for(int k=threadIdx.x;k<L;k+=256)any|=mask && mask[int64_t(k)*stride];int yes=__syncthreads_or(any);if(threadIdx.x==0){*valid=yes;*fix=0;}}
}
__global__ void uniform(Element const* v,Element* out,int L,int const* valid){
 if(*valid)return;int i=blockIdx.x,h=blockIdx.y,d=threadIdx.x%32,g=threadIdx.x/32;__shared__ float part[4][32];
 float sum=0;for(int k=g;k<L;k+=4)sum+=float(v[((int64_t(i)*4+h)*L+k)*32+d]);part[g][d]=sum;__syncthreads();Element mean=Element((part[0][d]+part[1][d]+part[2][d]+part[3][d])/float(L));
 for(int q=g;q<L;q+=4)out[((int64_t(i)*4+h)*L+q)*32+d]=mean;
}
}
at::Tensor core64_cuda(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor bias,c10::optional<at::Tensor> mask,double scale){
 using namespace ta_core64;int L=q.size(-2);
 TORCH_CHECK((L==384 || L==768 || L==1024) && q.numel()==int64_t(L)*4*L*32 && q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous(),"C128/H4/D32 contiguous square BF16 operands required");
 for(auto const& x:{k,v})TORCH_CHECK(x.device()==q.device() && x.sizes()==q.sizes() && x.scalar_type()==q.scalar_type() && x.is_contiguous(),"unsupported K/V");
 TORCH_CHECK(bias.device()==q.device() && bias.scalar_type()==at::kFloat && bias.is_contiguous() && bias.numel()==4*L*L,"contiguous float bias required");
 bool const* mp=nullptr;int64_t ms=1;
 if(mask){auto m=*mask;TORCH_CHECK(m.device()==q.device() && m.scalar_type()==at::kBool && m.dim()==5 && m.size(0)==1 && m.size(1)==L && m.size(2)==1 && m.size(3)==1 && m.size(4)==L && m.stride(1)==0,"row-broadcast mask required");mp=m.data_ptr<bool>();ms=m.stride(4);}
 c10::cuda::CUDAGuard guard(q.device());auto output=at::empty_like(q),prepared=at::empty_like(bias),fix=at::empty({1+(L/M)*((L+Rows-1)/Rows)*4},q.options().dtype(at::kInt)),valid=at::empty({1},fix.options());
 auto stream=at::cuda::getCurrentCUDAStream();prepare<<<(4*L*L+255)/256,256,0,stream>>>(bias.data_ptr<float>(),mp,ms,float(1./scale),L,prepared.data_ptr<float>(),valid.data_ptr<int>(),fix.data_ptr<int>());
 auto tq=[&](at::Tensor x){return make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),make_shape(L,32,L*4),ST{32,_1{},int64_t(L)*32}),SQ{},Shape<_64,_32>{},_1{});};
 auto tb=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(prepared.data_ptr<float>()),make_shape(L,L,4),ST{L,_1{},int64_t(L)*L}),SB{},Shape<_64,_64>{},_1{});
 Params p{tq(q),tq(k),tq(v),tb,(Element*)output.data_ptr(),fix.data_ptr<int>(),valid.data_ptr<int>(),L,float(scale)};
 int ct=(L/M)*((L+Rows-1)/Rows)*4;
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 attention<false><<<ct,288,sizeof(Shared),stream>>>(p);attention<true><<<ct,288,sizeof(Shared),stream>>>(p);
 uniform<<<dim3(L,4),128,0,stream>>>((Element const*)v.data_ptr(),(Element*)output.data_ptr(),L,valid.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
