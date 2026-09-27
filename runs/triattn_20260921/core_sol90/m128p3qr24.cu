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

namespace ta_sol_m128p3qr24 {
using namespace cute;
using Element=cutlass::bfloat16_t;
constexpr int M=128,N=32,LN=128,Ratio=LN/N,D=32,Rows=3,Stages=2;
constexpr int Consumers=Rows*128,Threads=Consumers+128;
using SQ=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_128,_32>{}));
using SK=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<Int<LN>,_32>{}));
using SV=decltype(composition(SK{},make_ordered_layout(Shape<_32,Int<LN>>{},Step<_2,_1>{})));
using SB=Layout<Shape<_256,Int<N/4>>,Stride<_1,_256>>;
using GB=Shape<int,int,int,int,int>;
using DB=Stride<_1,_256,int64_t,int64_t,int64_t>;
using S1=decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<Element>{},Shape<_8,Int<N>>{}));
using G3=Shape<int,int,int>;
using ST=Stride<int64_t,_1,int64_t>;
using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),G3{},ST{}),SQ{},Shape<_128,_32>{},_1{}));
using TK=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),G3{},ST{}),SK{},Shape<Int<LN>,_32>{},_1{}));
using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((float const*)nullptr),GB{},DB{}),SB{},Shape<_256,Int<N/4>>{},_1{}));
using QK=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,Int<N>,_32>>()));
using QR=decltype(make_tiled_mma(GMMA::rs_op_selector<Element,Element,float,Shape<_64,Int<N>,_32>>()));
using PV=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));
using LS=decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>{}));
struct Params { TQ q;TK k,v;TB bias;Element* out;int* fix;int* valid;int L;float scale; };
struct Shared {
 alignas(128) Element q[Rows][M*D],k[Stages][Rows][LN*D],v[Stages][Rows][LN*D],ones[8*N];
 alignas(128) float bias[8][64*N];
 cutlass::arch::ClusterTransactionBarrier qr,full[Stages],bf[8];
 cutlass::arch::ClusterBarrier empty[Stages],be[8];int bad;
};
__device__ __forceinline__ float ex2(float x) { float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y; }

template<bool Safe> __global__ __launch_bounds__(Threads,1)
void attention(CUTE_GRID_CONSTANT Params const p) {
 extern __shared__ __align__(128) unsigned char memory[];
 auto& s=*reinterpret_cast<Shared*>(memory);
 int tid=threadIdx.x,nq=768/M,nk=p.L/N,nr=(768+Rows-1)/Rows;
 if(!*p.valid)return;
 int tile=blockIdx.x;
 if constexpr(Safe) { if(tile>=p.fix[0])return;tile=p.fix[1+tile]; }
 int qt=tile%nq,rg=tile/nq%nr,h=tile/(nq*nr),i0=rg*Rows;
 if(tid==0) {
  s.qr.init(1);for(int st=0;st<8;st++){s.bf[st].init(1);s.be[st].init(Rows*4);}s.bad=0;
  for(int st=0;st<Stages;st++){s.full[st].init(1);s.empty[st].init(Rows*4);}
  cutlass::arch::fence_barrier_init();
 }
 for(int x=tid;x<8*N;x+=Threads)s.ones[x]=Element(1.f);
 cutlass::arch::fence_view_async_shared();__syncthreads();
 if(tid>=Consumers) {
  if constexpr(160>0)cutlass::arch::warpgroup_reg_dealloc<32>();
  if(tid==Consumers) {
   s.qr.arrive_and_expect_tx(Rows*M*D*sizeof(Element));
   for(int r=0;r<Rows;r++) {
    auto src=local_tile(p.q.get_tma_tensor(make_shape(768,32,768*4))(_,_,min(i0+r,768-1)*4+h),Shape<_128,_32>{},make_coord(qt,0));
    auto dst=make_tensor(make_smem_ptr(s.q[r]),SQ{});auto sl=p.q.get_slice(_0{});
    copy(p.q.with(reinterpret_cast<uint64_t&>(s.qr)),sl.partition_S(src),sl.partition_D(dst));
   }
   for(int seq=0;seq<768/LN;seq++) {
    int st=seq%Stages;
    if(seq>=Stages){s.empty[st].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
    s.full[st].arrive_and_expect_tx(2*Rows*LN*D*sizeof(Element));
    for(int r=0;r<Rows;r++) {
     int ih=min(i0+r,768-1)*4+h;
     auto ks=local_tile(p.k.get_tma_tensor(make_shape(768,32,768*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto vs=local_tile(p.v.get_tma_tensor(make_shape(768,32,768*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{}),vd=make_tensor(make_smem_ptr(s.v[st][r]),SK{});
     auto kl=p.k.get_slice(_0{}),vl=p.v.get_slice(_0{});
     copy(p.k.with(reinterpret_cast<uint64_t&>(s.full[st])),kl.partition_S(ks),kl.partition_D(kd));
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[st])),vl.partition_S(vs),vl.partition_D(vd));
    }

   }
  }
  if(tid==Consumers+32) {
   for(int seq=0;seq<2*768/N;seq++) {
    int st=seq%8;
    if(seq>=8){s.be[st].wait(((seq/8)-1)&1);asm volatile("":::"memory");}
    s.bf[st].arrive_and_expect_tx(64*N*sizeof(float));
    auto bs=p.bias.get_tma_tensor(make_shape(256,N/4,2*768/N,nq,4))(_,_,seq,qt,h);
    auto bd=make_tensor(make_smem_ptr(s.bias[st]),SB{});auto bl=p.bias.get_slice(_0{});
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bf[st])),bl.partition_S(bs),bl.partition_D(bd));
   }
  }
  return;
 }
 if constexpr(160>0)cutlass::arch::warpgroup_reg_alloc<160>();
 int wg=int(__reduce_max_sync(0xffffffffu,unsigned(tid)/128u)),t=tid%128;
 QK qk;PV pv;LS ls;auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);auto tl=ls.get_slice(0);
 using Score=decltype(partition_fragment_C(qk,Shape<_64,Int<N>>{}));
 using Output=decltype(partition_fragment_C(pv,Shape<_64,_32>{}));
 Score score;Score next_score;Score third_score;Output acc[2];clear(acc[0]);clear(acc[1]);
 auto pp=make_tensor_like<Element>(make_tensor(score.data(),flash::convert_layout_acc_Aregs<PV>(score.layout())));
 auto pp1=make_fragment_like(pp);auto pp2=make_fragment_like(pp);
 using Den=decltype(partition_fragment_C(ls,Shape<_64,_8>{}));
 Den den[2];clear(den[0]);clear(den[1]);
 auto lb=tl.partition_fragment_B(make_tensor(make_smem_ptr(s.ones),S1{}));
 float mx[2][2]={{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f},{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f}};
 bool missing_seed=false;float c=0x1.6a09e6p-3f*1.4426950408889634f;
 warpgroup_fence_operand(acc[0]);warpgroup_fence_operand(acc[1]);warpgroup_fence_operand(den[0]);warpgroup_fence_operand(den[1]);
 s.qr.wait(0);asm volatile("":::"memory");
 QR qr;qr.accumulate_=GMMA::ScaleOut::One;
 auto qshared=make_tensor(make_smem_ptr(s.q[wg]),SQ{});
 auto tqreg=qr.get_thread_slice(t);
 auto qa0=tqreg.partition_fragment_A(local_tile(qshared,Shape<_64,_32>{},make_coord(0,0)));
 auto qa1=make_fragment_like(qa0);
 auto qcopy=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},qr);
 auto qtcopy=qcopy.get_thread_slice(t);
 #pragma unroll
 for(int hh=0;hh<2;hh++){
  auto qhalf=local_tile(qshared,Shape<_64,_32>{},make_coord(hh,0));
  auto qs=qtcopy.partition_S(qhalf);auto qd=qtcopy.retile_D(hh==0?qa0:qa1);
  copy(qcopy,qs,qd);
 }
 warpgroup_fence_operand(qa0);warpgroup_fence_operand(qa1);

 auto init_score=[&](auto& score,int seq) __attribute__((always_inline)) {
  if(seq<2*nk){s.bf[seq%8].wait((seq/8)&1);asm volatile("":::"memory");}
  #pragma unroll
  for(int u=0;u<N/8;u++){
   float4 b=*reinterpret_cast<float4 const*>(s.bias[seq%8]+u*512+t*4);
   score(4*u)=b.x;score(4*u+1)=b.y;score(4*u+2)=b.z;score(4*u+3)=b.w;
  }
 };
 auto issue_qk=[&](auto& score,int seq,auto half) __attribute__((always_inline)) {
  constexpr int hh=decltype(half)::value;
  int kt=seq/(2*Ratio),st=kt%Stages,cx=(seq/2)%Ratio;
  if(seq<2*nk && seq%(2*Ratio)==0){s.full[st].wait((kt/Stages)&1);asm volatile("":::"memory");}
  auto& qa=hh==0?qa0:qa1;
  auto kall=make_tensor(make_smem_ptr(s.k[st][wg]),SK{});
  auto ks=local_tile(kall,Shape<Int<N>,_32>{},make_coord(cx,0));auto kb=tq.partition_fragment_B(ks);
  warpgroup_fence_operand(score);warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<2;kk++)gemm(qr,qa(_,_,kk),kb(_,_,kk),score);
  warpgroup_commit_batch();
  if(seq<2*nk){__syncwarp();if(t%32==0)s.be[seq%8].arrive();}
 };
 auto exponentiate=[&](auto& score,auto half,auto seed) __attribute__((always_inline)) {
  constexpr int hh=decltype(half)::value;
  auto sr=make_tensor(score.data(),flash::convert_layout_acc_rowcol(score.layout()));
  auto ar=make_tensor(acc[hh].data(),flash::convert_layout_acc_rowcol(acc[hh].layout()));
  #pragma unroll
  for(int row=0;row<2;row++){
   if constexpr(Safe || decltype(seed)::value){
    float m=sr(row,0);
    #pragma unroll
    for(int col=1;col<N/4;col++)m=fmaxf(m,sr(row,col));
    m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));m*=c;
    if constexpr(Safe){
     float next=fmaxf(mx[hh][row],m),alpha=mx[hh][row]==-INFINITY?0.f:ex2(mx[hh][row]-next);
     #pragma unroll
     for(int col=0;col<8;col++)ar(row,col)*=alpha;
     auto lr=make_tensor(den[hh].data(),flash::convert_layout_acc_rowcol(den[hh].layout()));lr(row,0)*=alpha;lr(row,1)*=alpha;mx[hh][row]=next;
    }else{if(m!=-INFINITY)mx[hh][row]=m+64.f;else missing_seed=true;}
   }
   #pragma unroll
   for(int col=0;col<N/4;col++){
    float x=sr(row,col);
    if constexpr(Safe)sr(row,col)=x==-INFINITY?0.f:ex2(fmaf(x,c,-mx[hh][row]));
    else sr(row,col)=ex2(fmaf(x,c,-mx[hh][row]));
   }
  }
 };
 auto pack=[&](auto& score,auto& pp) __attribute__((always_inline)) {
  auto packed=recast<uint32_t>(pp);
  #pragma unroll
  for(int n=0;n<N/4;n++){auto x=__floats2bfloat162_rn(score(2*n),score(2*n+1));packed(n)=reinterpret_cast<uint32_t const&>(x);}
 };
 auto issue_pv=[&](int seq,auto half,auto& pp) __attribute__((always_inline)) {
  constexpr int hh=decltype(half)::value;
  auto vall=make_tensor(make_smem_ptr(s.v[(seq/(2*Ratio))%Stages][wg]),SV{});
  auto vs=local_tile(vall,Shape<_32,Int<N>>{},make_coord(0,(seq/2)%Ratio));auto vb=tp.partition_fragment_B(vs);
  warpgroup_fence_operand(pp);if constexpr(Safe){warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}
  warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<N/16;kk++)gemm(pv,pp(_,_,kk),vb(_,_,kk),acc[hh]);
  #pragma unroll
  for(int kk=0;kk<N/16;kk++)gemm(ls,pp(_,_,kk),lb(_,_,kk),den[hh]);
  warpgroup_commit_batch();
 };
 auto issue_pv_prefenced=[&](int seq,auto half,auto& pp) __attribute__((always_inline)) {
  constexpr int hh=decltype(half)::value;
  auto vall=make_tensor(make_smem_ptr(s.v[(seq/(2*Ratio))%Stages][wg]),SV{});
  auto vs=local_tile(vall,Shape<_32,Int<N>>{},make_coord(0,(seq/2)%Ratio));auto vb=tp.partition_fragment_B(vs);
  if constexpr(Safe){warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}
  #pragma unroll
  for(int kk=0;kk<N/16;kk++)gemm(pv,pp(_,_,kk),vb(_,_,kk),acc[hh]);
  #pragma unroll
  for(int kk=0;kk<N/16;kk++)gemm(ls,pp(_,_,kk),lb(_,_,kk),den[hh]);
  warpgroup_commit_batch();
 };
 auto release=[&](int seq) __attribute__((always_inline)) {
  if(seq%(2*Ratio)==2*Ratio-1 && t%32==0)s.empty[(seq/(2*Ratio))%Stages].arrive();
 };
 auto drain=[&]() __attribute__((always_inline)) {
  warpgroup_wait<0>();warpgroup_fence_operand(score);warpgroup_fence_operand(next_score);warpgroup_fence_operand(third_score);warpgroup_fence_operand(pp);warpgroup_fence_operand(pp1);warpgroup_fence_operand(pp2);
  warpgroup_fence_operand(acc[0]);warpgroup_fence_operand(acc[1]);warpgroup_fence_operand(den[0]);warpgroup_fence_operand(den[1]);
 };
 if constexpr(Safe){
  auto step=[&](int seq,auto half) __attribute__((always_inline)){
   init_score(score,seq);issue_qk(score,seq,half);warpgroup_wait<0>();warpgroup_fence_operand(score);
   exponentiate(score,half,cute::true_type{});pack(score,pp);issue_pv(seq,half,pp);drain();release(seq);
  };
  for(int seq=0;seq<2*nk;seq+=2){step(seq,_0{});step(seq+1,_1{});}
 }else{
  clear(pp2);
  init_score(score,0);issue_qk(score,0,_0{});
  init_score(next_score,1);issue_qk(next_score,1,_1{});
  init_score(third_score,2);drain();
  auto step=[&](int seq,auto half,auto which,auto seed_possible) __attribute__((always_inline)) {
   constexpr int hh=decltype(half)::value,ci=decltype(which)::value,fi=(ci+2)%3;
   auto& sc=ci==0?score:(ci==1?next_score:third_score);
   auto& fs=fi==0?score:(fi==1?next_score:third_score);
   auto& prob=ci==0?pp:(ci==1?pp1:pp2);
   auto& prev=fi==0?pp:(fi==1?pp1:pp2);
   warpgroup_wait<2>();warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);
   if(seq>=3)release(seq-3);
   warpgroup_fence_operand(prev);issue_qk(fs,seq+2,half);
   if constexpr(decltype(seed_possible)::value){
    if(seq<2)exponentiate(sc,half,cute::true_type{});else exponentiate(sc,half,cute::false_type{});
   }else exponentiate(sc,half,cute::false_type{});
   pack(sc,prob);
   issue_pv_prefenced(max(seq-1,0),Int<1-hh>{},prev);
   init_score(sc,seq+3);
  };

  #pragma unroll 1
  for(int seq=0;seq<2*nk;seq+=24){
   step(seq+0,_0{},_0{},cute::true_type{});
   step(seq+1,_1{},_1{},cute::true_type{});
   step(seq+2,_0{},_2{},cute::false_type{});
   step(seq+3,_1{},_0{},cute::false_type{});
   step(seq+4,_0{},_1{},cute::false_type{});
   step(seq+5,_1{},_2{},cute::false_type{});
   step(seq+6,_0{},_0{},cute::false_type{});
   step(seq+7,_1{},_1{},cute::false_type{});
   step(seq+8,_0{},_2{},cute::false_type{});
   step(seq+9,_1{},_0{},cute::false_type{});
   step(seq+10,_0{},_1{},cute::false_type{});
   step(seq+11,_1{},_2{},cute::false_type{});
   step(seq+12,_0{},_0{},cute::false_type{});
   step(seq+13,_1{},_1{},cute::false_type{});
   step(seq+14,_0{},_2{},cute::false_type{});
   step(seq+15,_1{},_0{},cute::false_type{});
   step(seq+16,_0{},_1{},cute::false_type{});
   step(seq+17,_1{},_2{},cute::false_type{});
   step(seq+18,_0{},_0{},cute::false_type{});
   step(seq+19,_1{},_1{},cute::false_type{});
   step(seq+20,_0{},_2{},cute::false_type{});
   step(seq+21,_1{},_0{},cute::false_type{});
   step(seq+22,_0{},_1{},cute::false_type{});
   step(seq+23,_1{},_2{},cute::false_type{});
   drain();
  }
  issue_pv(2*nk-1,_1{},pp2);drain();release(2*nk-1);
 }

 auto id=pv.get_slice(t).partition_C(make_identity_tensor(Shape<_64,_32>{}));
 bool bad=missing_seed;
 #pragma unroll
 for(int hh=0;hh<2;hh++){
  auto ar=make_tensor(acc[hh].data(),flash::convert_layout_acc_rowcol(acc[hh].layout()));
  #pragma unroll
  for(int row=0;row<2;row++){
   auto lr=make_tensor(den[hh].data(),flash::convert_layout_acc_rowcol(den[hh].layout()));float val=lr(row,0);
   if constexpr(!Safe)bad|=!(val>0x1p-84f && val<0x1p-44f);
   float inv=val>0.f?1.f/val:0.f;
   #pragma unroll
   for(int col=0;col<8;col++)ar(row,col)*=inv;
  }
  #pragma unroll
  for(int n=0;n<size(acc[hh]);n+=2){
   auto x=__floats2bfloat162_rn(acc[hh](n),acc[hh](n+1));int q=qt*M+hh*64+get<0>(id(n)),d=get<1>(id(n));
   if(i0+wg<768)*reinterpret_cast<uint32_t*>(p.out+((int64_t(i0+wg)*4+h)*768+q)*32+d)=reinterpret_cast<uint32_t const&>(x);
  }
 }
 if constexpr(!Safe){if(__any_sync(0xffffffff,bad) && t%32==0 && atomicCAS(&s.bad,0,1)==0){int index=atomicAdd(p.fix,1);p.fix[index+1]=tile;}}
}
__global__ void prepare(float const* bias,bool const* mask,int64_t stride,float inv,int L,float* out,int* valid,int* fix){
 int kt=blockIdx.x,qt=blockIdx.y,h=blockIdx.z,nk=L/LN,nq=L/M;
 for(int idx=threadIdx.x;idx<M*LN;idx+=256){
  int e=idx&3,t=(idx>>2)&127,u=(idx>>9)%(N/8),hh=(idx/(64*N))%2,cc=idx/(M*N);
  int q=qt*M+hh*64+16*(t>>5)+(t&31)/4+8*(e/2),k=kt*LN+cc*N+8*u+2*(t%4)+e%2;
  out[((int64_t(h)*nq+qt)*nk+kt)*M*LN+idx]=(!mask || mask[int64_t(k)*stride])?bias[(int64_t(h)*L+q)*L+k]*inv:-INFINITY;
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
at::Tensor sol_forward_m128p3qr24(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor bias,c10::optional<at::Tensor> mask,double scale){
 using namespace ta_sol_m128p3qr24;int L=q.size(-2);TORCH_CHECK(L==768 && float(scale)==0x1.6a09e6p-3f,"L768 standard scale required");
 TORCH_CHECK((L==384 || L==768 || L==1024) && q.numel()==int64_t(L)*4*L*32 && q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.is_contiguous(),"C128/H4/D32 contiguous square BF16 operands required");
 for(auto const& x:{k,v})TORCH_CHECK(x.device()==q.device() && x.sizes()==q.sizes() && x.scalar_type()==q.scalar_type() && x.is_contiguous(),"unsupported K/V");
 TORCH_CHECK(bias.device()==q.device() && bias.scalar_type()==at::kFloat && bias.is_contiguous() && bias.numel()==4*L*L,"contiguous float bias required");
 bool const* mp=nullptr;int64_t ms=1;
 if(mask){auto m=*mask;TORCH_CHECK(m.device()==q.device() && m.scalar_type()==at::kBool && m.dim()==5 && m.size(0)==1 && m.size(1)==L && m.size(2)==1 && m.size(3)==1 && m.size(4)==L && m.stride(1)==0,"row-broadcast mask required");mp=m.data_ptr<bool>();ms=m.stride(4);}
 c10::cuda::CUDAGuard guard(q.device());
 auto output=at::empty_like(q),prepared=at::empty_like(bias),fix=at::empty({1+(L/M)*((L+Rows-1)/Rows)*4},q.options().dtype(at::kInt)),valid=at::empty({1},fix.options());
 auto stream=at::cuda::getCurrentCUDAStream();
 prepare<<<dim3(L/LN,L/M,4),256,0,stream>>>(bias.data_ptr<float>(),mp,ms,float(1./scale),L,prepared.data_ptr<float>(),valid.data_ptr<int>(),fix.data_ptr<int>());
 auto g=[&](at::Tensor x){return make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),make_shape(L,32,L*4),ST{32,_1{},int64_t(L)*32});};
 auto tq=make_tma_copy(SM90_TMA_LOAD{},g(q),SQ{},Shape<_128,_32>{},_1{});
 auto tk=make_tma_copy(SM90_TMA_LOAD{},g(k),SK{},Shape<Int<LN>,_32>{},_1{});
 auto tv=make_tma_copy(SM90_TMA_LOAD{},g(v),SK{},Shape<Int<LN>,_32>{},_1{});
 auto tb=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(prepared.data_ptr<float>()),make_shape(256,N/4,2*L/N,L/M,4),DB{_1{},_256{},64*N,int64_t(L/N)*M*N,int64_t(L/M)*(L/N)*M*N}),SB{},Shape<_256,Int<N/4>>{},_1{});
 Params p{tq,tk,tv,tb,(Element*)output.data_ptr(),fix.data_ptr<int>(),valid.data_ptr<int>(),L,float(scale)};
 int ct=(L/M)*((L+Rows-1)/Rows)*4;
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);
 attention<true><<<ct,Threads,sizeof(Shared),stream>>>(p);
 uniform<<<dim3(L,4),128,0,stream>>>((Element const*)v.data_ptr(),(Element*)output.data_ptr(),L,valid.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
