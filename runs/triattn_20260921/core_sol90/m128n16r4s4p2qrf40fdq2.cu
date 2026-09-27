#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
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

namespace ta_sol_m128n16r4s4p2qrf40fdq2 {
using namespace cute;
using Element=cutlass::bfloat16_t;
constexpr int M=128,N=16,LN=128,Ratio=LN/N,D=32,Rows=4,Stages=2;
constexpr int H=M/64,BiasSlots=Ratio*H;
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
using PV32=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));
using PV=decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));
using QKr=decltype(make_tiled_mma(GMMA::rs_op_selector<Element,Element,float,Shape<_64,Int<N>,_32>>()));
using LS=decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>{}));
struct Params { TQ q;TK k,v;TB bias;Element* out;int* fix;int* valid;int L;float scale; };
struct Shared {
 alignas(128) Element q[Rows][M*D],k[Stages][Rows][LN*D],v[Stages][Rows][LN*D],ones[N*D];
 alignas(128) float bias[BiasSlots][64*N];
 cutlass::arch::ClusterTransactionBarrier qr,full[Stages][Rows],vfull[Stages][Rows],bf[BiasSlots/2];
 cutlass::arch::ClusterBarrier empty[Stages][Rows],kempty[Stages][Rows],be[BiasSlots];int bad;
};
__device__ __forceinline__ float ex2(float x) { float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y; }

template<bool Safe> __global__ __launch_bounds__(Threads,1)
void attention(CUTE_GRID_CONSTANT Params const p) {
 extern __shared__ __align__(128) unsigned char memory[];
 auto& s=*reinterpret_cast<Shared*>(memory);
 int tid=threadIdx.x,nq=768/M,nk=768/N*H,nr=(768+Rows-1)/Rows;
 if(!*p.valid)return;
 int tile=blockIdx.x;
 if constexpr(Safe) { if(tile>=p.fix[0])return;tile=p.fix[1+tile]; }
 int qt=tile%nq,rg=tile/nq%nr,h=tile/(nq*nr),i0=rg*Rows;
 if(tid==0) {
  s.qr.init(1);for(int st=0;st<BiasSlots;st++){if(st<BiasSlots/2)s.bf[st].init(2);s.be[st].init(Rows*4);}s.bad=0;
  for(int st=0;st<Stages;st++)for(int r=0;r<Rows;r++){s.full[st][r].init(1);s.vfull[st][r].init(1);s.empty[st][r].init(4);s.kempty[st][r].init(4);}
  cutlass::arch::fence_barrier_init();
 }
 for(int x=tid;x<N*D;x+=Threads)s.ones[x]=Element(1.f);
 cutlass::arch::fence_view_async_shared();__syncthreads();
 if(tid>=Consumers) {
  if constexpr(112>0)cutlass::arch::warpgroup_reg_dealloc<32>();
  if(tid==Consumers) {
   s.qr.arrive_and_expect_tx(Rows*M*D*sizeof(Element));
   for(int r=0;r<Rows;r++) {
    auto src=local_tile(p.q.get_tma_tensor(make_shape(768,32,768*4))(_,_,min(i0+r,768-1)*4+h),Shape<_128,_32>{},make_coord(qt,0));
    auto dst=make_tensor(make_smem_ptr(s.q[r]),SQ{});auto sl=p.q.get_slice(_0{});
    copy(p.q.with(reinterpret_cast<uint64_t&>(s.qr)),sl.partition_S(src),sl.partition_D(dst));
   }
   for(int seq=0;seq<768/LN;seq++) {
    int st=seq%Stages;
    for(int r=0;r<Rows;r++) {
     if(seq>=Stages){s.kempty[st][r].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
     s.full[st][r].arrive_and_expect_tx(LN*D*sizeof(Element));
     int ih=min(i0+r,768-1)*4+h;
     auto ks=local_tile(p.k.get_tma_tensor(make_shape(768,32,768*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{});
     auto kl=p.k.get_slice(_0{});
     copy(p.k.with(reinterpret_cast<uint64_t&>(s.full[st][r])),kl.partition_S(ks),kl.partition_D(kd));
    }

   }
  }
  if(tid==Consumers+64) {
   for(int seq=0;seq<768/LN;seq++) {
    int st=seq%Stages;
    for(int r=0;r<Rows;r++) {
     if(seq>=Stages){s.empty[st][r].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
     s.vfull[st][r].arrive_and_expect_tx(LN*D*sizeof(Element));
     int ih=min(i0+r,768-1)*4+h;
     auto vs=local_tile(p.v.get_tma_tensor(make_shape(768,32,768*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto vd=make_tensor(make_smem_ptr(s.v[st][r]),SK{});auto vl=p.v.get_slice(_0{});
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.vfull[st][r])),vl.partition_S(vs),vl.partition_D(vd));
    }
   }
  }
  if(tid==Consumers+32) {
   for(int seq=0;seq<768/N*H;seq++) {
    int st=seq%BiasSlots;
    if(seq>=BiasSlots){s.be[st].wait(((seq/BiasSlots)-1)&1);asm volatile("":::"memory");}
    s.bf[st/2].arrive_and_expect_tx(64*N*sizeof(float));
    auto bs=p.bias.get_tma_tensor(make_shape(256,N/4,768/N*H,nq,4))(_,_,seq,qt,h);
    auto bd=make_tensor(make_smem_ptr(s.bias[st]),SB{});auto bl=p.bias.get_slice(_0{});
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bf[st/2])),bl.partition_S(bs),bl.partition_D(bd));
   }
  }
  return;
 }
 if constexpr(112>0)cutlass::arch::warpgroup_reg_alloc<112>();
 int wg=int(__reduce_max_sync(0xffffffffu,unsigned(tid)/128u)),t=tid%128;
 QK qk;PV pv;LS ls;auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);auto tl=ls.get_slice(0);
 using Score=decltype(partition_fragment_C(qk,Shape<_64,Int<N>>{}));
 using Output=decltype(partition_fragment_C(pv,Shape<_64,_40>{}));
 Score scores[4];auto& score=scores[0];auto& next_score=scores[1];Output acc[H];
 #pragma unroll
 for(int hh=0;hh<H;++hh)clear(acc[hh]);
 auto pp=make_tensor_like<Element>(make_tensor(score.data(),flash::convert_layout_acc_Aregs<PV>(score.layout())));
 auto pp1=make_fragment_like(pp);
 using Den=decltype(partition_fragment_C(ls,Shape<_64,_8>{}));
 auto den0=make_tensor(acc[0].data()+16,Den{}.layout());
 decltype(den0) den[2]={den0,make_tensor(acc[H-1].data()+16,Den{}.layout())};
 auto lb=tl.partition_fragment_B(make_tensor(make_smem_ptr(s.ones),S1{}));
 float mx[2][2]={{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f},{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f}};
 bool missing_seed=false;float c=0x1.6a09e6p-3f*1.4426950408889634f;
 
  #pragma unroll
  for(int hh=0;hh<H;++hh){warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}
 s.qr.wait(0);asm volatile("":::"memory");
 QKr qkr;auto qrthr=qkr.get_thread_slice(t);
 auto qshared=make_tensor(make_smem_ptr(s.q[wg]),SQ{});
 auto qrproto=qrthr.partition_fragment_A(local_tile(qshared,Shape<_64,_32>{},make_coord(0,0)));
 decltype(qrproto) qr_all[H];
 if constexpr(!Safe){
  auto cp=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},qkr);
  auto ct=cp.get_thread_slice(t);
  #pragma unroll
  for(int hh=0;hh<H;++hh){
   auto qs=ct.partition_S(local_tile(qshared,Shape<_64,_32>{},make_coord(hh,0)));
   if(hh==0){auto qd=ct.retile_D(qr_all[hh]);cute::copy(cp,qs,qd);warpgroup_fence_operand(qr_all[hh]);}
  }
 }

 auto init_score=[&](auto& score,int seq) __attribute__((always_inline)) {
  if(seq<nk && seq%2==0){s.bf[(seq%BiasSlots)/2].wait((seq/BiasSlots)&1);asm volatile("":::"memory");}
  #pragma unroll
  for(int u=0;u<N/8;u++){
   float4 b=*reinterpret_cast<float4 const*>(s.bias[seq%BiasSlots]+u*512+t*4);
   score(4*u)=b.x;score(4*u+1)=b.y;score(4*u+2)=b.z;score(4*u+3)=b.w;
  }
 };
 auto issue_qk=[&](auto& score,int seq,auto half) __attribute__((always_inline)) {
  constexpr int hh=decltype(half)::value;
  int kt=seq/(Ratio*H),st=kt%Stages,cx=(seq/H)%Ratio;
  if(seq<nk && seq%(Ratio*H)==0){s.full[st][wg].wait((kt/Stages)&1);asm volatile("":::"memory");}
  auto qall=make_tensor(make_smem_ptr(s.q[wg]),SQ{});
  auto qs=local_tile(qall,Shape<_64,_32>{},make_coord(hh,0));auto qa=tq.partition_fragment_A(qs);
  auto kall=make_tensor(make_smem_ptr(s.k[0][wg]),SK{});
  auto ks=local_tile(kall,Shape<Int<N>,_32>{},make_coord(0,0));auto kb0=tq.partition_fragment_B(ks);
  auto kdesc=kb0.data().desc_;
  kdesc.reg32_[0]+=uint32_t((st*Rows*LN*D+cx*N*D)/8);
  auto kb=make_tensor(GMMA::DescriptorIterator{kdesc},kb0.layout());
  warpgroup_fence_operand(score);warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   if constexpr(Safe)gemm(qk,qa(_,_,kk),kb(_,_,kk),score);
   else {
    if constexpr(hh==0)gemm(qkr,qr_all[hh](_,_,kk),kb(_,_,kk),score);
    else {if(kk<0)gemm(qkr,qr_all[hh](_,_,kk),kb(_,_,kk),score);else gemm(qk,qa(_,_,kk),kb(_,_,kk),score);}
   }
  }
  warpgroup_commit_batch();
  if(seq<nk){__syncwarp();if(t%32==0)s.be[seq%BiasSlots].arrive();}
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
   if constexpr(!Safe){
    #pragma unroll
    for(int col=0;col<N/4;col+=4){
     asm volatile(
      "fma.rn.ftz.f32 %0,%0,%4,%5;\n"
      "fma.rn.ftz.f32 %1,%1,%4,%5;\n"
      "fma.rn.ftz.f32 %2,%2,%4,%5;\n"
      "fma.rn.ftz.f32 %3,%3,%4,%5;\n"
      "ex2.approx.ftz.f32 %0,%0;\n"
      "ex2.approx.ftz.f32 %1,%1;\n"
      "ex2.approx.ftz.f32 %2,%2;\n"
      "ex2.approx.ftz.f32 %3,%3;\n"
      : "+f"(sr(row,col)),"+f"(sr(row,col+1)),"+f"(sr(row,col+2)),"+f"(sr(row,col+3))
      : "f"(c),"f"(-mx[hh][row]));
    }
   }else{
   #pragma unroll
   for(int col=0;col<N/4;col++){
    float x=sr(row,col);
    if constexpr(Safe)sr(row,col)=x==-INFINITY?0.f:ex2(fmaf(x,c,-mx[hh][row]));
    else sr(row,col)=ex2(fmaf(x,c,-mx[hh][row]));
   }
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
  if(seq%(Ratio*H)==0){s.vfull[(seq/(Ratio*H))%Stages][wg].wait(((seq/(Ratio*H))/Stages)&1);asm volatile("":::"memory");}
  auto vall=make_tensor(make_smem_ptr(s.v[0][wg]),SV{});
  auto vs=local_tile(vall,Shape<_32,Int<N>>{},make_coord(0,0));PV32 pv32;auto raw=pv32.get_slice(0).partition_fragment_B(vs);
  auto desc=raw.data().desc_;
  uint32_t base=cute::cast_smem_ptr_to_uint(s.v[0][wg]);
  desc.bitfield.leading_byte_offset_=(cute::cast_smem_ptr_to_uint(s.ones)-base)>>4;
  uint32_t delta=uint32_t((((seq/(Ratio*H))%Stages)*Rows*LN*D+((seq/H)%Ratio)*N*D)/8);
  desc.reg32_[0]+=delta-(delta<<16);
  auto vb=make_tensor(GMMA::DescriptorIterator{desc},raw.layout());
  warpgroup_fence_operand(pp);if constexpr(Safe){warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}
  warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<N/16;kk++)gemm(pv,pp(_,_,kk),vb(_,_,kk),acc[hh]);

  warpgroup_commit_batch();
 };
 auto issue_pv_prefenced=[&](int seq,auto half,auto& pp) __attribute__((always_inline)) {
  constexpr int hh=decltype(half)::value;
  if(seq%(Ratio*H)==0){s.vfull[(seq/(Ratio*H))%Stages][wg].wait(((seq/(Ratio*H))/Stages)&1);asm volatile("":::"memory");}
  auto vall=make_tensor(make_smem_ptr(s.v[0][wg]),SV{});
  auto vs=local_tile(vall,Shape<_32,Int<N>>{},make_coord(0,0));PV32 pv32;auto raw=pv32.get_slice(0).partition_fragment_B(vs);
  auto desc=raw.data().desc_;
  uint32_t base=cute::cast_smem_ptr_to_uint(s.v[0][wg]);
  desc.bitfield.leading_byte_offset_=(cute::cast_smem_ptr_to_uint(s.ones)-base)>>4;
  uint32_t delta=uint32_t((((seq/(Ratio*H))%Stages)*Rows*LN*D+((seq/H)%Ratio)*N*D)/8);
  desc.reg32_[0]+=delta-(delta<<16);
  auto vb=make_tensor(GMMA::DescriptorIterator{desc},raw.layout());
  if constexpr(Safe){warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}
  #pragma unroll
  for(int kk=0;kk<N/16;kk++)gemm(pv,pp(_,_,kk),vb(_,_,kk),acc[hh]);

  warpgroup_commit_batch();
 };
 auto release_k=[&](int seq) __attribute__((always_inline)) {
  if(seq%(Ratio*H)==Ratio*H-1 && t%32==0)s.kempty[(seq/(Ratio*H))%Stages][wg].arrive();
 };
 auto release=[&](int seq) __attribute__((always_inline)) {
  if(seq%(Ratio*H)==Ratio*H-1 && t%32==0)s.empty[(seq/(Ratio*H))%Stages][wg].arrive();
 };
 auto drain=[&]() __attribute__((always_inline)) {
  warpgroup_wait<0>();asm volatile("":::"memory");
  if constexpr(Safe){warpgroup_fence_operand(score);warpgroup_fence_operand(pp);}
  else {
   #pragma unroll
   for(int j=0;j<4;++j)warpgroup_fence_operand(scores[j]);
   warpgroup_fence_operand(pp);warpgroup_fence_operand(pp1);
  }
  #pragma unroll
  for(int hh=0;hh<H;++hh){warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}
 };
 if constexpr(Safe){
  auto step=[&](int seq,auto half) __attribute__((always_inline)){
   init_score(score,seq);issue_qk(score,seq,half);warpgroup_wait<0>();asm volatile("":::"memory");warpgroup_fence_operand(score);
   release_k(seq);exponentiate(score,half,cute::true_type{});pack(score,pp);issue_pv(seq,half,pp);drain();release(seq);
  };
  for(int seq=0;seq<nk;seq+=H){step(seq,_0{});if constexpr(H==2)step(seq+1,_1{});}
 }else{
  // Four scores, two P fragments; QK(k+2), E(k), PV(k-1), bias(k+3).
  // The first four QKs are drained so each query half can seed from 32 keys.
  clear(pp);clear(pp1);
  init_score(scores[0],0);issue_qk(scores[0],0,Int<0%H>{});
  init_score(scores[1],1);issue_qk(scores[1],1,Int<1%H>{});
  init_score(scores[2],2);issue_qk(scores[2],2,Int<2%H>{});
  init_score(scores[3],3);issue_qk(scores[3],3,Int<3%H>{});
  drain();
  auto seed_pair=[&](auto& a,auto& b,auto half) __attribute__((always_inline)) {
    constexpr int hh=decltype(half)::value;
    auto ar=make_tensor(a.data(),flash::convert_layout_acc_rowcol(a.layout()));
    auto br=make_tensor(b.data(),flash::convert_layout_acc_rowcol(b.layout()));
    #pragma unroll
    for(int row=0;row<2;++row) {
      float m=ar(row,0);
      #pragma unroll
      for(int col=1;col<N/4;++col)m=fmaxf(m,ar(row,col));
      #pragma unroll
      for(int col=0;col<N/4;++col)m=fmaxf(m,br(row,col));
      m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,1));
      m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,2));
      if(m!=-INFINITY)mx[hh][row]=m*c+64.f;else missing_seed=true;
    }
  };
  if constexpr(H==1)seed_pair(scores[0],scores[1],_0{});
  else {seed_pair(scores[0],scores[2],_0{});seed_pair(scores[1],scores[3],_1{});}
  exponentiate(scores[0],Int<0%H>{},cute::false_type{});
  pack(scores[0],pp);issue_pv(0,Int<0%H>{},pp);
  exponentiate(scores[1],Int<1%H>{},cute::false_type{});
  init_score(scores[0],4);
  auto step=[&](auto seqc) __attribute__((always_inline)) {
    constexpr int seq=decltype(seqc)::value;
    constexpr int cur=seq%4,future=(seq+2)%4,previous=(seq+3)%4;
    auto& prob=(seq-1)%2==0?pp:pp1;
    // No phantom tail QK: the final body has only three prior groups.
    // wait1 there retires PV(seq-3) before its P slot is reused.
    if constexpr(seq==768/N*H-1)warpgroup_wait<1>();
    else warpgroup_wait<2>();
    asm volatile("":::"memory");
    warpgroup_fence_operand(scores[cur]);warpgroup_fence_operand(prob);
    release_k(seq);if constexpr(seq>=3)release(seq-3);
    pack(scores[previous],prob);warpgroup_fence_operand(prob);
    if constexpr(seq+2<768/N*H)issue_qk(scores[future],seq+2,Int<(seq+2)%H>{});
    exponentiate(scores[cur],Int<seq%H>{},cute::false_type{});
    // Explicit fence is retained: compiler placement, not source order,
    // determines whether the packed P was written before QK's fence.
    issue_pv(seq-1,Int<(seq-1)%H>{},prob);
    if constexpr(seq+3<768/N*H)init_score(scores[previous],seq+3);
  };
  step(Int<2>{});
  step(Int<3>{});
  step(Int<4>{});
  step(Int<5>{});
  step(Int<6>{});
  step(Int<7>{});
  step(Int<8>{});
  step(Int<9>{});
  step(Int<10>{});
  step(Int<11>{});
  step(Int<12>{});
  step(Int<13>{});
  step(Int<14>{});
  step(Int<15>{});
  step(Int<16>{});
  step(Int<17>{});
  step(Int<18>{});
  step(Int<19>{});
  step(Int<20>{});
  step(Int<21>{});
  step(Int<22>{});
  step(Int<23>{});
  step(Int<24>{});
  step(Int<25>{});
  step(Int<26>{});
  step(Int<27>{});
  step(Int<28>{});
  step(Int<29>{});
  step(Int<30>{});
  step(Int<31>{});
  step(Int<32>{});
  step(Int<33>{});
  step(Int<34>{});
  step(Int<35>{});
  step(Int<36>{});
  step(Int<37>{});
  step(Int<38>{});
  step(Int<39>{});
  step(Int<40>{});
  step(Int<41>{});
  step(Int<42>{});
  step(Int<43>{});
  step(Int<44>{});
  step(Int<45>{});
  step(Int<46>{});
  step(Int<47>{});
  step(Int<48>{});
  step(Int<49>{});
  step(Int<50>{});
  step(Int<51>{});
  step(Int<52>{});
  step(Int<53>{});
  step(Int<54>{});
  step(Int<55>{});
  step(Int<56>{});
  step(Int<57>{});
  step(Int<58>{});
  step(Int<59>{});
  step(Int<60>{});
  step(Int<61>{});
  step(Int<62>{});
  step(Int<63>{});
  step(Int<64>{});
  step(Int<65>{});
  step(Int<66>{});
  step(Int<67>{});
  step(Int<68>{});
  step(Int<69>{});
  step(Int<70>{});
  step(Int<71>{});
  step(Int<72>{});
  step(Int<73>{});
  step(Int<74>{});
  step(Int<75>{});
  step(Int<76>{});
  step(Int<77>{});
  step(Int<78>{});
  step(Int<79>{});
  step(Int<80>{});
  step(Int<81>{});
  step(Int<82>{});
  step(Int<83>{});
  step(Int<84>{});
  step(Int<85>{});
  step(Int<86>{});
  step(Int<87>{});
  step(Int<88>{});
  step(Int<89>{});
  step(Int<90>{});
  step(Int<91>{});
  step(Int<92>{});
  step(Int<93>{});
  step(Int<94>{});
  step(Int<95>{});
  drain();
  pack(scores[3],pp1);issue_pv(768/N*H-1,Int<(768/N*H-1)%H>{},pp1);
  drain();release(768/N*H-1);
 }

 auto id=pv.get_slice(t).partition_C(make_identity_tensor(Shape<_64,_40>{}));
 bool bad=missing_seed;
 if constexpr(!Safe){
  #pragma unroll
  for(int hh=0;hh<H;++hh){
  #pragma unroll
  for(int j=0;j<16;++j)bad|=(__float_as_uint(acc[hh](j))&0x7f800000u)==0x7f800000u;
  }
 }
 #pragma unroll
 for(int hh=0;hh<H;hh++){
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
  for(int n=0;n<16;n+=2){
   auto x=__floats2bfloat162_rn(acc[hh](n),acc[hh](n+1));int q=qt*M+hh*64+get<0>(id(n)),d=get<1>(id(n));
   if(i0+wg<768)*reinterpret_cast<uint32_t*>(p.out+((int64_t(i0+wg)*4+h)*768+q)*32+d)=reinterpret_cast<uint32_t const&>(x);
  }
 }
 if constexpr(!Safe){if(__any_sync(0xffffffff,bad) && t%32==0 && atomicCAS(&s.bad,0,1)==0){int index=atomicAdd(p.fix,1);p.fix[index+1]=tile;}}
}
__global__ void prepare(float const* bias,bool const* mask,int64_t stride,float inv,int L,float* out,int* valid,int* fix){
 int kt=blockIdx.x,qt=blockIdx.y,h=blockIdx.z,nk=L/LN,nq=L/M;
 for(int idx=threadIdx.x;idx<M*LN;idx+=256){
  int e=idx&3,t=(idx>>2)&127,u=(idx>>9)%(N/8),hh=(idx/(64*N))%H,cc=idx/(M*N);
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
at::Tensor sol_forward_m128n16r4s4p2qrf40fdq2(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor bias,c10::optional<at::Tensor> mask,double scale){
 using namespace ta_sol_m128n16r4s4p2qrf40fdq2;int L=q.size(-2);TORCH_CHECK(L==768 && float(scale)==0x1.6a09e6p-3f,"L768 standard scale required");
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
 auto tb=make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr(prepared.data_ptr<float>()),make_shape(256,N/4,L/N*H,L/M,4),DB{_1{},_256{},64*N,int64_t(L/N)*M*N,int64_t(L/M)*(L/N)*M*N}),SB{},Shape<_256,Int<N/4>>{},_1{});
 Params p{tq,tk,tv,tb,(Element*)output.data_ptr(),fix.data_ptr<int>(),valid.data_ptr<int>(),L,float(scale)};
 int ct=(L/M)*((L+Rows-1)/Rows)*4;
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 int resident=0;
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,attention<false>,Threads,sizeof(Shared)));
 TORCH_CHECK(resident==1,"one-CTA occupancy required, got ",resident," smem=",sizeof(Shared));
 attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);
 attention<true><<<ct,Threads,sizeof(Shared),stream>>>(p);
 uniform<<<dim3(L,4),128,0,stream>>>((Element const*)v.data_ptr(),(Element*)output.data_ptr(),L,valid.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
