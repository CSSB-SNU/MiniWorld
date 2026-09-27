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

// L768 persistent CTA: one K/V fill, six query tiles, two consumer warpgroups.
namespace ta_sol_persistkv2cp {
using namespace cute;
using Element=cutlass::bfloat16_t;
constexpr int M=128,N=32,LN=128,Ratio=4,D=32,Rows=2,Stages=6,L=768;
constexpr int Consumers=256,Threads=384;
using SQ=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_128,_32>{}));
using SK=SQ;
using SV=decltype(composition(SK{},make_ordered_layout(Shape<_32,_128>{},Step<_2,_1>{})));
using SB=Layout<Shape<_256,_8>,Stride<_1,_256>>;
using GB=Shape<int,int,int,int,int>;
using DB=Stride<_1,_256,int64_t,int64_t,int64_t>;
using S1=decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<Element>{},Shape<_8,_32>{}));
using G3=Shape<int,int,int>;
using ST=Stride<int64_t,_1,int64_t>;
using TQ=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),G3{},ST{}),SQ{},Shape<_128,_32>{},_1{}));
using TK=TQ;
using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((float const*)nullptr),GB{},DB{}),SB{},Shape<_256,_8>{},_1{}));
using QK=decltype(make_tiled_mma(GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_32>>()));
using PV=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}));
using LS=decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>{}));
struct Params { TQ q;TK k,v;TB bias;Element* out;int* fix;int* valid;int L;float scale;float const* rawbias; };
struct Shared {
 alignas(128) Element q[Rows][M*D],k[Stages][Rows][LN*D],v[Stages][Rows][LN*D],ones[8*N];
 alignas(128) float bias[2][64*N];
 cutlass::arch::ClusterTransactionBarrier qr,full[Stages],bf[2];
 cutlass::arch::ClusterBarrier qe,be[2];
};
static_assert(sizeof(Shared)<=232448,"persistent K/V exceeds H100 shared memory");
__device__ __forceinline__ float ex2(float x) { float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y; }

template<bool Safe> __global__ __launch_bounds__(Threads,1)
void attention(CUTE_GRID_CONSTANT Params const p) {
 if(!*p.valid)return;
 int tile=blockIdx.x;
 if constexpr(Safe){if(!p.fix[tile])return;}
 extern __shared__ __align__(128) unsigned char memory[];
 auto& s=*reinterpret_cast<Shared*>(memory);
 int tid=threadIdx.x,rg=tile%(L/Rows),h=tile/(L/Rows),i0=rg*Rows;
 if(tid==0){
  s.qr.init(1);s.qe.init(Rows*4);
  for(int st=0;st<Stages;++st)s.full[st].init(2);
  for(int st=0;st<2;++st){s.bf[st].init(1);s.be[st].init(Rows*4);}
  cutlass::arch::fence_barrier_init();
 }
 for(int x=tid;x<8*N;x+=Threads)s.ones[x]=Element(1.f);
 cutlass::arch::fence_view_async_shared();__syncthreads();
 if(tid>=Consumers){
  cutlass::arch::warpgroup_reg_dealloc<32>();
  auto issue_q=[&](int qt) __attribute__((always_inline)) {
   if(qt){s.qe.wait((qt-1)&1);asm volatile("":::"memory");}
   s.qr.arrive_and_expect_tx(Rows*M*D*sizeof(Element));
   #pragma unroll
   for(int r=0;r<Rows;++r){
    auto src=local_tile(p.q.get_tma_tensor(make_shape(L,32,L*4))(_,_,(i0+r)*4+h),Shape<_128,_32>{},make_coord(qt,0));
    auto dst=make_tensor(make_smem_ptr(s.q[r]),SQ{});auto sl=p.q.get_slice(_0{});
    copy(p.q.with(reinterpret_cast<uint64_t&>(s.qr)),sl.partition_S(src),sl.partition_D(dst));
   }
  };
  if(tid==Consumers+32){
   issue_q(0);
   #pragma unroll 1
   for(int kt=0;kt<Stages;++kt){
    s.full[kt].arrive_and_expect_tx(Rows*LN*D*sizeof(Element));
    #pragma unroll
    for(int r=0;r<Rows;++r){
     auto src=local_tile(p.k.get_tma_tensor(make_shape(L,32,L*4))(_,_,(i0+r)*4+h),Shape<_128,_32>{},make_coord(kt,0));
     auto dst=make_tensor(make_smem_ptr(s.k[kt][r]),SK{});auto sl=p.k.get_slice(_0{});
     copy(p.k.with(reinterpret_cast<uint64_t&>(s.full[kt])),sl.partition_S(src),sl.partition_D(dst));
    }
   }
   #pragma unroll 1
   for(int qt=1;qt<6;++qt)issue_q(qt);
  }
  if(tid==Consumers+64){
   #pragma unroll 1
   for(int kt=0;kt<Stages;++kt){
    s.full[kt].arrive_and_expect_tx(Rows*LN*D*sizeof(Element));
    #pragma unroll
    for(int r=0;r<Rows;++r){
     auto src=local_tile(p.v.get_tma_tensor(make_shape(L,32,L*4))(_,_,(i0+r)*4+h),Shape<_128,_32>{},make_coord(kt,0));
     auto dst=make_tensor(make_smem_ptr(s.v[kt][r]),SK{});auto sl=p.v.get_slice(_0{});
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[kt])),sl.partition_S(src),sl.partition_D(dst));
    }
   }
  }
  if(tid/32==Consumers/32){
   int lane=tid%32;
   #pragma unroll 1
   for(int g=0;g<6*48;++g){
    int st=g&1;
    if(g>=2){if(lane==0)s.be[st].wait(((g/2)-1)&1);__syncwarp();asm volatile("":::"memory");}
    float const* src=p.rawbias+(int64_t(h)*6*48+g)*2048;
    float* dst=s.bias[st];
    #pragma unroll
    for(int u=0;u<16;++u){
     uint32_t sd=cast_smem_ptr_to_uint(dst+(u*32+lane)*4);
     float const* gs=src+(u*32+lane)*4;
     asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(sd),"l"(gs):"memory");
    }
    asm volatile("cp.async.commit_group;":::"memory");
    asm volatile("cp.async.wait_group 0;":::"memory");
    __syncwarp();if(lane==0)s.bf[st].arrive();
   }
  }
  return;
 }
 cutlass::arch::warpgroup_reg_alloc<224>();
 int wg=int(__reduce_max_sync(0xffffffffu,unsigned(tid)/128u)),t=tid%128;
 QK qk;PV pv;LS ls;
 qk.accumulate_=GMMA::ScaleOut::One;pv.accumulate_=GMMA::ScaleOut::One;ls.accumulate_=GMMA::ScaleOut::One;
 auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);auto tl=ls.get_slice(0);
 auto lb=tl.partition_fragment_B(make_tensor(make_smem_ptr(s.ones),S1{}));
 using Score=decltype(partition_fragment_C(qk,Shape<_64,_32>{}));
 using Output=decltype(partition_fragment_C(pv,Shape<_64,_32>{}));
 using Den=decltype(partition_fragment_C(ls,Shape<_64,_8>{}));
 auto qshared=make_tensor(make_smem_ptr(s.q[wg]),SQ{});
 auto qa0=qk.get_thread_slice(t).partition_fragment_A(local_tile(qshared,Shape<_64,_32>{},make_coord(0,0)));
 auto qa1=make_fragment_like(qa0);
 auto qcopy=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},qk);
 auto qthr=qcopy.get_thread_slice(t);
 #pragma unroll 1
 for(int qt=0;qt<6;++qt){
  Score sc[4];Output acc[2];Den den[2];
  auto pproto=make_tensor_like<Element>(make_tensor(sc[0].data(),flash::convert_layout_acc_Aregs<PV>(sc[0].layout())));
  decltype(pproto) pp[2];
  clear(acc[0]);clear(acc[1]);clear(den[0]);clear(den[1]);clear(pp[0]);clear(pp[1]);
  float nm[2][2]={};bool bad=false;float c=p.scale*1.4426950408889634f;
  s.qr.wait(qt&1);asm volatile("":::"memory");
  #pragma unroll
  for(int hh=0;hh<2;++hh){
   auto src=qthr.partition_S(local_tile(qshared,Shape<_64,_32>{},make_coord(hh,0)));
   auto dst=qthr.retile_D(hh==0?qa0:qa1);copy(qcopy,src,dst);
  }
  warpgroup_fence_operand(qa0);warpgroup_fence_operand(qa1);
  warpgroup_fence_operand(acc[0]);warpgroup_fence_operand(acc[1]);warpgroup_fence_operand(den[0]);warpgroup_fence_operand(den[1]);
  auto init=[&](auto& a,int seq) __attribute__((always_inline)) {
   if(seq<48){
    s.bf[seq&1].wait((seq/2)&1);asm volatile("":::"memory");
    #pragma unroll
    for(int u=0;u<4;++u){
     float4 b=*reinterpret_cast<float4 const*>(s.bias[seq&1]+u*512+t*4);
     a(4*u)=b.x;a(4*u+1)=b.y;a(4*u+2)=b.z;a(4*u+3)=b.w;
    }
   }else clear(a); // Do not consume the next query's prefetched slots.
  };
  auto issue_qk=[&](auto& a,int seq,auto hc) __attribute__((always_inline)) {
   constexpr int hh=decltype(hc)::value;
   int kt=seq/8;
   if(seq<48 && seq%8==0){s.full[kt].wait(0);asm volatile("":::"memory");}
   auto ks=local_tile(make_tensor(make_smem_ptr(s.k[kt<6?kt:0][wg]),SK{}),Shape<_32,_32>{},make_coord((seq/2)%4,0));
   auto kb=tq.partition_fragment_B(ks);auto& qa=hh==0?qa0:qa1;
   warpgroup_fence_operand(a);warpgroup_arrive();
   #pragma unroll
   for(int kk=0;kk<2;++kk)gemm(qk,qa(_,_,kk),kb(_,_,kk),a);
   warpgroup_commit_batch();
   if(seq<48){__syncwarp();if(t%32==0)s.be[seq&1].arrive();}
   if(seq==1){__syncwarp();if(t%32==0)s.qe.arrive();}
  };
  auto rowmax=[&](auto& a,int row) __attribute__((always_inline)) {
   auto sr=make_tensor(a.data(),flash::convert_layout_acc_rowcol(a.layout()));float m=sr(row,0);
   #pragma unroll
   for(int col=1;col<8;++col)m=fmaxf(m,sr(row,col));
   m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,1));return fmaxf(m,__shfl_xor_sync(0xffffffffu,m,2));
  };
  auto exponentiate=[&](auto& a,auto hc,auto seed) __attribute__((always_inline)) {
   constexpr int hh=decltype(hc)::value;
   auto sr=make_tensor(a.data(),flash::convert_layout_acc_rowcol(a.layout()));
   auto ar=make_tensor(acc[hh].data(),flash::convert_layout_acc_rowcol(acc[hh].layout()));
   auto lr=make_tensor(den[hh].data(),flash::convert_layout_acc_rowcol(den[hh].layout()));
   #pragma unroll
   for(int row=0;row<2;++row){
    if constexpr(Safe){
     float m=rowmax(a,row),cand=-(m*c);
     if(cand<nm[hh][row]){
      float f=ex2(cand-nm[hh][row]);
      #pragma unroll
      for(int col=0;col<8;++col)ar(row,col)*=f;
      lr(row,0)*=f;lr(row,1)*=f;nm[hh][row]=cand;
     }
    }else if constexpr(decltype(seed)::value){
     float m=rowmax(a,row);
     if(m>-INFINITY)nm[hh][row]=-(m*c)-64.f;else bad=true;
    }
    #pragma unroll
    for(int col=0;col<8;++col)sr(row,col)=ex2(fmaf(sr(row,col),c,nm[hh][row]));
   }
   warpgroup_fence_operand(a);
  };
  auto pack=[&](auto& a,auto& prob) __attribute__((always_inline)) {
   auto dst=recast<uint32_t>(prob);
   #pragma unroll
   for(int n=0;n<8;++n){auto x=__floats2bfloat162_rn(a(2*n),a(2*n+1));dst(n)=reinterpret_cast<uint32_t const&>(x);}
  };
  auto issue_pv=[&](auto& prob,int seq,auto hc,auto prefenced) __attribute__((always_inline)) {
   constexpr int hh=decltype(hc)::value;
   auto vs=local_tile(make_tensor(make_smem_ptr(s.v[seq/8][wg]),SV{}),Shape<_32,_32>{},make_coord(0,(seq/2)%4));
   auto vb=tp.partition_fragment_B(vs);
   if constexpr(!decltype(prefenced)::value){warpgroup_fence_operand(prob);warpgroup_arrive();}
   #pragma unroll
   for(int kk=0;kk<2;++kk)gemm(pv,prob(_,_,kk),vb(_,_,kk),acc[hh]);
   #pragma unroll
   for(int kk=0;kk<2;++kk)gemm(ls,prob(_,_,kk),lb(_,_,kk),den[hh]);
   warpgroup_commit_batch();
  };
  auto drain=[&]() __attribute__((always_inline)) {
   warpgroup_wait<0>();
   #pragma unroll
   for(int n=0;n<4;++n)warpgroup_fence_operand(sc[n]);
   #pragma unroll
   for(int hh=0;hh<2;++hh){warpgroup_fence_operand(pp[hh]);warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);}
  };
  auto rescale=[&](auto& pend) __attribute__((always_inline)) {
   constexpr uint32_t lo=uint32_t(127-84)<<23,width=(uint32_t(127-44)<<23)-lo;
   #pragma unroll
   for(int hh=0;hh<2;++hh){
    auto ar=make_tensor(acc[hh].data(),flash::convert_layout_acc_rowcol(acc[hh].layout()));
    auto lr=make_tensor(den[hh].data(),flash::convert_layout_acc_rowcol(den[hh].layout()));
    auto pr=make_tensor(pend.data(),flash::convert_layout_acc_rowcol(pend.layout()));
    bool out=false;
    #pragma unroll
    for(int row=0;row<2;++row)out|=(__float_as_uint(lr(row,0))-lo)>width;
    if(__any_sync(0xffffffffu,out)){
     #pragma unroll
     for(int row=0;row<2;++row){
      uint32_t bits=__float_as_uint(lr(row,0));int k=int((bits>>23)&255)-127+64;
      k=(bits-lo)>width?min(max(k,-126),126):0;float f=__uint_as_float(uint32_t(127-k)<<23);nm[hh][row]-=float(k);
      #pragma unroll
      for(int col=0;col<8;++col)ar(row,col)*=f;
      lr(row,0)*=f;lr(row,1)*=f;
      if(hh==1){
       #pragma unroll
       for(int col=0;col<8;++col)pr(row,col)*=f;
      }
     }
    }
    warpgroup_fence_operand(acc[hh]);warpgroup_fence_operand(den[hh]);
   }
  };
  if constexpr(Safe){
   auto step=[&](int seq,auto hc) __attribute__((always_inline)) {
    init(sc[0],seq);issue_qk(sc[0],seq,hc);drain();
    exponentiate(sc[0],hc,cute::true_type{});pack(sc[0],pp[0]);issue_pv(pp[0],seq,hc,cute::false_type{});drain();
   };
   #pragma unroll 1
   for(int seq=0;seq<48;seq+=2){step(seq,_0{});step(seq+1,_1{});}
  }else{
   init(sc[0],0);init(sc[1],1);issue_qk(sc[0],0,_0{});issue_qk(sc[1],1,_1{});drain();
   init(sc[2],2);init(sc[3],3);issue_qk(sc[2],2,_0{});issue_qk(sc[3],3,_1{});
   exponentiate(sc[0],_0{},cute::true_type{});pack(sc[0],pp[0]);issue_pv(pp[0],0,_0{},cute::false_type{});
   exponentiate(sc[1],_1{},cute::true_type{});init(sc[0],4);drain();
   auto step=[&](int period,auto dc) __attribute__((always_inline)) {
    constexpr int dd=decltype(dc)::value,e=dd+2,hh=e&1,ci=e&3,fi=(e+2)&3,pi=(e-1)&3,ph=(e-1)&1;
    int seq=period*16+e;
    if constexpr(dd!=0)warpgroup_wait<2>();
    warpgroup_fence_operand(sc[ci]);warpgroup_fence_operand(pp[ph]);
    pack(sc[pi],pp[ph]);warpgroup_fence_operand(pp[ph]);
    issue_qk(sc[fi],seq+2,Int<hh>{});
    exponentiate(sc[ci],Int<hh>{},cute::false_type{});
    issue_pv(pp[ph],seq-1,Int<ph>{},cute::true_type{});
    init(sc[pi],seq+3);
   };
   #pragma unroll 1
   for(int period=0;period<3;++period){
    step(period,_0{});step(period,_1{});step(period,_2{});step(period,_3{});
    step(period,_4{});step(period,_5{});step(period,_6{});step(period,_7{});
    step(period,_8{});step(period,Int<9>{});step(period,Int<10>{});step(period,Int<11>{});
    step(period,Int<12>{});step(period,Int<13>{});
    if(period<2){step(period,Int<14>{});step(period,Int<15>{});drain();rescale(sc[1]);}
    else{drain();rescale(sc[3]);}
   }
   pack(sc[3],pp[1]);issue_pv(pp[1],47,_1{},cute::false_type{});drain();
  }
  auto id=pv.get_slice(t).partition_C(make_identity_tensor(Shape<_64,_32>{}));
  #pragma unroll
  for(int hh=0;hh<2;++hh){
   auto ar=make_tensor(acc[hh].data(),flash::convert_layout_acc_rowcol(acc[hh].layout()));
   auto lr=make_tensor(den[hh].data(),flash::convert_layout_acc_rowcol(den[hh].layout()));
   #pragma unroll
   for(int row=0;row<2;++row){
    float val=lr(row,0),chk=0.f;
    #pragma unroll
    for(int col=0;col<8;++col)chk+=ar(row,col);
    chk=val+chk*0.f;if constexpr(!Safe)bad|=!(chk>0.f && chk<INFINITY);
    float inv=val>0.f?1.f/val:0.f;
    #pragma unroll
    for(int col=0;col<8;++col)ar(row,col)*=inv;
   }
   #pragma unroll
   for(int n=0;n<size(acc[hh]);n+=2){
    auto x=__floats2bfloat162_rn(acc[hh](n),acc[hh](n+1));int qq=qt*M+hh*64+get<0>(id(n)),d=get<1>(id(n));
    *reinterpret_cast<uint32_t*>(p.out+((int64_t(i0+wg)*4+h)*L+qq)*32+d)=reinterpret_cast<uint32_t const&>(x);
   }
  }
  if constexpr(!Safe){if(__any_sync(0xffffffffu,bad) && t%32==0)atomicExch(p.fix+tile,1);}
  warpgroup_fence_operand(qa0);warpgroup_fence_operand(qa1);
 }
}
__global__ void prepare(float const* bias,bool const* mask,int64_t stride,float inv,int L,float* out,int* valid,int* fix){
 int kt=blockIdx.x,qt=blockIdx.y,h=blockIdx.z,nk=L/LN,nq=L/M;
 for(int idx=threadIdx.x;idx<M*LN;idx+=256){
  int e=idx&3,t=(idx>>2)&127,u=(idx>>9)&3,hh=(idx/2048)%2,cc=idx/4096;
  int q=qt*M+hh*64+16*(t>>5)+(t&31)/4+8*(e/2),k=kt*LN+cc*32+8*u+2*(t%4)+e%2;
  out[((int64_t(h)*nq+qt)*nk+kt)*M*LN+idx]=(!mask || mask[int64_t(k)*stride])?bias[(int64_t(h)*L+q)*L+k]*inv:-INFINITY;
 }
 if(kt==0 && qt==0 && h==0){
  bool any=!mask;for(int k=threadIdx.x;k<L;k+=256)any|=mask && mask[int64_t(k)*stride];
  int yes=__syncthreads_or(any);if(threadIdx.x==0)*valid=yes;for(int x=threadIdx.x;x<(L/Rows)*4;x+=256)fix[x]=0;
 }
}
__global__ void uniform(Element const* v,Element* out,int L,int const* valid){
 if(*valid)return;int i=blockIdx.x,h=blockIdx.y,d=threadIdx.x%32,g=threadIdx.x/32;__shared__ float part[4][32];
 float sum=0;for(int k=g;k<L;k+=4)sum+=float(v[((int64_t(i)*4+h)*L+k)*32+d]);part[g][d]=sum;__syncthreads();
 Element mean=Element((part[0][d]+part[1][d]+part[2][d]+part[3][d])/float(L));
 for(int q=g;q<L;q+=4)out[((int64_t(i)*4+h)*L+q)*32+d]=mean;
}
}
at::Tensor sol_forward_persistkv2cp(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor bias,c10::optional<at::Tensor> mask,double scale){
 using namespace ta_sol_persistkv2cp;int L=q.size(-2);TORCH_CHECK(L==768,"persistent K/V prototype requires L768");
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
 Params p{tq,tk,tv,tb,(Element*)output.data_ptr(),fix.data_ptr<int>(),valid.data_ptr<int>(),L,float(scale),prepared.data_ptr<float>()};
 int ct=((L+Rows-1)/Rows)*4;
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 C10_CUDA_CHECK(cudaFuncSetAttribute(attention<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
 attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);
 attention<true><<<ct,Threads,sizeof(Shared),stream>>>(p);
 uniform<<<dim3(L,4),128,0,stream>>>((Element const*)v.data_ptr(),(Element*)output.data_ptr(),L,valid.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
