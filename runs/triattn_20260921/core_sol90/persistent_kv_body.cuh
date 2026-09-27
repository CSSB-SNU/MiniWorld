// L768 persistent CTA: one K/V fill, six query tiles, two consumer warpgroups.
namespace SOL_NAMESPACE {
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
struct Params { TQ q;TK k,v;TB bias;Element* out;int* fix;int* valid;int L;float scale; };
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
  if(tid==Consumers){
   #pragma unroll 1
   for(int g=0;g<6*48;++g){
    int st=g&1;
    if(g>=2){s.be[st].wait(((g/2)-1)&1);asm volatile("":::"memory");}
    s.bf[st].arrive_and_expect_tx(64*N*sizeof(float));
    auto src=p.bias.get_tma_tensor(make_shape(256,8,48,6,4))(_,_,g%48,g/48,h);
    auto dst=make_tensor(make_smem_ptr(s.bias[st]),SB{});auto sl=p.bias.get_slice(_0{});
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bf[st])),sl.partition_S(src),sl.partition_D(dst));
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
