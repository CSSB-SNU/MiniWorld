// An independent MMA warpgroup feeds four softmax warpgroups through two
// shared score/probability slots per pair row. The TMA producer is separate.
namespace SOL_NAMESPACE { namespace specialized {
constexpr int Threads=768,Producer=640,Stages=2,Slots=2;
static_assert(Rows==4 && N==32 && LN==64);
using SP=SQ;
using SPV=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>{}));
using SLS=decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::K>{}));
struct Shared {
 alignas(128) Element q[Rows][M*D],k[Stages][Rows][LN*D],v[Stages][Rows][LN*D],ones[8*N];
 alignas(128) float bias[Stages][M*LN],score[Rows][Slots][M*N];
 alignas(128) Element prob[Rows][Slots][M*N];
 cutlass::arch::ClusterTransactionBarrier qr,full[Stages];
 cutlass::arch::ClusterBarrier empty[Stages],score_ready[Rows][Slots],prob_ready[Rows][Slots];
 int bad;
};
__global__ __launch_bounds__(Threads,1) void attention(CUTE_GRID_CONSTANT Params const p) {
 extern __shared__ __align__(128) unsigned char memory[];
 auto& s=*reinterpret_cast<Shared*>(memory);
 int tid=threadIdx.x,nq=p.L/M,nk=p.L/N,nr=(p.L+Rows-1)/Rows;
 if(!*p.valid)return;
 int tile=blockIdx.x,qt=tile%nq,rg=tile/nq%nr,h=tile/(nq*nr),i0=rg*Rows;
 if(tid==0) {
  s.qr.init(1);s.bad=0;
  for(int st=0;st<Stages;st++){s.full[st].init(1);s.empty[st].init(4);}
  for(int r=0;r<Rows;r++)for(int st=0;st<Slots;st++){s.score_ready[r][st].init(4);s.prob_ready[r][st].init(4);}
  cutlass::arch::fence_barrier_init();
 }
 for(int x=tid;x<8*N;x+=Threads)s.ones[x]=Element(1.f);
 cutlass::arch::fence_view_async_shared();__syncthreads();
 if(tid>=Producer) {
  cutlass::arch::warpgroup_reg_dealloc<32>();
  if(tid==Producer) {
   s.qr.arrive_and_expect_tx(Rows*M*D*sizeof(Element));
   for(int r=0;r<Rows;r++) {
    auto src=local_tile(p.q.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,min(i0+r,p.L-1)*4+h),Shape<_64,_32>{},make_coord(qt,0));
    auto dst=make_tensor(make_smem_ptr(s.q[r]),SQ{});auto sl=p.q.get_slice(_0{});
    copy(p.q.with(reinterpret_cast<uint64_t&>(s.qr)),sl.partition_S(src),sl.partition_D(dst));
   }
   for(int seq=0;seq<p.L/LN;seq++) {
    int st=seq%Stages;
    if(seq>=Stages){s.empty[st].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
    s.full[st].arrive_and_expect_tx(2*Rows*LN*D*sizeof(Element)+M*LN*sizeof(float));
    for(int r=0;r<Rows;r++) {
     int ih=min(i0+r,p.L-1)*4+h;
     auto ks=local_tile(p.k.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto vs=local_tile(p.v.get_tma_tensor(make_shape(p.L,32,p.L*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{}),vd=make_tensor(make_smem_ptr(s.v[st][r]),SK{});
     auto kl=p.k.get_slice(_0{}),vl=p.v.get_slice(_0{});
     copy(p.k.with(reinterpret_cast<uint64_t&>(s.full[st])),kl.partition_S(ks),kl.partition_D(kd));
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[st])),vl.partition_S(vs),vl.partition_D(vd));
    }
    auto bs=p.bias.get_tma_tensor(make_shape(256,LN/4,p.L/LN,nq,4))(_,_,seq,qt,h);
    auto bd=make_tensor(make_smem_ptr(s.bias[st]),SB{});auto bl=p.bias.get_slice(_0{});
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.full[st])),bl.partition_S(bs),bl.partition_D(bd));
   }
  }
  return;
 }
 if(tid>=128) {
  cutlass::arch::warpgroup_reg_dealloc<56>();
  int r=(tid-128)/128,t=tid%128;
  float nm[2]={0.f,0.f},c=p.scale*1.4426950408889634f;
  for(int seq=0;seq<nk;seq++) {
   int st=seq%Slots;
   s.score_ready[r][st].wait((seq/Slots)&1);asm volatile("":::"memory");
   float values[16];
   #pragma unroll
   for(int u=0;u<4;u++) {
    float4 x=*reinterpret_cast<float4 const*>(s.score[r][st]+u*512+t*4);
    values[4*u]=x.x;values[4*u+1]=x.y;values[4*u+2]=x.z;values[4*u+3]=x.w;
   }
   if(seq==0) {
    #pragma unroll
    for(int row=0;row<2;row++) {
     float mx=values[row*2];
     #pragma unroll
     for(int u=0;u<4;u++){mx=fmaxf(mx,values[4*u+row*2]);mx=fmaxf(mx,values[4*u+row*2+1]);}
     mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
     if(mx!=-INFINITY)nm[row]=-(mx*c)-64.f;
     else if(t%32==0)atomicOr(&s.bad,1);
    }
   }
   #pragma unroll
   for(int n=0;n<16;n++)values[n]=ex2(fmaf(values[n],c,nm[(n%4)/2]));
   #pragma unroll
   for(int n=0;n<8;n++) {
    int e=(2*n)%4,u=(2*n)/4;
    int q=16*(t/32)+(t%32)/4+8*(e/2),k=8*u+2*(t%4);
    auto packed=__floats2bfloat162_rn(values[2*n],values[2*n+1]);
    *reinterpret_cast<uint32_t*>(s.prob[r][st]+SP{}(q,k))=reinterpret_cast<uint32_t const&>(packed);
   }
   cutlass::arch::fence_view_async_shared();__syncwarp();
   if(t%32==0)s.prob_ready[r][st].arrive();
  }
  return;
 }
 cutlass::arch::warpgroup_reg_alloc<224>();
 int t=tid;
 QK qk;SPV pv;SLS ls;
 auto tq=qk.get_slice(t);auto tp=pv.get_slice(t);auto tl=ls.get_slice(t);
 using Score=decltype(partition_fragment_C(qk,Shape<_64,_32>{}));
 using Output=decltype(partition_fragment_C(pv,Shape<_64,_32>{}));
 using Denom=decltype(partition_fragment_C(ls,Shape<_64,_8>{}));
 Score sc[Rows];Output out[Rows];Denom den[Rows];
 #pragma unroll
 for(int r=0;r<Rows;r++){clear(out[r]);clear(den[r]);warpgroup_fence_operand(out[r]);warpgroup_fence_operand(den[r]);}
 auto ones=make_tensor(make_smem_ptr(s.ones),S1{});auto lb=tl.partition_fragment_B(ones);
 s.qr.wait(0);asm volatile("":::"memory");
 auto issue_qk=[&](auto rc,int seq) __attribute__((always_inline)) {
  constexpr int r=decltype(rc)::value;
  int kt=seq/Ratio,st=kt%Stages,cx=seq%Ratio;
  #pragma unroll
  for(int u=0;u<4;u++) {
   float4 b=*reinterpret_cast<float4 const*>(s.bias[st]+cx*M*N+u*512+t*4);
   sc[r](4*u)=b.x;sc[r](4*u+1)=b.y;sc[r](4*u+2)=b.z;sc[r](4*u+3)=b.w;
  }
  auto sq=make_tensor(make_smem_ptr(s.q[r]),SQ{});auto qa=tq.partition_fragment_A(sq);
  auto sk_all=make_tensor(make_smem_ptr(s.k[st][r]),SK{});
  auto sk=local_tile(sk_all,Shape<_32,_32>{},make_coord(cx,0));auto kb=tq.partition_fragment_B(sk);
  warpgroup_fence_operand(sc[r]);warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<2;kk++)gemm(qk,qa(_,_,kk),kb(_,_,kk),sc[r]);
  warpgroup_commit_batch();
 };
 auto issue_pv=[&](auto rc,int seq) __attribute__((always_inline)) {
  constexpr int r=decltype(rc)::value;
  int st=seq%Slots;
  s.prob_ready[r][st].wait((seq/Slots)&1);asm volatile("":::"memory");
  auto pp=make_tensor(make_smem_ptr(s.prob[r][st]),SP{});
  auto pa=tp.partition_fragment_A(pp);auto la=tl.partition_fragment_A(pp);
  auto sv_all=make_tensor(make_smem_ptr(s.v[(seq/Ratio)%Stages][r]),SV{});
  auto sv=local_tile(sv_all,Shape<_32,_32>{},make_coord(0,seq%Ratio));auto vb=tp.partition_fragment_B(sv);
  warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<2;kk++)gemm(pv,pa(_,_,kk),vb(_,_,kk),out[r]);
  #pragma unroll
  for(int kk=0;kk<2;kk++)gemm(ls,la(_,_,kk),lb(_,_,kk),den[r]);
  warpgroup_commit_batch();
 };
 auto publish=[&](auto rc,auto waiting,int seq) __attribute__((always_inline)) {
  constexpr int r=decltype(rc)::value;
  warpgroup_wait<decltype(waiting)::value>();warpgroup_fence_operand(sc[r]);
  #pragma unroll
  for(int u=0;u<4;u++)*reinterpret_cast<float4*>(s.score[r][seq%Slots]+u*512+t*4)=make_float4(sc[r](4*u),sc[r](4*u+1),sc[r](4*u+2),sc[r](4*u+3));
  asm volatile("":::"memory");__syncwarp();
  if(t%32==0)s.score_ready[r][seq%Slots].arrive();
 };
 auto iteration=[&](auto first,int seq) __attribute__((always_inline)) {
  constexpr bool First=decltype(first)::value;
  if(seq%Ratio==0){s.full[(seq/Ratio)%Stages].wait(((seq/Ratio)/Stages)&1);asm volatile("":::"memory");}
  issue_qk(_0{},seq);issue_qk(_1{},seq);issue_qk(_2{},seq);issue_qk(_3{},seq);
  if constexpr(!First){issue_pv(_0{},seq-1);issue_pv(_1{},seq-1);issue_pv(_2{},seq-1);issue_pv(_3{},seq-1);}
  publish(_0{},Int<(First?0:4)+3>{},seq);publish(_1{},Int<(First?0:4)+2>{},seq);
  publish(_2{},Int<(First?0:4)+1>{},seq);publish(_3{},Int<(First?0:4)+0>{},seq);
  warpgroup_wait<0>();
  #pragma unroll
  for(int r=0;r<Rows;r++){warpgroup_fence_operand(out[r]);warpgroup_fence_operand(den[r]);warpgroup_fence_operand(sc[r]);}
  if constexpr(!First){if((seq-1)%Ratio==Ratio-1 && t%32==0)s.empty[((seq-1)/Ratio)%Stages].arrive();}
 };
 iteration(cute::true_type{},0);
 #pragma unroll 1
 for(int seq=1;seq<nk;seq++)iteration(cute::false_type{},seq);
 issue_pv(_0{},nk-1);issue_pv(_1{},nk-1);issue_pv(_2{},nk-1);issue_pv(_3{},nk-1);
 warpgroup_wait<0>();
 auto id=tp.partition_C(make_identity_tensor(Shape<_64,_32>{}));
 bool bad=false;
 #pragma unroll
 for(int r=0;r<Rows;r++) {
  warpgroup_fence_operand(out[r]);warpgroup_fence_operand(den[r]);
  auto ar=make_tensor(out[r].data(),flash::convert_layout_acc_rowcol(out[r].layout()));
  auto lr=make_tensor(den[r].data(),flash::convert_layout_acc_rowcol(den[r].layout()));
  #pragma unroll
  for(int row=0;row<2;row++) {
   float check=0;
   #pragma unroll
   for(int c=0;c<8;c++)check+=ar(row,c);
   check=lr(row,0)+check*0.f;
   bad|=!(lr(row,0)>0x1p-84f && lr(row,0)<0x1p-44f && check>0.f && check<INFINITY);
   float inv=1.f/lr(row,0);
   #pragma unroll
   for(int col=0;col<8;col++)ar(row,col)*=inv;
  }
  #pragma unroll
  for(int n=0;n<size(out[r]);n+=2) {
   auto x=__floats2bfloat162_rn(out[r](n),out[r](n+1));int q=qt*M+get<0>(id(n)),d=get<1>(id(n));
   if(i0+r<p.L)*reinterpret_cast<uint32_t*>(p.out+((int64_t(i0+r)*4+h)*p.L+q)*32+d)=reinterpret_cast<uint32_t const&>(x);
  }
 }
 if(__any_sync(0xffffffff,bad) && t%32==0)atomicOr(&s.bad,1);
 cutlass::arch::NamedBarrier::sync(128,1);
 if(t==0 && s.bad){int index=atomicAdd(p.fix,1);p.fix[index+1]=tile;}
}
}}
