 if constexpr(SOL_REG_CONSUMER>0)cutlass::arch::warpgroup_reg_alloc<SOL_REG_CONSUMER>();
 int wg=int(__reduce_max_sync(0xffffffffu,unsigned(tid)/128u)),t=tid%128;
 QK qk;PV pv;auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);
 using Score=decltype(partition_fragment_C(qk,Shape<_64,_32>{}));
 using Output=decltype(partition_fragment_C(pv,Shape<_64,_32>{}));
 Score score;Output acc[2];clear(acc[0]);clear(acc[1]);
 auto pp=make_tensor_like<Element>(make_tensor(score.data(),flash::convert_layout_acc_Aregs<PV>(score.layout())));
 float sum[2][2]={},mx[2][2]={{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f},{Safe?-INFINITY:0.f,Safe?-INFINITY:0.f}};
 bool missing_seed=false;float c=p.scale*1.4426950408889634f;
 warpgroup_fence_operand(acc[0]);warpgroup_fence_operand(acc[1]);
 s.qr.wait(0);asm volatile("":::"memory");
 auto issue_qk=[&](int seq,auto half) __attribute__((always_inline)) {
  constexpr int hh=decltype(half)::value;
  int kt=seq/(2*Ratio),st=kt%Stages,cx=(seq/2)%Ratio;
  if(seq<2*nk && seq%(2*Ratio)==0){s.full[st].wait((kt/Stages)&1);asm volatile("":::"memory");}
  if(seq<2*nk){s.bf[seq%8].wait((seq/8)&1);asm volatile("":::"memory");}
  #pragma unroll
  for(int u=0;u<4;u++){
   float4 b=*reinterpret_cast<float4 const*>(s.bias[seq%8]+u*512+t*4);
   score(4*u)=b.x;score(4*u+1)=b.y;score(4*u+2)=b.z;score(4*u+3)=b.w;
  }
  auto qall=make_tensor(make_smem_ptr(s.q[wg]),SQ{});
  auto qs=local_tile(qall,Shape<_64,_32>{},make_coord(hh,0));auto qa=tq.partition_fragment_A(qs);
  auto kall=make_tensor(make_smem_ptr(s.k[st][wg]),SK{});
  auto ks=local_tile(kall,Shape<_32,_32>{},make_coord(cx,0));auto kb=tq.partition_fragment_B(ks);
  warpgroup_fence_operand(score);warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<2;kk++)gemm(qk,qa(_,_,kk),kb(_,_,kk),score);
  warpgroup_commit_batch();
  if(seq<2*nk){__syncwarp();if(t%32==0)s.be[seq%8].arrive();}
 };
 auto exponentiate=[&](auto half,auto seed) __attribute__((always_inline)) {
  constexpr int hh=decltype(half)::value;
  auto sr=make_tensor(score.data(),flash::convert_layout_acc_rowcol(score.layout()));
  auto ar=make_tensor(acc[hh].data(),flash::convert_layout_acc_rowcol(acc[hh].layout()));
  #pragma unroll
  for(int row=0;row<2;row++){
   if constexpr(Safe || decltype(seed)::value){
    float m=sr(row,0);
    #pragma unroll
    for(int col=1;col<8;col++)m=fmaxf(m,sr(row,col));
    m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));m*=c;
    if constexpr(Safe){
     float next=fmaxf(mx[hh][row],m),alpha=mx[hh][row]==-INFINITY?0.f:ex2(mx[hh][row]-next);
     #pragma unroll
     for(int col=0;col<8;col++)ar(row,col)*=alpha;
     sum[hh][row]*=alpha;mx[hh][row]=next;
    }else{if(m!=-INFINITY)mx[hh][row]=m+64.f;else missing_seed=true;}
   }
   #pragma unroll
   for(int col=0;col<8;col++){
    float x=sr(row,col);
    if constexpr(Safe)sr(row,col)=x==-INFINITY?0.f:ex2(fmaf(x,c,-mx[hh][row]));
    else sr(row,col)=ex2(fmaf(x,c,-mx[hh][row]));
    sum[hh][row]+=sr(row,col);
   }
  }
 };
 auto pack=[&]() __attribute__((always_inline)) {
  auto packed=recast<uint32_t>(pp);
  #pragma unroll
  for(int n=0;n<8;n++){auto x=__floats2bfloat162_rn(score(2*n),score(2*n+1));packed(n)=reinterpret_cast<uint32_t const&>(x);}
 };
 auto issue_pv=[&](int seq,auto half) __attribute__((always_inline)) {
  constexpr int hh=decltype(half)::value;
  auto vall=make_tensor(make_smem_ptr(s.v[(seq/(2*Ratio))%Stages][wg]),SV{});
  auto vs=local_tile(vall,Shape<_32,_32>{},make_coord(0,(seq/2)%Ratio));auto vb=tp.partition_fragment_B(vs);
  warpgroup_fence_operand(pp);if constexpr(Safe)warpgroup_fence_operand(acc[hh]);
  warpgroup_arrive();
  #pragma unroll
  for(int kk=0;kk<2;kk++)gemm(pv,pp(_,_,kk),vb(_,_,kk),acc[hh]);
  warpgroup_commit_batch();
 };
 auto release=[&](int seq) __attribute__((always_inline)) {
  if(seq%(2*Ratio)==2*Ratio-1 && t%32==0)s.empty[(seq/(2*Ratio))%Stages].arrive();
 };
 auto drain=[&]() __attribute__((always_inline)) {
  warpgroup_wait<0>();warpgroup_fence_operand(score);warpgroup_fence_operand(pp);
  warpgroup_fence_operand(acc[0]);warpgroup_fence_operand(acc[1]);
 };
 if constexpr(Safe){
  auto step=[&](int seq,auto half) __attribute__((always_inline)){
   issue_qk(seq,half);warpgroup_wait<0>();warpgroup_fence_operand(score);
   exponentiate(half,cute::true_type{});pack();issue_pv(seq,half);drain();release(seq);
  };
  for(int seq=0;seq<2*nk;seq+=2){step(seq,_0{});step(seq+1,_1{});}
 }else{
  issue_qk(0,_0{});warpgroup_wait<0>();warpgroup_fence_operand(score);
  exponentiate(_0{},cute::true_type{});pack();issue_qk(1,_1{});issue_pv(0,_0{});
  auto step=[&](int seq,auto half,auto seed) __attribute__((always_inline)) {
   constexpr int hh=decltype(half)::value;
   warpgroup_wait<1>();warpgroup_fence_operand(score);
   exponentiate(half,seed);
   warpgroup_wait<0>();warpgroup_fence_operand(pp);release(seq-1);
   pack();issue_qk(seq+1,Int<1-hh>{});issue_pv(seq,half);
  };
  step(1,_1{},cute::true_type{});
  int seq=2;
  #pragma unroll 1
  for(;seq+7<2*nk;seq+=8){
   step(seq,_0{},cute::false_type{});step(seq+1,_1{},cute::false_type{});
   step(seq+2,_0{},cute::false_type{});step(seq+3,_1{},cute::false_type{});
   step(seq+4,_0{},cute::false_type{});step(seq+5,_1{},cute::false_type{});
   step(seq+6,_0{},cute::false_type{});step(seq+7,_1{},cute::false_type{});drain();
  }
  #pragma unroll 1
  for(;seq<2*nk;seq+=2){step(seq,_0{},cute::false_type{});step(seq+1,_1{},cute::false_type{});drain();}
  drain();release(2*nk-1);
 }
 auto id=pv.get_slice(t).partition_C(make_identity_tensor(Shape<_64,_32>{}));
 bool bad=missing_seed;
 #pragma unroll
 for(int hh=0;hh<2;hh++){
  auto ar=make_tensor(acc[hh].data(),flash::convert_layout_acc_rowcol(acc[hh].layout()));
  #pragma unroll
  for(int row=0;row<2;row++){
   float val=sum[hh][row];val+=__shfl_xor_sync(0xffffffff,val,1);val+=__shfl_xor_sync(0xffffffff,val,2);
   if constexpr(!Safe)bad|=!(val>0x1p-84f && val<0x1p-44f);
   float inv=val>0.f?1.f/val:0.f;
   #pragma unroll
   for(int col=0;col<8;col++)ar(row,col)*=inv;
  }
  #pragma unroll
  for(int n=0;n<size(acc[hh]);n+=2){
   auto x=__floats2bfloat162_rn(acc[hh](n),acc[hh](n+1));int q=qt*M+hh*64+get<0>(id(n)),d=get<1>(id(n));
   if(i0+wg<p.L)*reinterpret_cast<uint32_t*>(p.out+((int64_t(i0+wg)*4+h)*p.L+q)*32+d)=reinterpret_cast<uint32_t const&>(x);
  }
 }
 if constexpr(!Safe){if(__any_sync(0xffffffff,bad) && t%32==0 && atomicCAS(&s.bad,0,1)==0){int index=atomicAdd(p.fix,1);p.fix[index+1]=tile;}}
}
