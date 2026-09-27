 if constexpr(SOL_REG_CONSUMER>0)cutlass::arch::warpgroup_reg_alloc<SOL_REG_CONSUMER>();
 int wg=tid/128,t=tid%128,lane=t%32,wm=t/32;
 float acc[4][4]={},sum[2]={},mx[2]={Safe?-INFINITY:0.f,Safe?-INFINITY:0.f};
 bool missing_seed=false;
 float c=p.scale*1.4426950408889634f;
 uint32_t qa[2][4];
 auto ld4=[](uint32_t (&r)[4],Element const* ptr) __attribute__((always_inline)) {
  uint32_t a=__cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
 };
 auto ld2=[](uint32_t (&r)[2],Element const* ptr) __attribute__((always_inline)) {
  uint32_t a=__cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];":"=r"(r[0]),"=r"(r[1]):"r"(a));
 };
 auto ldt2=[](uint32_t (&r)[2],Element const* ptr) __attribute__((always_inline)) {
  uint32_t a=__cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];":"=r"(r[0]),"=r"(r[1]):"r"(a));
 };
 auto mma=[](float (&o)[4],uint32_t const* a,uint32_t const* b) __attribute__((always_inline)) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(o[0]),"+f"(o[1]),"+f"(o[2]),"+f"(o[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
 };
 s.qr.wait(0);asm volatile("":::"memory");
 #pragma unroll
 for(int kk=0;kk<2;kk++)ld4(qa[kk],s.q[wg]+SQ{}(make_coord(wm*16+lane%16,16*kk+8*(lane/16))));
 #pragma unroll 1
 for(int seq=0;seq<nk;seq++) {
  int kt=seq/Ratio,st=kt%Stages,cx=seq%Ratio;
  if(cx==0){s.full[st].wait((kt/Stages)&1);asm volatile("":::"memory");}
  s.bf[seq%4].wait((seq/4)&1);asm volatile("":::"memory");
  float sc[4][4];
  #pragma unroll
  for(int n=0;n<4;n++){
   float4 b=*reinterpret_cast<float4 const*>(s.bias[seq%4]+n*512+t*4);
   sc[n][0]=b.x;sc[n][1]=b.y;sc[n][2]=b.z;sc[n][3]=b.w;
  }
  #pragma unroll
  for(int kk=0;kk<2;kk++) {
   #pragma unroll
   for(int n=0;n<4;n++){
    uint32_t kb[2];
    ld2(kb,s.k[st][wg]+SK{}(make_coord(cx*N+n*8+lane%8,kk*16+8*((lane/8)%2))));
    mma(sc[n],qa[kk],kb);
   }
  }
  __syncwarp();if(lane==0)s.be[seq%4].arrive();
  #pragma unroll
  for(int row=0;row<2;row++) {
   if(Safe || seq==0) {
    float m=sc[0][row*2];
    #pragma unroll
    for(int n=0;n<4;n++){m=fmaxf(m,sc[n][row*2]);m=fmaxf(m,sc[n][row*2+1]);}
    m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));m*=c;
    if constexpr(Safe){
     float next=fmaxf(mx[row],m),alpha=mx[row]==-INFINITY?0.f:ex2(mx[row]-next);
     #pragma unroll
     for(int n=0;n<4;n++){acc[n][row*2]*=alpha;acc[n][row*2+1]*=alpha;}
     sum[row]*=alpha;mx[row]=next;
    }else{if(m!=-INFINITY)mx[row]=m+64.f;else missing_seed=true;}
   }
   #pragma unroll
   for(int n=0;n<4;n++){
    sc[n][row*2]=ex2(fmaf(sc[n][row*2],c,-mx[row]));
    sc[n][row*2+1]=ex2(fmaf(sc[n][row*2+1],c,-mx[row]));
   }
  }
  uint32_t pp[8];
  #pragma unroll
  for(int n=0;n<4;n++){
   #pragma unroll
   for(int row=0;row<2;row++){
    auto packed=__floats2bfloat162_rn(sc[n][row*2],sc[n][row*2+1]);
    pp[n*2+row]=reinterpret_cast<uint32_t const&>(packed);
    float2 fp=__bfloat1622float2(packed);sum[row]+=fp.x+fp.y;
   }
  }
  #pragma unroll
  for(int kk=0;kk<2;kk++){
   #pragma unroll
   for(int n=0;n<4;n++){
    uint32_t vb[2];
    ldt2(vb,s.v[st][wg]+SK{}(make_coord(cx*N+kk*16+lane%16,n*8)));
    mma(acc[n],pp+kk*4,vb);
   }
  }
  __syncwarp();if(cx==Ratio-1 && lane==0)s.empty[st].arrive();
 }
 #pragma unroll
 for(int row=0;row<2;row++){
  sum[row]+=__shfl_xor_sync(0xffffffff,sum[row],1);sum[row]+=__shfl_xor_sync(0xffffffff,sum[row],2);
 }
 if constexpr(!Safe){
  bool bad=missing_seed;
  #pragma unroll
  for(int row=0;row<2;row++)bad|=!(sum[row]>0x1p-84f && sum[row]<0x1p-44f);
  if(__any_sync(0xffffffff,bad) && lane==0 && atomicCAS(&s.bad,0,1)==0){int index=atomicAdd(p.fix,1);p.fix[index+1]=tile;}
 }
 #pragma unroll
 for(int row=0;row<2;row++){
  float inv=1.f/sum[row];int q=qt*M+wm*16+lane/4+row*8;
  #pragma unroll
  for(int n=0;n<4;n++){
   auto x=__floats2bfloat162_rn(acc[n][row*2]*inv,acc[n][row*2+1]*inv);
   int d=n*8+(lane%4)*2;
   if(i0+wg<p.L)*reinterpret_cast<uint32_t*>(p.out+((int64_t(i0+wg)*4+h)*p.L+q)*32+d)=reinterpret_cast<uint32_t const&>(x);
  }
 }
}
