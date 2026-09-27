// Full producer warpgroup: interleave TMA bias, X prefetch and Q/K/V WGMMA.
// Must be included inside the projected attention kernel, after CTA init.
if(tid>=Consumers){
 cutlass::arch::warpgroup_reg_dealloc<PRODUCER_REGS>();
 int pt=tid-Consumers;
 auto sync=[]() __attribute__((always_inline)){cutlass::arch::NamedBarrier::sync(128,7);};
 auto load_x=[&](int r,int kt) __attribute__((always_inline)){
  auto dst=make_tensor(make_smem_ptr(s.x[r]),SX{});
  auto src=local_tile(p.x.get_tma_tensor(make_shape(L,128,L))(_,_,i0+r),Shape<_128,_128>{},make_coord(kt,0));
  auto slice=p.x.get_slice(_0{});
  s.xr[r].arrive_and_expect_tx(128*128*sizeof(Element));
  copy(p.x.with(reinterpret_cast<uint64_t&>(s.xr[r])),slice.partition_S(src),slice.partition_D(dst));
 };
 if(pt==0){
  auto dst=make_tensor(make_smem_ptr(s.w),SWQ{});
  auto src=local_tile(p.w.get_tma_tensor(make_shape(512,128)),Shape<_32,_128>{},make_coord(h,0));
  auto slice=p.w.get_slice(_0{});
  s.wr.arrive_and_expect_tx(32*128*sizeof(Element));
  copy(p.w.with(reinterpret_cast<uint64_t&>(s.wr)),slice.partition_S(src),slice.partition_D(dst));
  load_x(0,qt);load_x(1,qt);
 }
 s.wr.wait(0);asm volatile("":::"memory");
 #pragma unroll
 for(int r=0;r<Rows;++r){
  s.xr[r].wait(0);asm volatile("":::"memory");
  auto w=make_tensor(make_smem_ptr(s.w),SWQ{});
  project<MQ>(s,w,r,0,0,pt);project<MQ>(s,w,r,1,0,pt);
 }
 cutlass::arch::fence_view_async_shared();sync();
 if(pt==0){
  s.qr.arrive();
  auto dst=make_tensor(make_smem_ptr(s.w),SWKV{});
  auto src=local_tile(p.wkv.get_tma_tensor(make_shape(256,128)),Shape<_64,_128>{},make_coord(h,0));
  auto slice=p.wkv.get_slice(_0{});
  s.wr.arrive_and_expect_tx(64*128*sizeof(Element));
  copy(p.wkv.with(reinterpret_cast<uint64_t&>(s.wr)),slice.partition_S(src),slice.partition_D(dst));
  load_x(0,0);load_x(1,0);
 }
 s.wr.wait(1);asm volatile("":::"memory");
 #pragma unroll 1
 for(int kt=0;kt<L/LN;++kt){
  int st=kt%Stages;
  if(kt>=Stages){s.empty[st].wait(((kt/Stages)-1)&1);asm volatile("":::"memory");}
  #pragma unroll
  for(int r=0;r<Rows;++r){
   s.xr[r].wait((kt+1)&1);asm volatile("":::"memory");
   auto w=make_tensor(make_smem_ptr(s.w),SWKV{});
   project<MKV>(s,w,r,0,st,pt);project<MKV>(s,w,r,1,st,pt);
   sync(); // TMA may overwrite X only after every producer lane drains MMA.
   if(pt==0 && kt+1<L/LN)load_x(r,kt+1);
  }
  cutlass::arch::fence_view_async_shared();sync();
  if(pt==0){
   s.full[st].arrive();
   #pragma unroll 1
   for(int hh=0;hh<8;++hh){
    if(kt){s.be[hh].wait((kt-1)&1);asm volatile("":::"memory");}
    auto dst=make_tensor(make_smem_ptr(s.bias[hh]),SB{});
    auto src=p.bias.get_tma_tensor(make_shape(256,8,48,6,4))(_,_,kt*8+hh,qt,h);
    auto slice=p.bias.get_slice(_0{});
    s.bf[hh].arrive_and_expect_tx(64*N*sizeof(float));
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bf[hh])),slice.partition_S(src),slice.partition_D(dst));
   }
  }
  sync();
 }
 return;
}
