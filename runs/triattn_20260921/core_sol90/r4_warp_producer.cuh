if(tid>=Consumers){
 // This is one warp, so no warpgroup register reconfiguration is legal here.
 if(tid==Consumers){
  s.qr.arrive_and_expect_tx(Rows*M*D*sizeof(Element));
  #pragma unroll
  for(int r=0;r<Rows;++r){
   auto src=local_tile(p.q.get_tma_tensor(make_shape(L,32,L*4))(_,_,(i0+r)*4+h),Shape<_128,_32>{},make_coord(qt,0));
   auto dst=make_tensor(make_smem_ptr(s.q[r]),SQ{});auto sl=p.q.get_slice(_0{});
   copy(p.q.with(reinterpret_cast<uint64_t&>(s.qr)),sl.partition_S(src),sl.partition_D(dst));
  }
  #pragma unroll 1
  for(int kt=0;kt<L/LN;++kt){
   int st=kt%Stages;
   if(kt>=Stages){s.empty[st].wait(((kt/Stages)-1)&1);asm volatile("":::"memory");}
   s.full[st].arrive_and_expect_tx(2*Rows*LN*D*sizeof(Element));
   #pragma unroll
   for(int r=0;r<Rows;++r){
    auto ks=local_tile(p.k.get_tma_tensor(make_shape(L,32,L*4))(_,_,(i0+r)*4+h),Shape<_128,_32>{},make_coord(kt,0));
    auto vs=local_tile(p.v.get_tma_tensor(make_shape(L,32,L*4))(_,_,(i0+r)*4+h),Shape<_128,_32>{},make_coord(kt,0));
    auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{}),vd=make_tensor(make_smem_ptr(s.v[st][r]),SK{});
    auto kl=p.k.get_slice(_0{}),vl=p.v.get_slice(_0{});
    copy(p.k.with(reinterpret_cast<uint64_t&>(s.full[st])),kl.partition_S(ks),kl.partition_D(kd));
    copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[st])),vl.partition_S(vs),vl.partition_D(vd));
   }
   // Feed bias before waiting on a later KV stage. A second blocking thread
   // inside this same warp must not be used as an independent producer.
   #pragma unroll 1
   for(int hh=0;hh<8;++hh){
    if(kt){s.be[hh].wait((kt-1)&1);asm volatile("":::"memory");}
    auto src=p.bias.get_tma_tensor(make_shape(256,8,48,6,4))(_,_,kt*8+hh,qt,h);
    auto dst=make_tensor(make_smem_ptr(s.bias[hh]),SB{});auto sl=p.bias.get_slice(_0{});
    s.bf[hh].arrive_and_expect_tx(64*N*sizeof(float));
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bf[hh])),sl.partition_S(src),sl.partition_D(dst));
   }
  }
 }
 return;
}
