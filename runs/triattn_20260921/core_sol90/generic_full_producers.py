"""Independent K/V TMA warps and one TMA transaction for two bias halves."""
def transform(s):
 decl='using SB=Layout<Shape<_256,Int<N/4>>,Stride<_1,_256>>;'
 assert decl in s
 s=s.replace(decl,decl+'\nusing SBF=Layout<Shape<_256,Int<N/4>,_2>,Stride<_1,_256,Int<64*N>>>;')
 s=s.replace('SB{},Shape<_256,Int<N/4>>{},_1{}','SBF{},Shape<_256,Int<N/4>,_2>{},_1{}')
 s=s.replace('s.be[st].init(Rows*4)','s.be[st].init(Rows*8)').replace('s.full[st].init(1)','s.full[st].init(2)')
 a=s.index('  if(tid==Consumers) {');b=s.index('  return;',a)
 s=s[:a]+'''  if(tid==Consumers+32) {
   s.qr.arrive_and_expect_tx(Rows*M*D*sizeof(Element));
   for(int r=0;r<Rows;r++) {
    auto src=local_tile(p.q.get_tma_tensor(make_shape(768,32,768*4))(_,_,min(i0+r,767)*4+h),Shape<_128,_32>{},make_coord(qt,0));
    auto dst=make_tensor(make_smem_ptr(s.q[r]),SQ{});auto sl=p.q.get_slice(_0{});
    copy(p.q.with(reinterpret_cast<uint64_t&>(s.qr)),sl.partition_S(src),sl.partition_D(dst));
   }
   for(int seq=0;seq<768/LN;seq++) {
    int st=seq%Stages;
    if(seq>=Stages){s.empty[st].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
    s.full[st].arrive_and_expect_tx(Rows*LN*D*sizeof(Element));
    for(int r=0;r<Rows;r++) {
     auto src=local_tile(p.k.get_tma_tensor(make_shape(768,32,768*4))(_,_,min(i0+r,767)*4+h),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto dst=make_tensor(make_smem_ptr(s.k[st][r]),SK{});auto sl=p.k.get_slice(_0{});
     copy(p.k.with(reinterpret_cast<uint64_t&>(s.full[st])),sl.partition_S(src),sl.partition_D(dst));
    }
   }
  }
  if(tid==Consumers+64) {
   for(int seq=0;seq<768/LN;seq++) {
    int st=seq%Stages;
    if(seq>=Stages){s.empty[st].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
    s.full[st].arrive_and_expect_tx(Rows*LN*D*sizeof(Element));
    for(int r=0;r<Rows;r++) {
     auto src=local_tile(p.v.get_tma_tensor(make_shape(768,32,768*4))(_,_,min(i0+r,767)*4+h),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto dst=make_tensor(make_smem_ptr(s.v[st][r]),SK{});auto sl=p.v.get_slice(_0{});
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[st])),sl.partition_S(src),sl.partition_D(dst));
    }
   }
  }
  if(tid==Consumers) {
   for(int seq=0;seq<768/N;seq++) {
    int st=seq%4;
    if(seq>=4){s.be[st].wait(((seq/4)-1)&1);asm volatile("":::"memory");}
    s.bf[st].arrive_and_expect_tx(128*N*sizeof(float));
    auto all=p.bias.get_tma_tensor(make_shape(256,N/4,2*768/N,nq,4))(_,_,_,qt,h);
    auto bs=local_tile(all,Shape<_256,Int<N/4>,_2>{},make_coord(0,0,seq));
    auto bd=make_tensor(make_smem_ptr(s.bias[2*st]),SBF{});auto bl=p.bias.get_slice(_0{});
    copy(p.bias.with(reinterpret_cast<uint64_t&>(s.bf[st])),bl.partition_S(bs),bl.partition_D(bd));
   }
  }
'''+s[b:]
 old='if(seq<2*nk){s.bf[seq%8].wait((seq/8)&1);asm volatile("":::"memory");}'
 assert old in s
 s=s.replace(old,'if(seq<2*nk && seq%2==0){s.bf[(seq/2)%4].wait((seq/8)&1);asm volatile("":::"memory");}')
 s=s.replace('s.be[seq%8].arrive()','s.be[(seq/2)%4].arrive()')
 assert s.count('SBF{},Shape<_256,Int<N/4>,_2>{},_1{}')==2
 return s
