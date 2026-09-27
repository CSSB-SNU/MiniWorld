"""Per-consumer KV barriers and paired bias completion for p3 experiments."""
def transform(s,split=False):
 def sub(a,b,n=1):
  nonlocal s
  assert s.count(a)==n,(a,s.count(a),n)
  s=s.replace(a,b)
 sub('qr,full[Stages],bf[8];','qr,full[Stages][Rows],bf[4];')
 sub('empty[Stages],be[8];','empty[Stages][Rows],be[8];')
 sub('s.bf[st].init(1);s.be[st].init(Rows*4);',
     'if(st<4)s.bf[st].init(2);s.be[st].init(Rows*4);')
 sub('for(int st=0;st<Stages;st++){s.full[st].init(1);s.empty[st].init(Rows*4);}',
     'for(int st=0;st<Stages;st++)for(int r=0;r<Rows;r++){s.full[st][r].init(1);s.empty[st][r].init(4);}')
 sub('''    if(seq>=Stages){s.empty[st].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
    s.full[st].arrive_and_expect_tx(2*Rows*LN*D*sizeof(Element));
    for(int r=0;r<Rows;r++) {''',
     '''    for(int r=0;r<Rows;r++) {
     if(seq>=Stages){s.empty[st][r].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
     s.full[st][r].arrive_and_expect_tx(2*LN*D*sizeof(Element));''')
 sub('reinterpret_cast<uint64_t&>(s.full[st])','reinterpret_cast<uint64_t&>(s.full[st][r])',2)
 sub('s.bf[st].arrive_and_expect_tx(', 's.bf[st/2].arrive_and_expect_tx(')
 sub('reinterpret_cast<uint64_t&>(s.bf[st])','reinterpret_cast<uint64_t&>(s.bf[st/2])')
 sub('if(seq<2*nk){s.bf[seq%8].wait((seq/8)&1);',
     'if(seq<2*nk && seq%2==0){s.bf[(seq%8)/2].wait((seq/8)&1);')
 sub('s.full[st].wait((kt/Stages)&1);','s.full[st][wg].wait((kt/Stages)&1);')
 sub('s.empty[(seq/(2*Ratio))%Stages].arrive();','s.empty[(seq/(2*Ratio))%Stages][wg].arrive();')
 if split:
  sub('empty[Stages][Rows],be[8];','empty[Stages][Rows],kempty[Stages][Rows],be[8];')
  sub('s.full[st][r].init(1);s.empty[st][r].init(4);',
      's.full[st][r].init(2);s.empty[st][r].init(4);s.kempty[st][r].init(4);')
  sub('s.empty[st][r].wait(((seq/Stages)-1)&1);','s.kempty[st][r].wait(((seq/Stages)-1)&1);')
  sub('s.full[st][r].arrive_and_expect_tx(2*LN*D*sizeof(Element));',
      's.full[st][r].arrive_and_expect_tx(LN*D*sizeof(Element));')
  sub('     auto vs=local_tile(p.v.get_tma_tensor(make_shape(768,32,768*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0));\n','')
  sub('auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{}),vd=make_tensor(make_smem_ptr(s.v[st][r]),SK{});',
      'auto kd=make_tensor(make_smem_ptr(s.k[st][r]),SK{});')
  sub('auto kl=p.k.get_slice(_0{}),vl=p.v.get_slice(_0{});','auto kl=p.k.get_slice(_0{});')
  sub('     copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[st][r])),vl.partition_S(vs),vl.partition_D(vd));\n','')
  sub('  if(tid==Consumers+32) {','''  if(tid==Consumers+64) {
   for(int seq=0;seq<768/LN;seq++) {
    int st=seq%Stages;
    for(int r=0;r<Rows;r++) {
     if(seq>=Stages){s.empty[st][r].wait(((seq/Stages)-1)&1);asm volatile("":::"memory");}
     s.full[st][r].arrive_and_expect_tx(LN*D*sizeof(Element));
     int ih=min(i0+r,768-1)*4+h;
     auto vs=local_tile(p.v.get_tma_tensor(make_shape(768,32,768*4))(_,_,ih),Shape<Int<LN>,_32>{},make_coord(seq,0));
     auto vd=make_tensor(make_smem_ptr(s.v[st][r]),SK{});auto vl=p.v.get_slice(_0{});
     copy(p.v.with(reinterpret_cast<uint64_t&>(s.full[st][r])),vl.partition_S(vs),vl.partition_D(vd));
    }
   }
  }
  if(tid==Consumers+32) {''')
  sub(' auto release=[&](int seq)', ''' auto release_k=[&](int seq) __attribute__((always_inline)) {
  if(seq%(2*Ratio)==2*Ratio-1 && t%32==0)s.kempty[(seq/(2*Ratio))%Stages][wg].arrive();
 };
 auto release=[&](int seq)''')
  sub('   warpgroup_wait<2>();warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);',
      '   warpgroup_wait<2>();warpgroup_fence_operand(sc);warpgroup_fence_operand(prob);release_k(seq);')
  sub('exponentiate(score,half,cute::true_type{});pack(score,pp);issue_pv(seq,half,pp);drain();release(seq);',
      'release_k(seq);exponentiate(score,half,cute::true_type{});pack(score,pp);issue_pv(seq,half,pp);drain();release(seq);')
 return s
