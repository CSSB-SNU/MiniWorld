"""Two query halves per iteration, two scores, three shared P pairs.

QK(pair j+1) commits before PV(pair j). The next wait1 retires the former
while leaving the latter in flight during E(pair j+1). Three P pairs let
the preceding publication prove all readers of the reused pair retired.
LN64 with three KV stages and four bias slots funds the six P tiles.
"""
from cooperative_shared_probability import transform as base_transform


def transform(s):
    s=base_transform(s)
    def replace(old,new,count=1):
        nonlocal s
        assert s.count(old)==count,(old,s.count(old),count)
        s=s.replace(old,new)
    replace('// L768: four cooperative WGs, full Q RS, three shared P slots per WG.',
            '// L768: four WGs, two scores, full Q RS, three shared P pairs per WG.')
    replace('LN=128,Ratio=4,D=32,Rows=4,Stages=2','LN=64,Ratio=2,D=32,Rows=4,Stages=3')
    replace('using SK=SQ;','using SK=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));')
    replace('make_ordered_layout(Shape<_32,_128>{},Step<_2,_1>{})','make_ordered_layout(Shape<_32,_64>{},Step<_2,_1>{})')
    replace('using TK=TQ;', 'using TK=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),G3{},ST{}),SK{},Shape<_64,_32>{},_1{}));')
    replace('pextra[Rows][64*N]','pextra[Rows][4*64*N]')
    replace('bias[6][64*N]','bias[4][64*N]');replace('bf[6]','bf[4]');replace('be[6]','be[4]')
    replace('for(int st=0;st<6;++st)','for(int st=0;st<4;++st)')
    replace('int st=seq%6;','int st=seq%4;')
    replace('if(seq>=6){s.be[st].wait(((seq/6)-1)&1);','if(seq>=4){s.be[st].wait(((seq/4)-1)&1);')
    replace('s.bf[seq%6].wait((seq/6)&1);','s.bf[seq%4].wait((seq/4)&1);')
    replace('s.bias[seq%6]','s.bias[seq%4]');replace('s.be[seq%6].arrive();','s.be[seq%4].arrive();')
    replace('for(int seq=0;seq<6;++seq)load_bias(seq);','for(int seq=0;seq<4;++seq)load_bias(seq);')
    replace('seq+6<48)load_bias(seq+6);','seq+4<48)load_bias(seq+4);',2)
    a=s.index(' auto load_kv=');b=s.index(' auto load_bias=',a)
    part=s[a:b];assert part.count('Shape<_128,_32>')==2
    s=s[:a]+part.replace('Shape<_128,_32>','Shape<_64,_32>')+s[b:]
    replace('load_kv(0);load_kv(1);','load_kv(0);load_kv(1);load_kv(2);')
    replace('if(tid==0 && seq/8+2<6)load_kv(seq/8+2);','if(tid==0 && seq/4+3<12)load_kv(seq/4+3);')
    s=s.replace('seq/8','seq/4').replace('seq%8','seq%4').replace('==7){','==3){')
    s=s.replace('(kt<6?kt:0)','(kt<12?kt:0)').replace('(seq/2)%4','(seq/2)%2')
    replace('Score sc[3];','Score sc[2];')
    replace('for(int n=0;n<3;++n)warpgroup_fence_operand(sc[n]);','for(int n=0;n<2;++n)warpgroup_fence_operand(sc[n]);')
    replace('''   int slot=(seq+3)%3;
   return slot==2?s.pextra[wg]:s.q[wg]+slot*64*N;''','''   int slot=seq%6;
   return slot>=2?s.pextra[wg]+(slot-2)*64*N:s.q[wg]+slot*64*N;''')
    replace('''    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    // The preceding publication follows waits retiring the old slot's PV.
    asm volatile("bar.sync %0,128;"::"r"(uint32_t(wg+8)):"memory");''','''    // Both halves are published together after the second pack.''')
    replace('   warpgroup_commit_batch();','   if constexpr(Safe)warpgroup_commit_batch();',2)
    replace('''   if constexpr(Safe)warpgroup_fence_operand(prob);
   warpgroup_arrive();''','''   if constexpr(Safe){warpgroup_fence_operand(prob);warpgroup_arrive();}
   // Hot QK pair's fence already covers the chained PV accumulator.''')
    replace('    auto pr=make_tensor(pend.data(),flash::convert_layout_acc_rowcol(pend.layout()));\n','')
    replace('''      if(hh==1){
       #pragma unroll
       for(int col=0;col<8;++col)pr(row,col)*=f;
      }''','''      // No exponentiated P is pending across this pair-period drain.''')
    a=s.index('  }else{\n   clear(sc[2]);');b=s.index('\n  auto id=',a)
    calls='\n'.join('    step(period*12+%d,Int<%d>{});'%(i,i) for i in range(12))
    s=s[:a]+'''  }else{
   auto issue_qk_pair=[&](int seq) __attribute__((always_inline)) {
    int kt=seq/4;
    if(seq<48 && seq%4==0){s.full[kt%Stages].wait((kt/Stages)&1);asm volatile("":::"memory");}
    auto ks=local_tile(make_tensor(make_smem_ptr(s.k[(kt<12?kt:0)%Stages][wg]),SK{}),Shape<_32,_32>{},make_coord((seq/2)%2,0));
    auto kb=tq.partition_fragment_B(ks);
    warpgroup_fence_operand(sc[0]);warpgroup_fence_operand(sc[1]);warpgroup_arrive();
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(qkr,qr0(_,_,kk),kb(_,_,kk),sc[0]);
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(qkr,qr1(_,_,kk),kb(_,_,kk),sc[1]);
    warpgroup_commit_batch();
    if(seq<48){__syncwarp();if(t%32==0){s.be[seq%4].arrive();s.be[(seq+1)%4].arrive();}}
   };
   auto issue_pv_pair=[&](int seq,int pseq) __attribute__((always_inline)) {
    auto vs=local_tile(make_tensor(make_smem_ptr(s.v[(seq/4)%Stages][wg]),SV{}),Shape<_32,_32>{},make_coord(0,(seq/2)%2));
    auto raw=tp.partition_fragment_B(vs);auto desc=raw.data().desc_;
    uint32_t va=cast_smem_ptr_to_uint(s.v[(seq/4)%Stages][wg])+uint32_t((seq/2)%2)*N*D*sizeof(Element);
    uint32_t oa=cast_smem_ptr_to_uint(s.ones);
    desc.bitfield.leading_byte_offset_=(oa-va)>>4;
    auto vb=make_tensor(GMMA::DescriptorIterator{desc},raw.layout());
    PVS pvs;pvs.accumulate_=GMMA::ScaleOut::One;
    auto sp0=make_tensor(make_smem_ptr(probability_ptr(pseq)),SP{});
    auto sp1=make_tensor(make_smem_ptr(probability_ptr(pseq+1)),SP{});
    auto pa0=pvs.get_slice(0).partition_fragment_A(sp0);
    auto pa1=pvs.get_slice(0).partition_fragment_A(sp1);
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(pvs,pa0(_,_,kk),vb(_,_,kk),acc[0]);
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(pvs,pa1(_,_,kk),vb(_,_,kk),acc[1]);
    warpgroup_commit_batch();
   };
   // A zero-probability group makes the first iteration use the same
   // unconditional wait1 as every later iteration. Pair slot2 is free;
   // the previous publication at pair1 protects its first reuse at pair2.
   {Score zero;clear(zero);pack(zero,pp[0],4);pack(zero,pp[1],5);}
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
   asm volatile("bar.sync %0,128;"::"r"(uint32_t(wg+8)):"memory");
   init(sc[0],0);init(sc[1],1);
   issue_qk_pair(0);issue_pv_pair(0,4);
   auto step=[&](int pair,auto pc) __attribute__((always_inline)){
    constexpr int off=decltype(pc)::value;
    warpgroup_wait<1>();
    warpgroup_fence_operand(sc[0]);warpgroup_fence_operand(sc[1]);
    // Pair j-2 retired at this wait; the previous publication already
    // proves all warps retired j-3, whose P pair is being overwritten.
    if(pair>=2)release(2*pair-3);
    if(tid==0 && 2*pair+4<48){load_bias(2*pair+4);load_bias(2*pair+5);}
    if constexpr(off==0){
     if(pair==0){exponentiate(sc[0],_0{},cute::true_type{});exponentiate(sc[1],_1{},cute::true_type{});}
     else {exponentiate(sc[0],_0{},cute::false_type{});exponentiate(sc[1],_1{},cute::false_type{});}
    }else{exponentiate(sc[0],_0{},cute::false_type{});exponentiate(sc[1],_1{},cute::false_type{});}
    pack(sc[0],pp[0],2*pair);pack(sc[1],pp[1],2*pair+1);
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    asm volatile("bar.sync %0,128;"::"r"(uint32_t(wg+8)):"memory");
    // Keep exactly two committed groups even at the last iteration.
    // The unused final QK writes zero-seeded scores and reads live stage0.
    init(sc[0],2*pair+2);init(sc[1],2*pair+3);
    issue_qk_pair(2*pair+2);
    issue_pv_pair(2*pair,2*pair);
   };
   #pragma unroll 1
   for(int period=0;period<2;++period){
'''+calls+'''
    drain();rescale(sc[0]);
   }
   release(47);
  }
'''+s[b:]
    assert s.count('sc[2]')==1 and 'Score sc[2];' in s
    return s
