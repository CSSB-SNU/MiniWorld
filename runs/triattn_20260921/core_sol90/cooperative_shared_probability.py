"""R4 cooperative QK2-ahead: full Q in registers, P in three shared slots.

After each WG loads its Q tile, a WG barrier transfers that storage to the
first two P slots. A third slot per WG replaces two of the eight bias slots.
Bias uses six slots, with unchanged KV ring; the 24-step period divides all
three rings. Dummy P(-1) occupies slot2, real P0 occupies slot0.
"""
import re


def transform(s):
    def replace(old,new,count=1):
        nonlocal s
        assert s.count(old)==count,(old,s.count(old),count)
        s=s.replace(old,new)

    replace('// L768 persistent CTA: one K/V fill, six query tiles, two consumer warpgroups.',
            '// L768: four cooperative WGs, full Q RS, three shared P slots per WG.')
    replace('using QK=','using SP=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>{}));\nusing QKR=decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>{}));\nusing PVS=decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>{}));\nusing QK=')
    replace(' alignas(128) float bias[8][64*N];',
            ' alignas(128) Element pextra[Rows][64*N];\n alignas(128) float bias[6][64*N];')
    replace('bf[8]','bf[6]');replace('be[8]','be[6]')
    replace('for(int st=0;st<8;++st)','for(int st=0;st<6;++st)')
    replace('int st=seq&7;','int st=seq%6;')
    replace('if(seq>=8){s.be[st].wait(((seq/8)-1)&1);',
            'if(seq>=6){s.be[st].wait(((seq/6)-1)&1);')
    replace('for(int seq=0;seq<8;++seq)load_bias(seq);','for(int seq=0;seq<6;++seq)load_bias(seq);')
    replace('s.bf[seq&7].wait((seq/8)&1);','s.bf[seq%6].wait((seq/6)&1);')
    replace('s.bias[seq&7]','s.bias[seq%6]')
    replace('s.be[seq&7].arrive();','s.be[seq%6].arrive();')
    replace('seq+8<48)load_bias(seq+8);','seq+6<48)load_bias(seq+6);',2)
    replace('  s.qr.wait(0);asm volatile("":::"memory");', '''  s.qr.wait(0);asm volatile("":::"memory");
  QKR qkr;qkr.accumulate_=GMMA::ScaleOut::One;
  auto qshared=make_tensor(make_smem_ptr(s.q[wg]),SQ{});
  auto qr0=qkr.get_slice(t).partition_fragment_A(local_tile(qshared,Shape<_64,_32>{},make_coord(0,0)));
  auto qr1=make_fragment_like(qr0);
  if constexpr(!Safe){
   auto qcopy=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,Element>{},qkr);
   auto qthread=qcopy.get_slice(t);
   #pragma unroll
   for(int hh=0;hh<2;++hh){
    auto qh=local_tile(qshared,Shape<_64,_32>{},make_coord(hh,0));
    copy(qcopy,qthread.partition_S(qh),qthread.retile_D(hh==0?qr0:qr1));
   }
   warpgroup_fence_operand(qr0);warpgroup_fence_operand(qr1);
   // All synchronous Q loads must finish before any warp reuses Q for P.
   asm volatile("bar.sync %0,128;"::"r"(uint32_t(wg+8)):"memory");
  }
  auto probability_ptr=[&](int seq) __attribute__((always_inline)) {
   int slot=(seq+3)%3;
   return slot==2?s.pextra[wg]:s.q[wg]+slot*64*N;
  };''')
    replace('''   #pragma unroll
   for(int kk=0;kk<2;++kk)gemm(qk,qa(_,_,kk),kb(_,_,kk),a);''','''   if constexpr(!Safe){
    auto& qr=hh==0?qr0:qr1;
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(qkr,qr(_,_,kk),kb(_,_,kk),a);
   }else{
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(qk,qa(_,_,kk),kb(_,_,kk),a);
   }''')
    replace('''  auto pack=[&](auto& a,auto& prob) __attribute__((always_inline)) {
   auto dst=recast<uint32_t>(prob);
   #pragma unroll
   for(int n=0;n<8;++n){auto x=__floats2bfloat162_rn(a(2*n),a(2*n+1));dst(n)=reinterpret_cast<uint32_t const&>(x);}
  };''','''  auto pack=[&](auto& a,auto& prob,int seq=0) __attribute__((always_inline)) {
   if constexpr(!Safe){
    auto sp=make_tensor(make_smem_ptr(probability_ptr(seq)),SP{});
    auto cp=make_tensor_like<Element>(a);auto words=recast<uint32_t>(cp);
    #pragma unroll
    for(int n=0;n<8;++n){auto x=__floats2bfloat162_rn(a(2*n),a(2*n+1));words(n)=reinterpret_cast<uint32_t const&>(x);}
    auto r2s=make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N,Element>{},qk);
    auto thread_copy=r2s.get_slice(t);
    copy(r2s,thread_copy.retile_S(cp),thread_copy.partition_D(sp));
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    // The preceding publication follows waits retiring the old slot's PV.
    asm volatile("bar.sync %0,128;"::"r"(uint32_t(wg+8)):"memory");
   }else{
    auto dst=recast<uint32_t>(prob);
    #pragma unroll
    for(int n=0;n<8;++n){auto x=__floats2bfloat162_rn(a(2*n),a(2*n+1));dst(n)=reinterpret_cast<uint32_t const&>(x);}
   }
  };''')
    replace('auto hc,auto prefenced) __attribute__((always_inline))',
            'auto hc,auto prefenced,int pseq=0) __attribute__((always_inline))')
    replace('''   warpgroup_fence_operand(prob);warpgroup_arrive();
   #pragma unroll
   for(int kk=0;kk<2;++kk)gemm(pv,prob(_,_,kk),vb(_,_,kk),acc[hh]);''','''   if constexpr(Safe)warpgroup_fence_operand(prob);
   warpgroup_arrive();
   if constexpr(!Safe){
    PVS pvs;pvs.accumulate_=GMMA::ScaleOut::One;
    auto sp=make_tensor(make_smem_ptr(probability_ptr(pseq)),SP{});
    auto pa=pvs.get_slice(0).partition_fragment_A(sp);
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(pvs,pa(_,_,kk),vb(_,_,kk),acc[hh]);
   }else{
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(pv,prob(_,_,kk),vb(_,_,kk),acc[hh]);
   }''')
    s,n=re.subn(r'warpgroup_fence_operand\((pp\[hh\]|previous)\);',r'if constexpr(Safe){warpgroup_fence_operand(\1);}',s)
    assert n==3,n
    replace('pack(old,previous);','pack(old,previous,seq-1);')
    replace('issue_pv(previous,max(seq-1,0),Int<1-hh>{},cute::true_type{});',
            'issue_pv(previous,max(seq-1,0),Int<1-hh>{},cute::true_type{},seq-1);')
    replace('pack(sc[2],pp[1]);issue_pv(pp[1],47,_1{},cute::false_type{});',
            'pack(sc[2],pp[1],47);issue_pv(pp[1],47,_1{},cute::false_type{},47);')
    replace('''    // Retire QK(k) and PV(k-3); QK(k+1) and PV(k-2) may remain. P(k-1) recycles
    // the slot formerly occupied by P(k-3), so no second wait is needed.''','''    // Retire QK(k) and PV(k-3); QK(k+1) and PV(k-2) may remain.
    // P(k-1) reuses P(k-4), retired before the preceding publication barrier.
    // Dummy P(-1) uses slot2, separately from real P0 in slot0.''')
    return s
