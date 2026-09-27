"""M64/R2 four-score full-Q kernel, two resident cooperative CTAs.

Unlike the M128 two-score pair experiment, QK(k+2) issues BEFORE E(k),
and bias(k+3) is loaded one whole body before QK. Three shared P slots
are protected by the preceding WG publication barrier; Q aliases slot0.
"""
from cooperative_shared_probability import transform as base_transform


def transform(s):
    s=base_transform(s)
    def replace(old,new,count=1):
        nonlocal s
        assert s.count(old)==count,(old,s.count(old),count)
        s=s.replace(old,new)
    replace('// L768: four cooperative WGs, full Q RS, three shared P slots per WG.',
            '// L768: M64/R2, four scores/full Q/shared P, two cooperative CTAs.')
    replace('M=128,N=32,LN=128,Ratio=4,D=32,Rows=4,Stages=2',
            'M=64,N=32,LN=64,Ratio=2,D=32,Rows=2,Stages=3')
    replace('Consumers=512,Threads=512','Consumers=256,Threads=256')
    replace('__launch_bounds__(Threads,1)','__launch_bounds__(Threads,2)')
    # Q and K have the same64x32 geometry again, including TMA source boxes.
    s=s.replace('Shape<_128,_32>','Shape<_64,_32>').replace('Shape<_32,_128>','Shape<_32,_64>')
    replace('pextra[Rows][64*N]','pextra[Rows][2*64*N]')
    replace('bias[6][64*N]','bias[4][64*N]');replace('bf[6]','bf[4]');replace('be[6]','be[4]')
    replace('sizeof(Shared)<=232448','sizeof(Shared)<=116224')
    replace('for(int st=0;st<6;++st)','for(int st=0;st<4;++st)')
    replace('int st=seq%6;','int st=seq%4;')
    replace('if(seq>=6){s.be[st].wait(((seq/6)-1)&1);','if(seq>=4){s.be[st].wait(((seq/4)-1)&1);')
    replace('s.bf[seq%6].wait((seq/6)&1);','s.bf[seq%4].wait((seq/4)&1);')
    replace('s.bias[seq%6]','s.bias[seq%4]');replace('s.be[seq%6].arrive();','s.be[seq%4].arrive();')
    replace('for(int seq=0;seq<6;++seq)load_bias(seq);','for(int seq=0;seq<4;++seq)load_bias(seq);')
    replace('seq+6<48)load_bias(seq+6);','seq+4<24)load_bias(seq+4);',2)
    replace('load_kv(0);load_kv(1);','load_kv(0);load_kv(1);load_kv(2);')
    replace('seq/8+2<6)load_kv(seq/8+2);','seq/2+3<12)load_kv(seq/2+3);')
    s=s.replace('seq/8','seq/2').replace('seq%8','seq%2').replace('==7){','==1){')
    s=s.replace('(kt<6?kt:0)','(kt<12?kt:0)').replace('(seq/2)%4','seq%2')
    s=s.replace('seq<48','seq<24')
    replace('qt=tile%6,rg=(tile/6)%(L/Rows),h=tile/(6*(L/Rows))',
            'qt=tile%12,rg=(tile/12)%(L/Rows),h=tile/(12*(L/Rows))')
    replace('make_shape(256,8,48,6,4)','make_shape(256,8,24,12,4)')
    replace('x<6*(L/Rows)*4','x<12*(L/Rows)*4')
    replace('make_shape(256,N/4,2*L/N,L/M,4)','make_shape(256,N/4,L/N,L/M,4)')
    replace('hh=(idx/2048)%2,cc=idx/4096','hh=0,cc=idx/2048')
    replace('Score sc[3];','Score sc[4];')
    replace('for(int n=0;n<3;++n)warpgroup_fence_operand(sc[n]);','for(int n=0;n<4;++n)warpgroup_fence_operand(sc[n]);')
    replace('for(int hh=0;hh<2;++hh){','for(int hh=0;hh<1;++hh){',4)
    replace('constexpr int hh=decltype(hc)::value;','constexpr int hh=0;',3)
    replace('  auto qr1=make_fragment_like(qr0);\n','')
    s=s.replace('hh==0?qr0:qr1','qr0').replace('warpgroup_fence_operand(qr0);warpgroup_fence_operand(qr1);','warpgroup_fence_operand(qr0);')
    assert 'qr1' not in s
    replace('for(int n=0;n<4;++n)warpgroup_fence_operand(sc[n]);',
            'for(int n=0;n<(Safe?1:4);++n)warpgroup_fence_operand(sc[n]);')
    replace('return slot==2?s.pextra[wg]:s.q[wg]+slot*64*N;',
            'return slot==0?s.q[wg]:s.pextra[wg]+(slot-1)*64*N;')
    replace('if(hh==1){','if(hh==0){')
    replace('''   if constexpr(Safe)warpgroup_fence_operand(prob);
   warpgroup_arrive();''','''   if constexpr(Safe)warpgroup_fence_operand(prob);
   if constexpr(Safe || !decltype(prefenced)::value)warpgroup_arrive();''')
    a=s.index('  }else{\n   clear(sc[2]);');b=s.index('\n  auto id=',a)
    calls='\n'.join('    step(period*12+%d,Int<%d>{},cute::%s_type{});'%(i,i%4,'true' if i==0 else 'false') for i in range(12))
    s=s[:a]+'''  }else{
   clear(sc[3]);
   init(sc[0],0);issue_qk(sc[0],0,_0{});
   init(sc[1],1);issue_qk(sc[1],1,_0{});init(sc[2],2);drain();
   auto step=[&](int seq,auto sic,auto seed_possible) __attribute__((always_inline)){
    constexpr int si=decltype(sic)::value;
    auto& cur=sc[si];auto& future=sc[(si+2)%4];auto& old=sc[(si+3)%4];
    warpgroup_wait<2>();warpgroup_fence_operand(cur);
    if(seq>=3)release(seq-3);
    if(tid==0 && seq+4<24)load_bias(seq+4);
    pack(old,pp[0],seq-1);
    issue_qk(future,seq+2,_0{});
    if constexpr(decltype(seed_possible)::value){
     if(seq==0)exponentiate(cur,_0{},cute::true_type{});else exponentiate(cur,_0{},cute::false_type{});
    }else exponentiate(cur,_0{},cute::false_type{});
    issue_pv(pp[0],max(seq-1,0),_0{},cute::true_type{},seq-1);
    init(old,seq+3);
   };
   #pragma unroll 1
   for(int period=0;period<2;++period){
'''+calls+'''
    drain();rescale(sc[3]);
   }
   pack(sc[3],pp[0],23);issue_pv(pp[0],23,_0{},cute::false_type{},23);drain();release(23);
  }
'''+s[b:]
    marker=' attention<false><<<ct,Threads,sizeof(Shared),stream>>>(p);'
    replace(marker,''' int resident=0;
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,attention<false>,Threads,sizeof(Shared)));
 TORCH_CHECK(resident>=2,"two resident CTAs required; got ",resident," smem=",sizeof(Shared));
'''+marker)
    return s
