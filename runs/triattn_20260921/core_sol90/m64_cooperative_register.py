"""Four scores/two register P buffers, cooperative M64/R2, two CTAs.

The single M64 output frees20 registers versus M128; omit Q's8-register
cache as well, retaining full asynchronous QK2-ahead scheduling with P RS.
Removing shared P frees16KB, funding a fourth64-wide KV stage.
"""
from m64_cooperative_shared import transform as shared_transform


def transform(s):
    s=shared_transform(s)
    def replace(old,new,count=1):
        nonlocal s
        assert s.count(old)==count,(old,s.count(old),count)
        s=s.replace(old,new)
    replace('// L768: M64/R2, four scores/full Q/shared P, two cooperative CTAs.',
            '// L768: M64/R2, four scores/shared Q/register P, two cooperative CTAs.')
    replace('Rows=2,Stages=3','Rows=2,Stages=4')
    replace(' alignas(128) Element pextra[Rows][2*64*N];\n','')
    replace('load_kv(0);load_kv(1);load_kv(2);','load_kv(0);load_kv(1);load_kv(2);load_kv(3);')
    replace('seq/2+3<12)load_kv(seq/2+3);','seq/2+4<12)load_kv(seq/2+4);')
    a=s.index('  QKR qkr;');b=s.index('  warpgroup_fence_operand(acc[0]);',a)
    s=s[:a]+s[b:]
    old='''   if constexpr(!Safe){
    auto& qr=qr0;
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(qkr,qr(_,_,kk),kb(_,_,kk),a);
   }else{
    #pragma unroll
    for(int kk=0;kk<2;++kk)gemm(qk,qa(_,_,kk),kb(_,_,kk),a);
   }'''
    replace(old,'''   #pragma unroll
   for(int kk=0;kk<2;++kk)gemm(qk,qa(_,_,kk),kb(_,_,kk),a);''')
    a=s.index('  auto pack=');b=s.index('  auto issue_pv=',a)
    s=s[:a]+'''  auto pack=[&](auto& a,auto& prob,int pseq=0) __attribute__((always_inline)) {
   auto dst=recast<uint32_t>(prob);
   #pragma unroll
   for(int n=0;n<8;++n){auto x=__floats2bfloat162_rn(a(2*n),a(2*n+1));dst(n)=reinterpret_cast<uint32_t const&>(x);}
  };
'''+s[b:]
    replace('   if constexpr(Safe)warpgroup_fence_operand(prob);',
            '   if constexpr(Safe || !decltype(prefenced)::value)warpgroup_fence_operand(prob);')
    a=s.index('   if constexpr(!Safe){\n    PVS pvs;');b=s.index('\n   warpgroup_commit_batch();',a)
    s=s[:a]+'''   #pragma unroll
   for(int kk=0;kk<2;++kk)gemm(pv,prob(_,_,kk),vb(_,_,kk),acc[hh]);
'''+s[b:]
    replace('''   for(int n=0;n<(Safe?1:4);++n)warpgroup_fence_operand(sc[n]);''','''   for(int n=0;n<(Safe?1:4);++n)warpgroup_fence_operand(sc[n]);
   warpgroup_fence_operand(pp[0]);warpgroup_fence_operand(pp[1]);''')
    replace('if constexpr(Safe){warpgroup_fence_operand(pp[hh]);}','')
    replace('''    constexpr int si=decltype(sic)::value;
    auto& cur''','''    constexpr int si=decltype(sic)::value,pb=(si+3)%2;
    auto& cur''')
    replace('    warpgroup_wait<2>();warpgroup_fence_operand(cur);',
            '    warpgroup_wait<2>();warpgroup_fence_operand(cur);warpgroup_fence_operand(pp[pb]);')
    replace('    pack(old,pp[0],seq-1);','    pack(old,pp[pb],seq-1);warpgroup_fence_operand(pp[pb]);')
    replace('issue_pv(pp[0],max(seq-1,0),_0{},cute::true_type{},seq-1);',
            'issue_pv(pp[pb],max(seq-1,0),_0{},cute::true_type{},seq-1);')
    replace('pack(sc[3],pp[0],23);issue_pv(pp[0],23,_0{},cute::false_type{},23);',
            'pack(sc[3],pp[1],23);issue_pv(pp[1],23,_0{},cute::false_type{},23);')
    assert 'pextra' not in s and 'probability_ptr' not in s and 'qr0' not in s
    return s
