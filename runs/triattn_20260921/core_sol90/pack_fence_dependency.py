"""Make all eight P conversions precede QK's existing hardware fence.

Empty operand fences do not constrain ptxas scheduling by themselves:
installed SASS moves several F2FPs after QK's WG.AR, then injects a PV WG.AR.
OR all packed P words, AND the existing host-initialized zero, and XOR that
zero into the ready future QK accumulator before its operand/hardware fence.
No in-flight P/score register is read: body wait2 retires the reused P slot,
and future score contains only completed bias LDS. Arithmetic bits unchanged.
"""


def transform(s, q_dependency=False, warp_dependency=False):
    if warp_dependency:
        assert not q_dependency
        old='''        if constexpr (kFast) { pack_chunk(accC[bp], PCb[pb]); warpgroup_fence_operand(PCb[pb]); }'''
        assert s.count(old)==1
        return s.replace(old,'''        if constexpr (kFast) {
            pack_chunk(accC[bp], PCb[pb]);warpgroup_fence_operand(PCb[pb]);
            auto packed_words=recast<uint32_t>(PCb[pb]);uint32_t packed_dep=0;
            #pragma unroll
            for(int word=0;word<8;++word)packed_dep|=packed_words(word);
            // Host params.zero is exactly0: every lane's mask is0xffffffff.
            // Depend on completed conversions without writing any Q or
            // accumulator register used by asynchronous MMA.
            __syncwarp(~(packed_dep&uint32_t(params.zero)));
        }''')
    old='auto hc, auto cc, auto stc) __attribute__((always_inline)) {   // S += Q_h K_c^T'
    assert s.count(old)==1
    s=s.replace(old,'auto hc, auto cc, auto stc, uint32_t packed_dep=0) __attribute__((always_inline)) {   // S += Q_h K_c^T')
    a=s.index('    auto issue_qk =');b=s.index('    auto issue_pv =',a)
    block=s[a:b]
    old='        warpgroup_fence_operand(acc);'
    assert block.count(old)==1
    block=block.replace(old,'''        if constexpr(kFast)acc(0)=__uint_as_float(__float_as_uint(acc(0))^packed_dep);
'''+old)
    s=s[:a]+block+s[b:]
    a=s.index('    auto body =');b=s.index('    // drained state:',a)
    block=s[a:b]
    old='''        if constexpr (kFast) { pack_chunk(accC[bp], PCb[pb]); warpgroup_fence_operand(PCb[pb]); }
        issue_qk(accC[bn], tK, Int<h2>{}, Int<c2>{}, Int<(t2 & 1) * R>{});'''
    assert block.count(old)==1
    block=block.replace(old,'''        uint32_t packed_dep=0;
        if constexpr (kFast) {
            pack_chunk(accC[bp], PCb[pb]);warpgroup_fence_operand(PCb[pb]);
            auto packed_words=recast<uint32_t>(PCb[pb]);
            #pragma unroll
            for(int word=0;word<8;++word)packed_dep|=packed_words(word);
            packed_dep&=uint32_t(params.zero);
        }
        issue_qk(accC[bn], tK, Int<h2>{}, Int<c2>{}, Int<(t2 & 1) * R>{},packed_dep);''')
    s=s[:a]+block+s[b:]
    if q_dependency:
        old='        if constexpr(kFast)acc(0)=__uint_as_float(__float_as_uint(acc(0))^packed_dep);'
        assert s.count(old)==1
        s=s.replace(old,'''        if constexpr(kFast){
            // QK(k) retired at body k's wait2. This is the SAME query
            // half as future QK(k+2); the other half QK(k+1) may pend.
            // Thus this cached A word has no remaining readers here.
            auto qw=recast<uint32_t>(hh==0?tQr0:tQr1);
            qw(0)^=packed_dep;warpgroup_fence_operand(qw(0));
        }''')
    return s
