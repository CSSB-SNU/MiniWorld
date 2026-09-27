"""Order packed P before the QK fence through the real bias-ready probe.

The existing SASS still has F2FP writes after a QK WARPGROUP.ARRIVE. This
control uses the packed values in the address of a useful mbarrier test,
whose result controls the later bias wait. No arithmetic/format change.
"""
def transform(s,both,active=True):
 marker='    CUTLASS_DEVICE bool test_full(int stage, uint32_t phase)'
 pos=s.index(marker)
 s=s[:pos]+'''    CUTLASS_DEVICE bool test_full_dep(int stage,uint32_t phase,uint32_t dep,int zero) {
        uint32_t ready;
        uint32_t addr=cute::cast_smem_ptr_to_uint(&st.full[stage])+(dep&uint32_t(zero));
        asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p,[%1],%2; selp.b32 %0,1,0,p; }" : "=r"(ready) : "r"(addr),"r"(phase) : "memory");
        return ready;
    }
'''+s[pos:]
 old='        if constexpr (kFast) { pack_chunk(accC[bp], PCb[pb]); warpgroup_fence_operand(PCb[pb]); }\n'
 assert s.count(old)==1
 s=s.replace(old,'')
 pos=s.index('        // early non-blocking probes')
 s=s[:pos]+old+s[pos:]
 old='        if constexpr (h3 == 0 && !kNoProbe) { if (need_b3) { b3_ready = pipe_b.test_full(c3, bph3); } }'
 assert s.count(old)==1
 condition='(h3 == 0 || kFast)' if both else 'h3 == 0'
 probe_guard='(kFast || !kNoProbe)' if active else '!kNoProbe'
 new='''        if constexpr (CONDITION && PROBE_GUARD) {
            if(need_b3){
                if constexpr(kFast){
                    auto pw=recast<uint32_t>(PCb[pb]);uint32_t a,b,dep;
                    asm("lop3.b32 %0,%1,%2,%3,0xfe;" : "=r"(a) : "r"(pw(0)),"r"(pw(1)),"r"(pw(2)));
                    asm("lop3.b32 %0,%1,%2,%3,0xfe;" : "=r"(b) : "r"(pw(3)),"r"(pw(4)),"r"(pw(5)));
                    asm("lop3.b32 %0,%1,%2,%3,0xfe;" : "=r"(dep) : "r"(a),"r"(b),"r"(pw(6)));
                    dep|=pw(7);
                    b3_ready=pipe_b.test_full_dep(c3,bph3,dep,params.zero);
                }else b3_ready=pipe_b.test_full(c3,bph3);
            }
        }'''.replace('CONDITION',condition).replace('PROBE_GUARD',probe_guard)
 s=s.replace(old,new)
 if both:
  s=s.replace('bool const need_b3 = (h3 == 0) &&', 'bool const need_b3 = (h3 == 0 || kFast) &&')
  s=s.replace('if constexpr (h3 == 0) { if (need_b3 && !b3_ready)', 'if constexpr (h3 == 0 || kFast) { if (need_b3 && !b3_ready)')
 return s
