"""Let each TMA producer warp compute four EX2 values per consumer thread.

Only the fast L768 specialization uses the helper queues. Other kernels keep
their existing producer protocol. Two per-warp shared request/response slots
keep the next body's probability packing behind an explicit completion wait.
Producer waits poll and service the EX2 queues, avoiding a cyclic wait between
TMA empties and consumers waiting for EX2. No kernel arithmetic is approximated.
"""
def transform(s):
    marker='template <int kFlags_ = 0>\nstruct Traits {'
    assert s.count(marker)==1
    s=s.replace(marker,'''template<bool Enabled> struct ExpHelperStorage {};
template<> struct ExpHelperStorage<true> {
    alignas(16) float e_input[3][2][128][4];
    alignas(16) float e_output[3][2][128][4];
    cutlass::arch::ClusterBarrier e_ready[3][2][4];
    cutlass::arch::ClusterBarrier e_done[3][2][4];
};

'''+marker)
    assert s.count('    struct SharedStorage {')==2  # Pipe and Traits
    marker='    struct SharedStorage {\n        cute::array_aligned<Element'
    assert s.count(marker)==1
    s=s.replace(marker,'    struct SharedStorage : ExpHelperStorage<(kFlags_ & 1073741824) != 0> {\n        cute::array_aligned<Element')
    old='        shared.bad = 0;'
    assert s.count(old)==1
    s=s.replace(old,'''        if constexpr(kFast){
            for(int r=0;r<3;++r)for(int slot=0;slot<2;++slot)for(int w=0;w<4;++w){
                shared.e_ready[r][slot][w].init(1);
                shared.e_done[r][slot][w].init(1);
            }
        }
'''+old)
    # Recover the eight Q-register slots for helper counters and addresses.
    old='static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0 || (kFlags_ & 1073741824) != 0;'
    assert s.count(old)==1
    s=s.replace(old,'static constexpr bool kQinRegs = (kFlags_ & 1048576) != 0;')
    old='''        if constexpr(kFast) {
            tiled_mma_qkr.accumulate_ = GMMA::ScaleOut::One;
            tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
            auto& tQr = hh==0?tQr0:tQr1;
            cute::gemm(tiled_mma_qkr, tQr(_,_,0), tK(_,_,0,c,st), acc);
            cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);
        } else if constexpr(kQinRegs) {'''
    assert s.count(old)==1
    s=s.replace(old,'        if constexpr(kQinRegs) {')
    # Entire producer warps participate in helper service. Only lane0 issues
    # the original TMA transactions; the register allocation stays full-WG.
    a=s.index('    if (wg_idx == 0) {');b=s.index('    {\n    // ================================================= CONSUMERS',a)
    producer=s[a:b]
    marker='        if ((warp_idx_in_wg == 1 || (kFast && warp_idx_in_wg == 2)) && lane_predicate) {'
    assert producer.count(marker)==1
    helper='''        int e_next[3]={0,0,0};
        auto help_exp = [&]() __attribute__((always_inline)) -> bool {
            bool worked=false;
            if constexpr(kFast){
                #pragma unroll
                for(int r=0;r<3;++r){
                    int seq=e_next[r],slot=seq&1;
                    bool ready=false;
                    if(lane==0 && seq<8*n_tiles)ready=shared.e_ready[r][slot][warp_idx_in_wg].test_wait((seq>>1)&1);
                    ready=__shfl_sync(0xffffffffu,int(ready),0);
                    if(ready){
                        // The acquire was performed by lane0. Warp sync
                        // carries its visibility to lanes reading the input.
                        __syncwarp();asm volatile("":::"memory");
                        int t=warp_idx_in_wg*32+lane;
                        float4 value=*reinterpret_cast<float4 const*>(shared.e_input[r][slot][t]);
                        asm volatile(
                            "ex2.approx.ftz.f32 %0,%0;\\n"
                            "ex2.approx.ftz.f32 %1,%1;\\n"
                            "ex2.approx.ftz.f32 %2,%2;\\n"
                            "ex2.approx.ftz.f32 %3,%3;\\n"
                            : "+f"(value.x),"+f"(value.y),"+f"(value.z),"+f"(value.w));
                        *reinterpret_cast<float4*>(shared.e_output[r][slot][t])=value;
                        asm volatile("":::"memory");__syncwarp();
                        if(lane==0)shared.e_done[r][slot][warp_idx_in_wg].arrive();
                        ++e_next[r];worked=true;
                    }
                }
            }
            return worked;
        };
        auto wait_empty = [&](auto& pipe,int stage,uint32_t phase) __attribute__((always_inline)) {
            if constexpr(kFast){
                for(;;){
                    bool ready=false;
                    if(lane==0)ready=pipe.st.empty[stage].test_wait(phase^1u);
                    ready=__shfl_sync(0xffffffffu,int(ready),0);
                    if(ready)break;
                    if(!help_exp())__nanosleep(16);
                }
                asm volatile("":::"memory");
            }else pipe.producer_wait_empty(stage,phase);
        };
'''
    producer=producer.replace(marker,helper+marker.replace('&& lane_predicate','&& (kFast || lane_predicate)'))
    producer=producer.replace('} else if (warp_idx_in_wg == 0 && lane_predicate) {','} else if (warp_idx_in_wg == 0 && (kFast || lane_predicate)) {')
    producer=producer.replace('pipe_k.producer_wait_empty(st, kph);','wait_empty(pipe_k,st,kph);')
    producer=producer.replace('pipe_kv.producer_wait_empty(st, kph);','wait_empty(pipe_kv,st,kph);')
    producer=producer.replace('pipe_b.producer_wait_empty(mm, bph);','wait_empty(pipe_b,mm,bph);')
    for name in ('K','V'):
        low=name.lower()
        old='''                    pipe_kv.producer_expect(st, T::kBytesNAME);
                    copy(params.tma_low.with(*pipe_kv.full_barrier(st), 0), tNAMEgNAME(_, j, i), tNAMEsNAME(_, st));'''.replace('NAME',name).replace('low',low)
        assert producer.count(old)==1
        producer=producer.replace(old,'                    if(!kFast || lane==0){\n'+old+'\n                    }')
    a0=producer.index('                    if (T::kThinH1 && (mm & 1)) {')
    b0=producer.index('\n                }\n            }\n        }',a0)
    producer=producer[:a0]+'                    if(!kFast || lane==0){\n'+producer[a0:b0]+'\n                    }'+producer[b0:]
    marker='        if constexpr (!kList) { break; }\n        }   // producer tile loop'
    assert producer.count(marker)==1
    producer=producer.replace(marker,'''        if constexpr(kFast){
            while(e_next[0]<8*n_tiles || e_next[1]<8*n_tiles || e_next[2]<8*n_tiles){
                if(!help_exp())__nanosleep(16);
            }
        }
'''+marker)
    s=s[:a]+producer+s[b:]
    # Completion can occur at a period drain before normal packing. The
    # completed counter prevents packing from reloading pre-rescale values.
    marker='    auto pack_chunk ='
    assert s.count(marker)==1
    consumer='''    int e_submitted=0,e_packed=0,e_completed=-1;
    auto finish_exp = [&](AccC& acc,int seq) __attribute__((always_inline)) {
        if constexpr(kFast){
            if(seq>e_completed){
                int slot=seq&1,w=t128/32;
                shared.e_done[cwg][slot][w].wait((seq>>1)&1);
                asm volatile("":::"memory");
                float4 value=*reinterpret_cast<float4 const*>(shared.e_output[cwg][slot][t128]);
                acc(0)=value.x;acc(1)=value.y;acc(2)=value.z;acc(3)=value.w;
                e_completed=seq;
            }
        }
    };
'''
    s=s.replace(marker,consumer+marker)
    old='''    auto pack_chunk = [&](AccC& acc, decltype(p_proto)& tP) __attribute__((always_inline)) {
        auto dst32'''
    assert s.count(old)==1
    s=s.replace(old,'''    auto pack_chunk = [&](AccC& acc, decltype(p_proto)& tP) __attribute__((always_inline)) {
        if constexpr(kFast)finish_exp(acc,e_packed++);
        auto dst32''')
    a=s.index('        if constexpr(kFast){',s.index('    auto exp_chunk ='))
    b=s.index('        }else{',a)
    s=s[:a]+'''        if constexpr(kFast){
            int seq=e_submitted++,slot=seq&1;
            float4 value;
            value.x=fmaf(acc(0),c_l2,nm[hh][0]);value.y=fmaf(acc(1),c_l2,nm[hh][0]);
            value.z=fmaf(acc(2),c_l2,nm[hh][1]);value.w=fmaf(acc(3),c_l2,nm[hh][1]);
            *reinterpret_cast<float4*>(shared.e_input[cwg][slot][t128])=value;
            asm volatile("":::"memory");__syncwarp();
            if(lane==0)shared.e_ready[cwg][slot][t128/32].arrive();
            // Native EX2 for the other twelve entries, unchanged rounding.
            #pragma unroll
            for(int mi=0;mi<kNRows;++mi){
                #pragma unroll
                for(int ni=2;ni<kNC;ni+=2){
                    asm volatile(
                        "fma.rn.ftz.f32 %0,%0,%2,%3;\\n"
                        "fma.rn.ftz.f32 %1,%1,%2,%3;\\n"
                        "ex2.approx.ftz.f32 %0,%0;\\n"
                        "ex2.approx.ftz.f32 %1,%1;\\n"
                        : "+f"(s_rc(mi,ni)),"+f"(s_rc(mi,ni+1)) : "f"(c_l2),"f"(nm[hh][mi]));
                }
            }
            acc(0)=0.f;acc(1)=0.f;acc(2)=0.f;acc(3)=0.f;
'''+s[b:]
    old='        l_check(accC[decltype(bufc)::value], 1);'
    assert s.count(old)==1
    s=s.replace(old,'        if constexpr(kFast)finish_exp(accC[decltype(bufc)::value],e_submitted-1);\n'+old)
    return s


def transform_static_finish(s):
    """Use the known pipeline position to collect each response exactly once.

    Body zero always follows either the prologue or a full drain, both of
    which collect the pending response. This removes the dynamic completed
    branch that caused ptxas C7520 in v3, while preserving drain rescaling.
    """
    s = transform(s)
    s = s.replace('int e_submitted=0,e_packed=0,e_completed=-1;', 'int e_submitted=0;')
    old = '''            if(seq>e_completed){
                int slot=seq&1,w=t128/32;
                shared.e_done[cwg][slot][w].wait((seq>>1)&1);
                asm volatile("":::"memory");
                float4 value=*reinterpret_cast<float4 const*>(shared.e_output[cwg][slot][t128]);
                acc(0)=value.x;acc(1)=value.y;acc(2)=value.z;acc(3)=value.w;
                e_completed=seq;
            }'''
    assert s.count(old) == 1
    s = s.replace(old, '''            int slot=seq&1,w=t128/32;
            shared.e_done[cwg][slot][w].wait((seq>>1)&1);
            asm volatile("":::"memory");
            float4 value=*reinterpret_cast<float4 const*>(shared.e_output[cwg][slot][t128]);
            acc(0)=value.x;acc(1)=value.y;acc(2)=value.z;acc(3)=value.w;''')
    s = s.replace('        if constexpr(kFast)finish_exp(acc,e_packed++);\n', '')
    marker = '        if constexpr (kFast) { pack_chunk(accC[bp], PCb[pb]); warpgroup_fence_operand(PCb[pb]); }'
    assert s.count(marker) == 1
    s = s.replace(marker, '''        if constexpr(kFast && dd!=0)finish_exp(accC[bp],16*p+ep);
''' + marker)
    # There are two schedules in the source; only the !kW schedule is hot.
    marker = '    pack_chunk(accC[0], PCb[0]);'
    assert s.count(marker) == 2
    s = s.replace(marker, '    if constexpr(kFast)finish_exp(accC[0],0);\n' + marker)
    marker = '    warpgroup_wait<0>();                                        // (half slot 4 is released by body 0)'
    assert s.count(marker) == 1
    s = s.replace(marker, marker + '\n    if constexpr(kFast)finish_exp(accC[1],1);')
    assert 'e_completed' not in s and 'e_packed' not in s
    return s


def transform_explicit_pv_fence(s):
    """Make the post-EX2 PV fence explicit on the uniform consumer path."""
    s = transform(s)
    marker = '''    auto issue_pv_prefenced = [&](decltype(p_proto)& tP, OpV const& tV, auto hc, auto cc, auto stc) __attribute__((always_inline)) {   // no fence on acc_o (chained PVs)
        constexpr int hh = decltype(hc)::value, c = decltype(cc)::value, st = decltype(stc)::value;'''
    assert s.count(marker) == 1
    return s.replace(marker, marker + '\n        warpgroup_fence_operand(tP);warpgroup_arrive();')


def transform_direct_pack(s):
    """Read helper results directly into P, never back into QK accumulators.

    Pending drain rescaling is remembered as one exact factor per row and
    applied at collection. This removes conditional writes to WGMMA score
    registers and removes the completed-result branch altogether.
    """
    s = transform(s)
    a = s.index('    int e_submitted=0,e_packed=0,e_completed=-1;')
    b = s.index('    auto chunk_rowmax =', a)
    original_pack = s[s.index('        auto dst32 = recast<uint32_t>(tP);', a):b]
    # Reuse the original generic packing loop verbatim; no changes to SAFE.
    assert original_pack.endswith('    };\n')
    generic_body = original_pack[:-len('    };\n')]
    s = s[:a] + '''    int e_submitted=0,e_packed=0;
    float e_scale0=1.f,e_scale1=1.f;
    auto pack_chunk = [&](AccC& acc, decltype(p_proto)& tP) __attribute__((always_inline)) {
        if constexpr(kFast){
            int seq=e_packed++,slot=seq&1;
            shared.e_done[cwg][slot][t128/32].wait((seq>>1)&1);
            asm volatile("":::"memory");
            float4 value=*reinterpret_cast<float4 const*>(shared.e_output[cwg][slot][t128]);
            auto p0=__floats2bfloat162_rn(value.x*e_scale0,value.y*e_scale0);
            auto p1=__floats2bfloat162_rn(value.z*e_scale1,value.w*e_scale1);
            auto dst32=recast<uint32_t>(tP);
            dst32(0)=reinterpret_cast<uint32_t const&>(p0);
            dst32(1)=reinterpret_cast<uint32_t const&>(p1);
            #pragma unroll
            for(int pr=2;pr<CW/4;++pr){
                auto pair=__floats2bfloat162_rn(acc(2*pr),acc(2*pr+1));
                dst32(pr)=reinterpret_cast<uint32_t const&>(pair);
            }
        }else{
''' + generic_body + '''        }
    };
''' + s[b:]
    old = '            int seq=e_submitted++,slot=seq&1;'
    assert s.count(old) == 1
    s = s.replace(old, old + '\n            e_scale0=1.f;e_scale1=1.f;')
    old = '        if constexpr(kFast)finish_exp(accC[decltype(bufc)::value],e_submitted-1);\n'
    assert s.count(old) == 1
    s = s.replace(old, '')
    old = '                    if (hh == h_pend) {'
    assert s.count(old) == 1
    s = s.replace(old, old + '''
                        if constexpr(kFast){
                            if(mi==0)e_scale0*=f;else e_scale1*=f;
                        }''')
    assert 'finish_exp' not in s and 'e_completed' not in s
    return s


def transform_packed_response(s):
    """Producer converts its four EX2 results into two final BF16 pairs.

    The consumer collects only two packed registers. Any hot periodic
    rescale requests original SAFE recomputation, because early BF16
    rounding is not assumed to commute with every extreme rescale.
    """
    s = transform_direct_pack(s)
    old = '    alignas(16) float e_output[3][2][128][4];'
    assert s.count(old) == 1
    s = s.replace(old, '    alignas(8) uint2 e_output[3][2][128];')
    old = '                        *reinterpret_cast<float4*>(shared.e_output[r][slot][t])=value;'
    assert s.count(old) == 1
    s = s.replace(old, '''                        auto p0=__floats2bfloat162_rn(value.x,value.y);
                        auto p1=__floats2bfloat162_rn(value.z,value.w);
                        shared.e_output[r][slot][t]=make_uint2(
                            reinterpret_cast<uint32_t const&>(p0),reinterpret_cast<uint32_t const&>(p1));''')
    old = '''            float4 value=*reinterpret_cast<float4 const*>(shared.e_output[cwg][slot][t128]);
            auto p0=__floats2bfloat162_rn(value.x*e_scale0,value.y*e_scale0);
            auto p1=__floats2bfloat162_rn(value.z*e_scale1,value.w*e_scale1);
            auto dst32=recast<uint32_t>(tP);
            dst32(0)=reinterpret_cast<uint32_t const&>(p0);
            dst32(1)=reinterpret_cast<uint32_t const&>(p1);'''
    assert s.count(old) == 1
    s = s.replace(old, '''            uint2 value=shared.e_output[cwg][slot][t128];
            auto dst32=recast<uint32_t>(tP);
            dst32(0)=value.x;dst32(1)=value.y;''')
    s = s.replace('    float e_scale0=1.f,e_scale1=1.f;\n', '')
    s = s.replace('            e_scale0=1.f;e_scale1=1.f;\n', '')
    old = '''                        if constexpr(kFast){
                            if(mi==0)e_scale0*=f;else e_scale1*=f;
                        }'''
    assert s.count(old) == 1
    s = s.replace(old, '')
    marker = '            if (__any_sync(0xffffffffu, out)) {'
    assert s.count(marker) == 1
    s = s.replace(marker, marker + '\n                if constexpr(kFast)bad=true; // Early BF16 helper result: rescaling requires original SAFE.')
    assert 'e_scale' not in s
    return s


def transform_wg_queue(s, packed=True):
    """One ready/done barrier per entire consumer WG, four warp arrivals.

    All four producer warps still compute disjoint lane ranges. Consumers
    now wait on one common completion address before the WG's next WGMMA.
    Each producer sees a request only after all four consumer warps publish.
    """
    s = transform_packed_response(s) if packed else transform(s)
    s = s.replace('ClusterBarrier e_ready[3][2][4];', 'ClusterBarrier e_ready[3][2];')
    s = s.replace('ClusterBarrier e_done[3][2][4];', 'ClusterBarrier e_done[3][2];')
    old = '''            for(int r=0;r<3;++r)for(int slot=0;slot<2;++slot)for(int w=0;w<4;++w){
                shared.e_ready[r][slot][w].init(1);
                shared.e_done[r][slot][w].init(1);
            }'''
    assert s.count(old) == 1
    s = s.replace(old, '''            for(int r=0;r<3;++r)for(int slot=0;slot<2;++slot){
                shared.e_ready[r][slot].init(4);
                shared.e_done[r][slot].init(4);
            }''')
    for name in ('e_ready', 'e_done'):
        s = s.replace(name+'[r][slot][warp_idx_in_wg]', name+'[r][slot]')
        s = s.replace(name+'[cwg][slot][t128/32]', name+'[cwg][slot]')
        s = s.replace(name+'[cwg][slot][w]', name+'[cwg][slot]')
    return s
