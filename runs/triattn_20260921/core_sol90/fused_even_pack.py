"""Exact ordinary arithmetic with early P storage in even score registers.

Any hot range correction requests the original SAFE computation of the CTA;
packed BF16 values are never rescaled as if they were unrounded FP32 values.
The original non-fast schedules and their running rescale are unchanged.
"""


def transform(s, full_q=False):
    if full_q:
        old = '            cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);'
        assert s.count(old) == 1
        s = s.replace(old, '            cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);')
        old = '''        if constexpr(kFast) {
            auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});
            warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);
        } else {warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);}'''
        assert s.count(old) == 2
        s = s.replace(old, '        warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);')

    old = '            if constexpr (kPrmtPack) {'
    assert s.count(old) == 1
    s = s.replace(old, '''            if constexpr(kFast){dst32(pr)=__float_as_uint(acc(2*pr));}
            else if constexpr (kPrmtPack) {''')

    start = s.index('    auto exp_chunk =')
    stop = s.index('    // E-phase token ring', start)
    block = s[start:stop]
    old = '''                    "ex2.approx.ftz.f32 %3,%3;\\n"'''
    assert block.count(old) == 1
    block = block.replace(old, old + '''
                    "{.reg .b32 ep0,ep1;\\n"
                    "cvt.rn.bf16x2.f32 ep0,%1,%0;\\n"
                    "cvt.rn.bf16x2.f32 ep1,%3,%2;\\n"
                    "mov.b32 %0,ep0;\\n"
                    "mov.b32 %2,ep1;}\\n"''')
    old = '''                        : "+f"(s_rc(mi,ni+0)), "+f"(s_rc(mi,ni+1)), "+f"(s_rc(mi,ni+2)), "+f"(s_rc(mi,ni+3)) : "f"(c_l2), "f"(nm[hh][mi]));'''
    assert block.count(old) == 1
    block = block.replace(old, old + '''
                    s_rc(mi,ni+1)=0.f;s_rc(mi,ni+3)=0.f;''')
    old = '        warpgroup_fence_operand(acc);'
    assert block.count(old) == 1
    block = block.replace(old, '''        if constexpr(kFast){
            #pragma unroll
            for(int pair=0;pair<8;++pair){asm volatile("" : "+f"(acc(2*pair)));}
        }else{warpgroup_fence_operand(acc);}''')
    s = s[:start] + block + s[stop:]

    start = s.index('    auto l_check =')
    stop = s.index('    // dep:', start)
    block = s[start:stop]
    old = '        if constexpr (!kSafe) {'
    assert block.count(old) == 1
    block = block.replace(old, '''        if constexpr(kFast){
            constexpr uint32_t lo=uint32_t(127-84)<<23;
            constexpr uint32_t width=(uint32_t(127-44)<<23)-lo;
            bool outside=false;
            #pragma unroll
            for(int hh=0;hh<2;++hh){
                warpgroup_fence_operand(acc_o[hh]);
                auto lr=make_tensor(acc_l[hh].data(),flash::convert_layout_acc_rowcol(acc_l[hh].layout()));
                #pragma unroll
                for(int mi=0;mi<kNRows;++mi)outside|=(__float_as_uint(lr(mi,0))-lo)>width;
            }
            // No new offset or rounded-P rescale: SAFE overwrites this CTA.
            bad|=__any_sync(0xffffffffu,outside);
        }else if constexpr (!kSafe) {''')
    return s[:start] + block + s[stop:]
