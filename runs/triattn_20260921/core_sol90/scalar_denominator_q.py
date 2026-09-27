"""N32 PV and scalar sums of rounded BF16 P free four denominator registers.

Only installed hot flag1073741824 changes. Sum per-thread partial rows before
PV uses P, reduce the four column-owner lanes at rescale/output. This changes
sum association, requiring FP64 qualification; QK/PV remain native BF16/FP32.
"""

def transform(s):
    def replace(old,new,count=1):
        nonlocal s
        assert s.count(old)==count,(old,s.count(old),count)
        s=s.replace(old,new)
    replace('''    using TiledMmaPV = decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}));''',
'''    static constexpr int kOutN=(kFlags_ & 1073741824) ? 32 : 40;
    using TiledMmaPV = std::conditional_t<(kOutN==32),
        decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{},AtomLayout{})),
        decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}))>;''')
    replace('make_identity_tensor(make_shape(_64{}, _40{}))',
            'make_identity_tensor(make_shape(_64{}, Int<T::kOutN>{}))')
    replace('partition_fragment_C(tiled_mma_pv, make_shape(_64{}, _40{}))',
            'partition_fragment_C(tiled_mma_pv, make_shape(_64{}, Int<T::kOutN>{}))')
    replace('decltype(size(AccO{}))::value == 20',
            'decltype(size(AccO{}))::value == T::kOutN/2')
    replace('''    auto l_view0=make_tensor(acc_o[0].data()+16,AccL{}.layout());
    decltype(l_view0) acc_l[2]={l_view0,make_tensor(acc_o[1].data()+16,AccL{}.layout())};''',
'''    AccL scalar_l[2];
    auto l_view0=make_tensor(kFast?scalar_l[0].data():acc_o[0].data()+16,AccL{}.layout());
    decltype(l_view0) acc_l[2]={l_view0,make_tensor(kFast?scalar_l[1].data():acc_o[1].data()+16,AccL{}.layout())};''')
    replace('''            cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);''',
            '''            cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);''')
    replace('''            auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});
            warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);''',
            '''            warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);''',2)
    replace('''        desc.bitfield.leading_byte_offset_=(oneaddr-vaddr)>>4;''',
            '''        if constexpr(!kFast)desc.bitfield.leading_byte_offset_=(oneaddr-vaddr)>>4;''')
    replace('''        desc.reg32_[0] -= uint32_t((st*T::kStageElemsV+col*CW*kHeadDim)/8)<<16;''',
            '''        if constexpr(!kFast)desc.reg32_[0] -= uint32_t((st*T::kStageElemsV+col*CW*kHeadDim)/8)<<16;''')
    a=s.index('    auto issue_pv =')
    helper='''    auto sum_probability = [&](auto const& prob,auto hc) __attribute__((always_inline)) {
        if constexpr(kFast){
            constexpr int hh=decltype(hc)::value;
            auto packed=recast<uint32_t>(prob);
            auto lr=make_tensor(acc_l[hh].data(),flash::convert_layout_acc_rowcol(acc_l[hh].layout()));
            #pragma unroll
            for(int mi=0;mi<2;++mi){
                #pragma unroll
                for(int u=0;u<4;++u){
                    uint32_t bits=packed(2*u+mi);
                    lr(mi,0)=__fadd_rn(lr(mi,0),__uint_as_float(bits<<16));
                    lr(mi,0)=__fadd_rn(lr(mi,0),__uint_as_float(bits&0xffff0000u));
                }
            }
        }
    };
    auto total_denominator = [&](float partial) __attribute__((always_inline)) {
        if constexpr(kFast){
            partial=__fadd_rn(partial,__shfl_xor_sync(0xffffffffu,partial,1));
            partial=__fadd_rn(partial,__shfl_xor_sync(0xffffffffu,partial,2));
        }
        return partial;
    };
'''
    s=s[:a]+helper+s[a:]
    a=s.index('    auto issue_pv =');b=s.index('    auto pack_chunk =',a)
    block=s[a:b]
    old='''        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;'''
    assert block.count(old)==2
    block=block.replace(old,'''        sum_probability(tP,hc);
'''+old)
    s=s[:a]+block+s[b:]
    # The unused duplicate denominator column stays constant0 in hot code.
    replace('l_rc(mi, 0) *= f; l_rc(mi, 1) *= f;',
            'l_rc(mi, 0) *= f; if constexpr(!kFast)l_rc(mi, 1) *= f;',2)
    a=s.index('    auto l_check =');b=s.index('    // dep:',a)
    block=s[a:b]
    old='__float_as_uint(l_rc(mi, 0))';assert block.count(old)==2
    block=block.replace(old,'__float_as_uint(total_denominator(l_rc(mi, 0)))')
    s=s[:a]+block+s[b:]
    replace('float const l = l_rc(mi, 0);','float const l = total_denominator(l_rc(mi, 0));')
    return s
