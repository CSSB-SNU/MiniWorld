"""Hot-only producer lifetime reduction, with an optional mixed full-Q WG.

The old mixed24 producer spilled primarily in try_wait's suspend path. Use a
non-suspending parity test loop and specialize K/V outside the tile loop.
The complete producer WG releases registers after Q issue and mask derivation,
before any wait for a consumer. Generic and SAFE keep their original source.
"""
def transform(s, mixed=False):
    marker = '    CUTLASS_DEVICE void producer_expect('
    assert s.count(marker) == 1
    method = '''    CUTLASS_DEVICE void producer_spin_empty(int stage, uint32_t phase) {
        uint32_t addr = cute::cast_smem_ptr_to_uint(&st.empty[stage]);
        asm volatile("{ .reg .pred done; WAIT_EMPTY: \\n"
                     "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1; \\n"
                     "@!done bra WAIT_EMPTY; }" :: "r"(addr), "r"(phase ^ 1u) : "memory");
    }
'''
    s = s.replace(marker, method + marker)
    a = s.index('            // ---- the K/V stream:')
    b = s.index('        } else if (warp_idx_in_wg == 0 && lane_predicate) {', a)
    original = s[a:b]
    specialized = '''            auto stream_kv = [&](auto value_tag) __attribute__((always_inline)) {
                constexpr bool is_value = decltype(value_tag)::value;
                auto const& tma = is_value ? params.tma_v : params.tma_k;
                auto* data = is_value ? shared.smem_v.data() : shared.smem_k.data();
                auto sKV = make_tensor(make_smem_ptr(data), typename T::SmemLayoutK{});
                auto mKV = tma.get_tma_tensor(params.shape_qk)(_, _, h, _, b);
                auto gKV = local_tile(domain_offset(make_coord(key0, _0{}, _0{}), mKV),
                    make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}, _));
                auto block_tma = tma.get_slice(_0{});
                auto tG = group_modes<0, 3>(block_tma.partition_S(gKV));
                auto tS = group_modes<0, 3>(block_tma.partition_D(sKV));
                for (int j = 0; j < n_tiles; ++j) {
                    int const set = (j & 1) * R;
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        int const st = set + r;
                        if constexpr(is_value) pipe_kv.producer_spin_empty(st, (j >> 1) & 1);
                        else pipe_k.producer_spin_empty(st, (j >> 1) & 1);
                        pipe_kv.producer_expect(st, T::kBytesK);
                        copy(tma.with(*pipe_kv.full_barrier(st), 0), tG(_, j, i0+r), tS(_, st));
                    }
                }
            };
            if (warp_idx_in_wg == 1) stream_kv(cute::false_type{});
            else stream_kv(cute::true_type{});
'''
    # Hot shape N768 is divisible by R3, all hot rows share jb=0, je=n_tiles.
    s = s[:a] + '            if constexpr(kFast) {\n' + specialized + '            } else {\n' + original + '            }\n' + s[b:]
    old = '                    pipe_b.producer_wait_empty(mm, bph);'
    assert s.count(old) == 1
    s = s.replace(old, '                    if constexpr(kFast) pipe_b.producer_spin_empty(mm, bph);\n                    else pipe_b.producer_wait_empty(mm, bph);')
    old = '        cutlass::arch::warpgroup_reg_dealloc<32>();'
    assert s.count(old) == 1
    s = s.replace(old, '        if constexpr(!kFast) cutlass::arch::warpgroup_reg_dealloc<32>();')
    marker = '        if (dead) { '
    assert s.count(marker) == 1
    s = s.replace(marker, '        if constexpr(kFast) cutlass::arch::warpgroup_reg_dealloc<%d>();\n' % (24 if mixed else 32) + marker)
    if not mixed:
        return s
    marker = '    {\n    // ================================================= CONSUMERS'
    assert s.count(marker) == 1
    s = s.replace(marker, '    auto consume = [&](auto full_tag, auto regs_tag) __attribute__((always_inline)) {\n    constexpr bool kFullQR=decltype(full_tag)::value;\n    // ================================================= CONSUMERS')
    assert s.count('warpgroup_reg_alloc<160>();') == 1
    s = s.replace('warpgroup_reg_alloc<160>();', 'warpgroup_reg_alloc<decltype(regs_tag)::value>();')
    old = '    }   // consumers\n'
    assert s.count(old) == 1
    s = s.replace(old, '''    };   // consumers
    if constexpr(kFast){
        // Full-WG contracts: (24 + 168 + 160 + 160) * 128 == 65536.
        if(wg_idx==1) consume(cute::true_type{},Int<168>{});
        else consume(cute::false_type{},Int<160>{});
    }else consume(cute::false_type{},Int<160>{});
''')
    old = '            if constexpr(hh==0){'
    assert s.count(old) == 1
    s = s.replace(old, '            if constexpr(hh==0 || kFullQR){')
    old = '''            auto a1=tQr1(_,_,Int<0>{});
            warpgroup_fence_operand(tQr0);warpgroup_fence_operand(a1);'''
    assert s.count(old) == 2
    s = s.replace(old, '''            warpgroup_fence_operand(tQr0);
            if constexpr(kFullQR) warpgroup_fence_operand(tQr1);
            else {auto a1=tQr1(_,_,Int<0>{});warpgroup_fence_operand(a1);}''')
    return s
