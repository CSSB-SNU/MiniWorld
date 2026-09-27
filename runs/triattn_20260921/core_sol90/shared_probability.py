"""Same-consumer shared P, three slots, original four-score QK2-ahead schedule.

At body k, pack P(k-1) reuses the slot of P(k-4). Every warp retired PV(k-4)
at body k-1's wait2, before its shared-P publication barrier. Thus that prior
WG barrier proves all readers are finished before the current stores. After
stores, proxy fence plus a WG barrier publishes all rows before SS PV issue.
Prologue slots have no prior readers; tail's old slot is already protected
by the preceding body barrier. No extra pre-store barrier is required.
"""
import re


def transform(s, full_q=False, matrix_store=False):
    old='    using TiledMmaPV = decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}));'
    assert s.count(old)==1
    s=s.replace(old,'''    static constexpr bool kSharedP=(kFlags_&1073741824)!=0 && !kSafe;
    using TiledMmaPVR = decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}));
    using TiledMmaPVS = decltype(make_tiled_mma(SM90::GMMA::MMA_64x40x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}));
    using TiledMmaPV = std::conditional_t<kSharedP,TiledMmaPVS,TiledMmaPVR>;''')
    s=s.replace('flash::convert_layout_acc_Aregs<typename T::TiledMmaPV>', 'flash::convert_layout_acc_Aregs<typename T::TiledMmaPVR>')
    old='    struct SharedStorage {\n        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 1024> smem_q;'
    assert s.count(old)==1
    s=s.replace(old,'''    using SmemLayoutP=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,Int<CW>>{}));
    struct SharedPStorage { cute::array_aligned<Element,R*3*64*CW,1024> smem_probability; };
    struct NoPStorage {};
    struct SharedStorage : std::conditional_t<kSharedP,SharedPStorage,NoPStorage> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, 1024> smem_q;''')
    # Hot P operands are shared descriptors; no hot register-A fence remains.
    s,n=re.subn(r'warpgroup_fence_operand\((PCb\[[^\]]+\])\);',r'if constexpr(!kFast){warpgroup_fence_operand(\1);}',s)
    assert n>=8,n
    for helper in ('issue_pv','issue_pv_prefenced'):
        a=s.index('    auto '+helper+' =')
        b=s.index('    auto ',a+8)
        block=s[a:b]
        old='auto hc, auto cc, auto stc) __attribute__((always_inline))'
        assert block.count(old)==1
        block=block.replace(old,'auto hc, auto cc, auto stc, int pseq=0) __attribute__((always_inline))')
        block=block.replace('        warpgroup_fence_operand(tP);','        if constexpr(!kFast){warpgroup_fence_operand(tP);}')
        old='''        #pragma unroll
        for (int kb = 0; kb < kNKB; ++kb) { cute::gemm(tiled_mma_pv, tP(_, _, kb), fused_v_operand(tV(_, _, kb, c, st),Int<st>{},Int<c>{}), acc_o[hh]); }'''
        assert block.count(old)==1
        block=block.replace(old,'''        if constexpr(kFast){
            auto sp=make_tensor(make_smem_ptr(shared.smem_probability.data()+(cwg*3+pseq%3)*64*CW),typename T::SmemLayoutP{});
            auto pa=wg_mma_pv.partition_fragment_A(sp);
            #pragma unroll
            for(int kb=0;kb<kNKB;++kb){cute::gemm(tiled_mma_pv,pa(_,_,kb),fused_v_operand(tV(_,_,kb,c,st),Int<st>{},Int<c>{}),acc_o[hh]);}
        }else{
'''+old+'''
        }''')
        s=s[:a]+block+s[b:]
    a=s.index('    auto pack_chunk =');b=s.index('    auto chunk_rowmax =',a)
    block=s[a:b]
    old='decltype(p_proto)& tP) __attribute__((always_inline))'
    assert block.count(old)==1
    block=block.replace(old,'decltype(p_proto)& tP, int pseq=0) __attribute__((always_inline))')
    start=block.index('        auto dst32 =')
    stop=block.rindex('    };')
    original=block[start:stop]
    block=block[:start]+'''        if constexpr(kFast){
            Element* dest=shared.smem_probability.data()+(cwg*3+pseq%3)*64*CW;
            auto physical=as_position_independent_swizzle_layout(typename T::SmemLayoutP{});
            #pragma unroll
            for(int pair=0;pair<8;++pair){
                int e=(2*pair)%4,u=(2*pair)/4;
                int row=16*(t128/32)+(t128%32)/4+8*(e/2),col=8*u+2*(t128%4);
                auto packed=__floats2bfloat162_rn(acc(2*pair),acc(2*pair+1));
                *reinterpret_cast<uint32_t*>(dest+physical(row,col))=reinterpret_cast<uint32_t const&>(packed);
            }
            asm volatile("fence.proxy.async.shared::cta;":::"memory");
            // NamedBarrier user IDs0..2 map to hardware8..10. The original
            // CTA prologue rendezvous uses ID3/hardware11, so no collision.
            asm volatile("bar.sync %0,128;"::"r"(uint32_t(cwg+8)):"memory");
        }else{
'''+original+'''        }
'''+block[stop:]
    s=s[:a]+block+s[b:]
    if matrix_store:
        a=s.index('            auto physical=as_position_independent_swizzle_layout')
        b=s.index('            asm volatile("fence.proxy.async.shared::cta;',a)
        s=s[:a]+'''            auto sp=make_tensor(make_smem_ptr(dest),typename T::SmemLayoutP{});
            auto cprob=make_tensor_like<Element>(acc);auto words=recast<uint32_t>(cprob);
            #pragma unroll
            for(int pair=0;pair<8;++pair){
                auto packed=__floats2bfloat162_rn(acc(2*pair),acc(2*pair+1));
                words(pair)=reinterpret_cast<uint32_t const&>(packed);
            }
            auto r2s=make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N,Element>{},tiled_mma_qk);
            auto thread_copy=r2s.get_thread_slice(t128);
            copy(r2s,thread_copy.retile_S(cprob),thread_copy.partition_D(sp));
'''+s[b:]
    # Only the actual fast schedule needs explicit P sequence positions;
    # generic/SAFE and discarded schedule-W calls keep the default argument.
    a=s.index('    auto body =');b=s.index('    // drained state:',a)
    block=s[a:b]
    old='pack_chunk(accC[bp], PCb[pb]);'
    assert block.count(old)==2
    block=block.replace(old,'pack_chunk(accC[bp], PCb[pb],k-1);')
    old='issue_pv_prefenced(PCb[pb], tV, Int<hp>{}, Int<cp>{}, Int<(tp & 1) * R>{});'
    assert block.count(old)==1
    block=block.replace(old,old.replace('>{});','>{},k-1);'))
    s=s[:a]+block+s[b:]
    old='pack_chunk(accC[3], PCb[1]);'
    assert s.count(old)==1
    s=s.replace(old,'pack_chunk(accC[3], PCb[1],8*n_w-1);')
    old='issue_pv(PCb[1], tV0, _1{}, _3{}, Int<set * R>{});'
    assert s.count(old)==1
    s=s.replace(old,'issue_pv(PCb[1], tV0, _1{}, _3{}, Int<set * R>{},8*n_w-1);')
    if full_q:
        old='            cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);'
        assert s.count(old)==1
        s=s.replace(old,'            cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);')
        old='''        if constexpr(kFast) {
            auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});
            warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);
        } else {warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);}'''
        assert s.count(old)==2
        s=s.replace(old,'        warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);')
    return s
