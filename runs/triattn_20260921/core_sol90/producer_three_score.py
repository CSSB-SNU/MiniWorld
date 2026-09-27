"""Three-score/two-P hot schedule for the FP32 WG-wide producer helper.

The former fourth score held prefetched bias. Recycle E(k-1)'s score directly
into QK(k+2) after packing instead, saving16 consumer registers at the cost
of bias-load lead time. Fully static chunk positions keep modulo3 descriptors
constant, with exactly the original drain/rescale positions and short streams.
"""
from producer_exp_helper import transform_wg_queue


def transform(s, helper=True, full_q=False, dependent_init=False):
    if helper:
        s = transform_wg_queue(s, packed=False)
    if full_q:
        assert not helper
        old = '            cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);'
        assert s.count(old) == 1
        s = s.replace(old, '            cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);')
    if dependent_init:
        assert not helper
        a=s.index('    auto init_chunk =')
        b=s.index('    auto issue_qk =',a)
        body=s[a:b]
        old='auto cc, auto hc) __attribute__((always_inline))'
        assert body.count(old)==1
        body=body.replace(old,'auto cc, auto hc, uint32_t ordered_zero=0) __attribute__((always_inline))')
        old='bias_thread + c * T::kSlotElems'
        assert body.count(old)==1
        body=body.replace(old,'bias_thread + ordered_zero + c * T::kSlotElems')
        s=s[:a]+body+s[b:]
    # The old score[3] references live only in unused generic lambdas and
    # the unchanged non-fast schedule. The new hot path never touches it.
    marker = '    if constexpr (T::kW) {'
    assert s.count(marker) == 1
    start = s.index(marker)
    end = s.index('    // ---- epilogue: O / l -> bf16', start)
    original = s[start:end]
    hot = '''    if constexpr(kFast){
        OpK const helper_k=mk_K(0);OpV const helper_v=mk_V(0);
        shared.barrier_q.wait(0);asm volatile("":::"memory");
        pipe_kv.wait_full(kv0,0);
        pipe_b.wait_full(0,0);
        init_chunk(accC[0],_0{},_0{});init_chunk(accC[1],_0{},_1{});
        issue_qk(accC[0],helper_k,_0{},_0{},_0{});
        issue_qk(accC[1],helper_k,_1{},_0{},_0{});
        bias_release(0,0u);bias_release(1,0u);
        warpgroup_wait<0>();
        warpgroup_fence_operand(accC[0]);warpgroup_fence_operand(accC[1]);
        // QK2 has the third score. QK3 reuses S0 after E0 was packed.
        pipe_b.wait_full(1,0);init_chunk(accC[2],_1{},_0{});
        issue_qk(accC[2],helper_k,_0{},_1{},_0{});bias_release(2,0u);
        exp_chunk(accC[0],0,0,false,cute::true_type{});
        pack_chunk(accC[0],PCb[0]);issue_pv(PCb[0],helper_v,_0{},_0{},_0{});
        exp_chunk(accC[1],1,0,false,cute::true_type{});
        init_chunk(accC[0],_1{},_1{});
        issue_qk(accC[0],helper_k,_1{},_1{},_0{});bias_release(3,0u);
        warpgroup_wait<0>();
        warpgroup_fence_operand(accC[0]);warpgroup_fence_operand(accC[1]);warpgroup_fence_operand(accC[2]);
        warpgroup_fence_operand(PCb[0]);warpgroup_fence_operand(PCb[1]);
        warpgroup_fence_operand(acc_o[0]);warpgroup_fence_operand(acc_o[1]);
        bad|=need_seed && seeded!=0xfu;
        cutlass::arch::NamedBarrier::sync(T::kNumMmaThreads,kRingBar0+3);
        auto helper_step=[&](auto seqc) __attribute__((always_inline)){
            constexpr int seq=decltype(seqc)::value,future=seq+2,previous=seq-1;
            constexpr int si=seq%3,fi=future%3,pi=previous&1;
            static_assert(fi==previous%3);
            warpgroup_wait<2>();
            warpgroup_fence_operand(accC[si]);warpgroup_fence_operand(PCb[pi]);
            if constexpr((seq&7)==7)pipe_k.release(kv0+((seq/8)&1)*R,warp_leader);
            if constexpr(seq>=3 && ((seq-3)&7)==7)pipe_kv.release(kv0+(((seq-3)/8)&1)*R,warp_leader);
            pack_chunk(accC[fi],PCb[pi]);warpgroup_fence_operand(PCb[pi]);
            if constexpr((future&1)==0){
                if(future<8*n_w)pipe_b.wait_full((future/2)&3,(future/8)&1);
            }
            if constexpr((future&7)==0){
                if(future<8*n_w)pipe_kv.wait_full(kv0+((future/8)&1)*R,(future/16)&1);
            }
            init_chunk(accC[fi],Int<(future/2)&3>{},Int<future&1>{});
            issue_qk(accC[fi],helper_k,Int<future&1>{},Int<(future/2)&3>{},Int<((future/8)&1)*R>{});
            if(future<8*n_w)bias_release(future&7,0u);
            exp_chunk(accC[si],seq&1,seq/2,false,cute::false_type{});
            issue_pv_prefenced(PCb[pi],helper_v,Int<previous&1>{},Int<(previous/2)&3>{},Int<((previous/8)&1)*R>{});
        };
        auto helper_drain=[&](auto seqc) __attribute__((always_inline)){
            constexpr int seq=decltype(seqc)::value;
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_o[0]);warpgroup_fence_operand(acc_o[1]);
            warpgroup_fence_operand(acc_l[0]);warpgroup_fence_operand(acc_l[1]);
            warpgroup_fence_operand(accC[0]);warpgroup_fence_operand(accC[1]);warpgroup_fence_operand(accC[2]);
            warpgroup_fence_operand(PCb[0]);warpgroup_fence_operand(PCb[1]);
            finish_exp(accC[seq%3],seq);l_check(accC[seq%3],1);
        };
        auto helper_finish=[&](auto seqc) __attribute__((always_inline)){
            constexpr int seq=decltype(seqc)::value;
            helper_drain(seqc);pack_chunk(accC[seq%3],PCb[1]);
            issue_pv(PCb[1],helper_v,_1{},_3{},Int<((seq/8)&1)*R>{});
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc_o[0]);warpgroup_fence_operand(acc_o[1]);
            warpgroup_fence_operand(acc_l[0]);warpgroup_fence_operand(acc_l[1]);
            pipe_kv.release(kv0+((seq/8)&1)*R,warp_leader);
        };
        do{
'''
    for seq in range(2, 48):
        hot += '            helper_step(Int<%d>{});\n' % seq
        if seq in (17, 33):
            hot += '            helper_drain(Int<%d>{});\n' % seq
        if seq % 8 == 7:
            if seq < 47:
                hot += '            if(n_w==%d){helper_finish(Int<%d>{});break;}\n' % (seq//8+1, seq)
            else:
                hot += '            helper_finish(Int<47>{});\n'
    hot += '        }while(false);\n    }else{\n' + original + '    } // unchanged non-fast schedules\n\n'
    if not helper:
        old = '        shared.barrier_q.wait(0);asm volatile("":::"memory");'
        assert hot.count(old) == 1
        hot = hot.replace(old, old + '''
        load_q_regs();
        warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);''')
        old = '            finish_exp(accC[seq%3],seq);l_check(accC[seq%3],1);'
        assert hot.count(old) == 1
        hot = hot.replace(old, '            l_check(accC[seq%3],1);')
    if dependent_init:
        old='            init_chunk(accC[fi],Int<(future/2)&3>{},Int<future&1>{});'
        assert hot.count(old)==1
        hot=hot.replace(old,'''            auto packed_words=recast<uint32_t>(PCb[pi]);
            uint32_t packed_sum=0;
            #pragma unroll
            for(int word=0;word<8;++word)packed_sum+=packed_words(word);
            // Params.zero is zero at every host launch, but is opaque to
            // ptxas: make all eight P results precede the next LDS address.
            uint32_t ordered_zero=packed_sum & uint32_t(params.zero);
            init_chunk(accC[fi],Int<(future/2)&3>{},Int<future&1>{},ordered_zero);''')
    return s[:start] + hot + s[end:]
