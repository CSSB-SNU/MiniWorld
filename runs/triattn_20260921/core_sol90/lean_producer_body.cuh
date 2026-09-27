// Exact fast-path producer, using the descriptor coordinates emitted by the
// installed CUTE path. The tensor maps still carry all global strides.
// Q/K/V coordinates: (D, S, H, pair-row, B); bias: (0, 0, chunk, qtile, BH).
// Each complete producer WG relinquishes registers before lanes specialize.
if constexpr(kFast) {
    cutlass::arch::warpgroup_reg_dealloc<kLeanProducerRegs>();
    int const pw = warp_idx & 3;
    if(pw == 1 && lane_predicate) {
        shared.barrier_q.arrive_and_expect_tx(T::kBytesQ * R);
        #pragma unroll 1
        for(int r=0;r<R;++r) {
            cute::SM90_TMA_LOAD_5D::copy(params.tma_q.get_tma_descriptor(),
                reinterpret_cast<uint64_t*>(&shared.barrier_q), uint64_t(1)<<60,
                shared.smem_q.data()+r*kBlockM*kHeadDim,
                0,g_qtile*kBlockM,g_h,min(g_rg*R+r,kShapeN-1),g_b);
        }
    }
    int first=0,last=kShapeN/32;
    bool dead=false;
    if(params.rowkind!=nullptr) {
        dead=params.rowkind[int64_t(g_b)*kShapeN]==2;
        first=min(params.kcstart[g_b],kShapeN/32-1);
        last=max(params.kcend[g_b],first+1);
    }
    if(dead) {
        if(params.force_fix!=0 && tid==0) {
            int const slot=atomicAdd(params.fix,1);
            params.fix[1+3*slot]=g_qtile;
            params.fix[2+3*slot]=g_rg;
            params.fix[3+3*slot]=g_bh;
        }
        return;
    }
    int const nt=(last-first+3)/4;
    if(lane_predicate && pw==0) {
        #pragma unroll 1
        for(int seq=0;seq<8*nt;++seq) {
            int const slot=seq&7;
            pipe_b.producer_wait_empty(slot,(seq>>3)&1);
            pipe_b.producer_expect(slot>>1,T::kBytesHalf);
            cute::SM90_TMA_LOAD_5D::copy(params.tma_b.get_tma_descriptor(),
                pipe_b.full_barrier(slot>>1),uint64_t(1)<<60,
                shared.smem_bias.data()+slot*T::kHalfElems,
                0,0,2*first+seq,g_qtile,g_bh);
        }
    } else if(lane_predicate && (pw==1 || pw==2)) {
        // A single role selection keeps only one tensor map and data base live.
        auto const* desc=pw==1?params.tma_k.get_tma_descriptor():params.tma_v.get_tma_descriptor();
        auto* data=pw==1?shared.smem_k.data():shared.smem_v.data();
        auto& free_pipe=pw==1?pipe_k:pipe_kv;
        #pragma unroll 1
        for(int j=0;j<nt;++j) {
            #pragma unroll 1
            for(int r=0;r<R;++r) {
                int const st=(j&1)*R+r;
                free_pipe.producer_wait_empty(st,(j>>1)&1);
                pipe_kv.producer_expect(st,T::kBytesK);
                cute::SM90_TMA_LOAD_5D::copy(desc,pipe_kv.full_barrier(st),uint64_t(1)<<60,
                    data+st*T::kStageElemsK,0,32*first+j*kBlockN,
                    g_h,min(g_rg*R+r,kShapeN-1),g_b);
            }
        }
    }
    return;
}
