"""Use the two idle producer warps to prepare shared FP16 Q/K for SS QK."""
def transform(name,s):
 from scaled_q_regs import transform as scaled
 s=scaled(name,s)
 if name!='triattn_m1_sm90.cuh':return s
 s=s.replace('kQinRegs = !kSafe','kQinRegs = false')
 # Preserve original SAFE input handling; these barriers are hot-only.
 marker='        cutlass::arch::ClusterTransactionBarrier barrier_q;'
 assert s.count(marker)==1
 s=s.replace(marker,marker+'\n        cutlass::arch::ClusterBarrier q_converted, k_converted[kStagesKV];')
 marker='        shared.barrier_q.init(1);'
 s=s.replace(marker,marker+'''
        if constexpr(!kSafe){
            shared.q_converted.init(2);
            #pragma unroll
            for(int st=0;st<T::kStagesKV;st++)shared.k_converted[st].init(2);
        }
''')
 a=s.index('    auto wait_kv_ready =');b=s.index('    // register copies of this warpgroup',a)
 s=s[:a]+'''    auto wait_kv_ready = [&](int st,uint32_t phase) __attribute__((always_inline)) {
        if constexpr(kSafe)pipe_kv.wait_full(st,phase);
        else {shared.k_converted[st].wait(phase);asm volatile("":::"memory");}
    };
'''+s[b:]
 # Dead rows still wait for the original TMA transaction before leaving.
 old='    shared.barrier_q.wait(0); asm volatile("" ::: "memory");\n    load_q_regs();'
 assert s.count(old)==2
 s=s.replace(old,'''    if constexpr(kSafe)shared.barrier_q.wait(0);else shared.q_converted.wait(0);
    asm volatile("" ::: "memory");
    load_q_regs();''')
 # Producer warps 0 and 1 retain their original bias and Q/K/V TMA streams.
 # Warps 2/3 jointly convert disjoint words after TMA completion. Each warp
 # fences its writes, then contributes exactly one arrival to the ready gate.
 marker='        if constexpr (!kList) { break; }\n        }   // producer tile loop'
 assert s.count(marker)==1
 conversion='''        if constexpr(!kSafe){
            if(warp_idx_in_wg>=2){
                int lane64=tid-64;
                auto convert_words=[&](uint32_t* words,int count,float factor) __attribute__((always_inline)) {
                    #pragma unroll 1
                    for(int index=lane64;index<count;index+=64){
                        uint32_t u=words[index];
                        float lo=__uint_as_float(u<<16),hi=__uint_as_float(u&0xffff0000u);
                        __half2 pair=__floats2half2_rn(lo*factor,hi*factor);
                        words[index]=reinterpret_cast<uint32_t const&>(pair);
                    }
                    cutlass::arch::fence_view_async_shared();
                    __syncwarp();
                };
                shared.barrier_q.wait(0);asm volatile("":::"memory");
                convert_words(reinterpret_cast<uint32_t*>(shared.smem_q.data()),
                              cute::cosize_v<typename T::SmemLayoutQ>/2,params.scale*1.4426950408889634f);
                if((tid&31)==0)shared.q_converted.arrive();
                for(int j=0;j<n_tiles;j++){
                    #pragma unroll
                    for(int r=0;r<R;r++){
                        int st=(j&1)*R+r;
                        uint32_t phase=(j>>1)&1;
                        pipe_kv.wait_full(st,phase);
                        convert_words(reinterpret_cast<uint32_t*>(shared.smem_k.data()+st*T::kStageElemsK),T::kStageElemsK/2,1.f);
                        if((tid&31)==0)shared.k_converted[st].arrive();
                    }
                }
            }
        }
'''
 s=s.replace(marker,conversion+marker)
 # The SAFE reset barrier is reached from distinct role branches. The ID
 # supplied to the user API is relative to CUTLASS' reserved barrier range.
 s=s.replace('constexpr uint32_t kBarReinit = uint32_t(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier) + 4;', 'constexpr uint32_t kBarReinit = 4;')
 s=s.replace('cutlass::arch::NamedBarrier::sync(T::kNumThreads, kBarReinit);','cutlass::arch::NamedBarrier(T::kNumThreads,kBarReinit).arrive_and_wait_unaligned();')
 return s
