"""Initialize QK with ScaleOut::Zero and add bias only after QK retires.

This is a numerical experiment: it changes FP32 addition order, not Q/K/V
precision or EX2. It removes bias LDS from the next QK accumulator's live
range. The producer receives each bias release after the actual late loads,
through an explicit dependency covering all four LDS vectors and all lanes.
"""
from producer_three_score import transform as three_score


def transform(s):
    s=three_score(s,helper=False,full_q=True)
    a=s.index('    auto init_chunk =');b=s.index('    auto issue_qk =',a)
    block=s[a:b]
    header_end=block.index('\n')+1
    assert block.endswith('    };\n')
    block=block[:header_end]+'        if constexpr(!kFast){\n'+block[header_end:-len('    };\n')]+'        }\n    };\n'
    s=s[:a]+block+s[b:]
    a=s.index('    auto issue_qk =');b=s.index('    auto issue_pv =',a)
    block=s[a:b]
    old='        warpgroup_fence_operand(acc);'
    assert block.count(old)==1
    block=block.replace(old,'        if constexpr(!kFast)warpgroup_fence_operand(acc);')
    old='''        if constexpr(kFast) {
            tiled_mma_qkr.accumulate_ = GMMA::ScaleOut::One;'''
    assert block.count(old)==1
    block=block.replace(old,old.replace('ScaleOut::One','ScaleOut::Zero'))
    old='            cute::gemm(tiled_mma_qkr, tQr(_,_,0), tK(_,_,0,c,st), acc);'
    assert block.count(old)==1
    block=block.replace(old,old+'\n            tiled_mma_qkr.accumulate_ = GMMA::ScaleOut::One;')
    s=s[:a]+block+s[b:]
    marker='        prep_chunk(acc, hh, kc, words, seedc);'
    assert s.count(marker)==1
    s=s.replace(marker,'''        if constexpr(kFast){
            int c=kc&3;
            if(hh==0)pipe_b.wait_full(c,(kc/4)&1);
            #pragma unroll
            for(int u=0;u<4;++u){
                float4 value=*reinterpret_cast<float4 const*>(bias_thread+c*T::kSlotElems+(hh*4+u)*512);
                acc(4*u)=__fadd_rn(acc(4*u),value.x);
                acc(4*u+1)=__fadd_rn(acc(4*u+1),value.y);
                acc(4*u+2)=__fadd_rn(acc(4*u+2),value.z);
                acc(4*u+3)=__fadd_rn(acc(4*u+3),value.w);
            }
            uint32_t dependency=__float_as_uint(acc(0))|__float_as_uint(acc(4))|
                                __float_as_uint(acc(8))|__float_as_uint(acc(12));
            dependency=__reduce_or_sync(0xffffffffu,dependency);
            uint32_t ordered_zero=dependency&uint32_t(params.zero);
            asm volatile("":::"memory");__syncwarp();
            pipe_b.release(2*c+hh+ordered_zero,warp_leader);
        }
'''+marker)
    a=s.index('    if constexpr(kFast){\n        OpK const helper_k=')
    b=s.index('    }else{\n    if constexpr (T::kW) {',a)
    block=s[a:b]
    for n in range(4):
        old='bias_release(%d,0u);'%n
        assert block.count(old)==1
        block=block.replace(old,'')
    for c in range(2):
        old='pipe_b.wait_full(%d,0);'%c
        assert block.count(old)==1
        block=block.replace(old,'')
    old='''            if constexpr((future&1)==0){
                if(future<8*n_w)pipe_b.wait_full((future/2)&3,(future/8)&1);
            }
'''
    assert block.count(old)==1
    block=block.replace(old,'')
    old='            if(future<8*n_w)bias_release(future&7,0u);\n'
    assert block.count(old)==1
    block=block.replace(old,'')
    return s[:a]+block+s[b:]
