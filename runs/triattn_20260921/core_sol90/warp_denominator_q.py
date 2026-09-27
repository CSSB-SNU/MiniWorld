"""Two useful denominator registers per half via synchronous SM80 BF16 MMA.

Full Q and all QK/PV scheduling derive from scalar_denominator_q. This replaces
the per-element scalar sums by m16n8k16 on each warp, feeding the SAME A
fragment as WGMMA and BF16 ones. Duplicate output columns are temporary only.
MMA reads P before its new asynchronous PV issue; old use retired at wait2.
"""
from scalar_denominator_q import transform as scalar_transform

def transform(s,split_k=False):
    s=scalar_transform(s)
    s=s.replace('#include <cuda_bf16.h>',
                '#include <cuda_bf16.h>\n#include <cute/arch/mma_sm80.hpp>',1)
    a=s.index('    auto sum_probability =');b=s.index('    auto issue_pv =',a)
    s=s[:a]+'''    auto sum_probability = [&](auto const& prob,auto hc) __attribute__((always_inline)) {
        if constexpr(kFast){
            constexpr int hh=decltype(hc)::value;
            auto lr=make_tensor(acc_l[hh].data(),flash::convert_layout_acc_rowcol(acc_l[hh].layout()));
            #pragma unroll
            for(int kb=0;kb<2;++kb){
                auto words=recast<uint32_t>(prob(_,_,kb));
                static_assert(decltype(size(words))::value==4);
                float duplicate0,duplicate1;
                SM80_16x8x16_F32BF16BF16F32_TN::fma(
                    lr(0,0),duplicate0,lr(1,0),duplicate1,
                    words(0),words(1),words(2),words(3),
                    0x3f803f80u,0x3f803f80u,
                    lr(0,0),lr(0,0),lr(1,0),lr(1,0));
            }
        }
    };
    auto total_denominator = [&](float value) __attribute__((always_inline)) {
        return value; // Warp MMA already includes the four column owners.
    };
'''+s[b:]
    if split_k:
        a=s.index('    auto sum_probability =');b=s.index('    auto total_denominator =',a)
        block=s[a:b]
        block=block.replace('auto const& prob,auto hc)', 'auto const& prob,auto hc,auto kbc)')
        old='''            #pragma unroll
            for(int kb=0;kb<2;++kb){'''
        assert block.count(old)==1
        block=block.replace(old,'''            {
                constexpr int kb=decltype(kbc)::value;''')
        s=s[:a]+block+s[b:]
        a=s.index('    auto issue_pv =');b=s.index('    auto issue_pv_prefenced =',a)
        block=s[a:b];assert block.count('sum_probability(tP,hc);')==1
        block=block.replace('sum_probability(tP,hc);','sum_probability(tP,hc,_0{});sum_probability(tP,hc,_1{});')
        s=s[:a]+block+s[b:]
        assert s.count('sum_probability(tP,hc);')==1
        s=s.replace('sum_probability(tP,hc);','sum_probability(tP,hc,_1{});')
        a=s.index('    auto body =');b=s.index('    // drained state:',a)
        block=s[a:b];assert block.count('        ring_wait();')==1
        block=block.replace('        ring_wait();',
'''        // Two independent tensor instructions around E avoid immediately
        // consuming the first denominator MMA result in the second MMA.
        if constexpr(kFast)sum_probability(PCb[pb],Int<hp>{},_0{});
        ring_wait();''')
        s=s[:a]+block+s[b:]
    return s
