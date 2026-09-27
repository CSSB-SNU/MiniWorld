"""Lossless raw BF16 bias transport with exact FP32 SAFE storage retained.

Each staged key-column independently records whether all live raw bias bits
are BF16-representable. Nonrepresentable inputs force original SAFE output.
The inverse scale is the same host-rounded float used by the original stage.
SAFE can round differently from the original hot path on those inputs; this
is not a claim of bitwise equivalence for arbitrary FP32 bias.
"""


def transform(name, s, full_q=False, decode_at_init=False):
    if name == 'm1_binding.cu':
        old = '    float* out = dst + ((int64_t(bh) * nq + qt) * nkc + blockIdx.x) * 4096;'
        assert s.count(old) == 1
        s = s.replace(old, '''    float* tile=dst+(int64_t(bh)*nq+qt)*nkc*6148;
    float* out=tile+blockIdx.x*4096;
    uint32_t* raw_out=reinterpret_cast<uint32_t*>(tile+nkc*4096+blockIdx.x*2048);
    bool raw_inexact=false;''')
        old = '''        if (q < S && live) { val = to_f32(base[q * sq + key * sk]) * inv_scale; }
        out[idx] = val;'''
        assert s.count(old) == 1
        s = s.replace(old, '''        float raw=val;
        if(q<S && live){raw=to_f32(base[q*sq+key*sk]);val=raw*inv_scale;}
        out[idx]=val;
        uint32_t bits=__float_as_uint(raw);
        raw_inexact|=(bits&0xffffu)!=0u;
        uint32_t neighbor=__shfl_xor_sync(0xffffffffu,bits,1);
        if((e&1)==0){
            // Two neighboring u groups form one LDS.128 for each thread.
            int pos=hh*1024+(u/2)*512+t*4+(u%2)*2+e/2;
            raw_out[pos]=(bits>>16)|(neighbor&0xffff0000u);
        }''')
        old = '        out[idx]=val;'
        assert s.count(old) == 1
        start = s.index('__global__ void stage_bias_m1_kernel')
        end = s.index('\ntorch::Tensor stage_bias_m1(', start)
        block = s[start:end]
        tail = '    }\n}\n'
        assert block.endswith(tail)
        block = block[:-len(tail)] + '''    }
    int inexact=__syncthreads_or(raw_inexact);
    if(threadIdx.x==0)reinterpret_cast<uint32_t*>(tile+nkc*6144)[blockIdx.x]=uint32_t(inexact);
}
'''
        s = s[:start]+block+s[end:]
        old = 'nq, W4, 4096}, bias.options()'
        assert s.count(old) == 1
        s = s.replace(old, 'nq, W4, 6148}, bias.options()')
    if name == 'triattn_m1_sm90.cuh':
        old = '    static constexpr int kSlotElems = 2 * 4 * 128 * 4, kHalfElems = kSlotElems / 2, kHalves = 2 * kSlotsB;'
        assert s.count(old) == 1
        s = s.replace(old, '''    static constexpr bool kRawBias=(kFlags_&1073741824)!=0;
    static constexpr int kBiasRows=kRawBias?4:8;
    static constexpr int kSlotElems=2*256*kBiasRows,kHalfElems=kSlotElems/2,kHalves=2*kSlotsB;''')
        s = s.replace('Layout<Shape<_256, _8>, Stride<_1, _256>>', 'Layout<Shape<_256, Int<kBiasRows>>, Stride<_1, _256>>')
        s = s.replace('SmemLayoutBiasHalf{}, make_shape(_256{}, _8{})', 'SmemLayoutBiasHalf{}, make_shape(_256{}, Int<kBiasRows>{})')
        start = s.index('    struct Params {')
        stop = s.index('\n    };',start)
        s = s[:stop]+'\n        float raw_inv_scale;\n        uint32_t const* raw_bad;'+s[stop:]
        old = '    uint32_t const* maskrow = irregular ?'
        assert s.count(old) == 1
        s = s.replace(old, '''    if constexpr(kFast){
        // There are exactly28 staged columns at S768; all flags were
        // written by the preceding stage kernel, including padding.
        static_assert(4*((768+127)/128+1)<=32);
        uint32_t invalid=lane<28?params.raw_bad[(int64_t(bh)*params.n_qtiles+qtile)*28*6148+lane]:0u;
        bad|=__any_sync(0xffffffffu,invalid!=0u);
    }
'''+old)
        start = s.index('    auto init_chunk =')
        stop = s.index('    auto issue_qk =',start)
        block = s[start:stop]
        old = '''        } else {
            #pragma unroll'''
        assert block.count(old) == 1
        block = block.replace(old, '''        }else if constexpr(kFast){
            #pragma unroll
            for(int u=0;u<2;++u){
                float4 packed=*reinterpret_cast<float4 const*>(bias_thread+c*T::kSlotElems+hh*1024+u*512);
                acc(4*u)=packed.x;acc(4*u+1)=packed.y;acc(4*u+2)=packed.z;acc(4*u+3)=packed.w;
            }
            #pragma unroll
            for(int z=8;z<16;++z)acc(z)=0.f;
        } else {
            #pragma unroll''')
        s=s[:start]+block+s[stop:]
        start=s.index('    auto issue_qk =')
        stop=s.index('    auto issue_pv =',start)
        block=s[start:stop]
        old='        warpgroup_fence_operand(acc);'
        assert block.count(old)==1
        block=block.replace(old,'''        if constexpr(kFast){
            #pragma unroll
            for(int pair=7;pair>=0;--pair){
                uint32_t bits=__float_as_uint(acc(pair));
                float lo=__uint_as_float(bits<<16),hi=__uint_as_float(bits&0xffff0000u);
                asm("mul.rn.ftz.f32 %0,%1,%2;":"=f"(acc(2*pair)):"f"(lo),"f"(params.raw_inv_scale));
                asm("mul.rn.ftz.f32 %0,%1,%2;":"=f"(acc(2*pair+1)):"f"(hi),"f"(params.raw_inv_scale));
            }
        }
'''+old)
        if full_q:
            old='            cute::gemm(tiled_mma_qk, tQ(_,_,1,hh,cwg), tK(_,_,1,c,st), acc);'
            assert block.count(old)==1
            block=block.replace(old,'            cute::gemm(tiled_mma_qkr, tQr(_,_,1), tK(_,_,1,c,st), acc);')
        s=s[:start]+block+s[stop:]
        if decode_at_init:
            # Move the raw->scaled conversion under the previous body's
            # outstanding QK/PV instead of directly before the next QK.
            start=s.index('    auto issue_qk =')
            a=s.index('        if constexpr(kFast){',start)
            b=s.index('        warpgroup_fence_operand(acc);',a)
            conversion=s[a:b]
            assert 'for(int pair=7;' in conversion
            s=s[:a]+s[b:]
            start=s.index('    auto init_chunk =')
            end=s.index('    auto issue_qk =',start)
            block=s[start:end]
            tail='    };\n'
            assert block.endswith(tail)
            block=block[:-len(tail)]+conversion+tail
            s=s[:start]+block+s[end:]
        if full_q:
            old='''        if constexpr(kFast) {
            auto a0=tQr0(_,_,Int<0>{});auto a1=tQr1(_,_,Int<0>{});
            warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);
        } else {warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);}'''
            assert s.count(old)==2
            s=s.replace(old,'        warpgroup_fence_operand(tQr0);warpgroup_fence_operand(tQr1);')
    if name == 'launch_m1.cuh':
        s=s.replace('a.bias.size(3) == T::kSlotElems','a.bias.size(3) == 6148')
        s=s.replace('make_shape(256, 8, 2 * n_kcol','make_shape(256, T::kBiasRows, 2 * n_kcol')
        s=s.replace('int64_t(T::kSlotElems) * n_kcol','int64_t(6148) * n_kcol')
        s=s.replace('make_gmem_ptr(a.bias.data_ptr<float>())','make_gmem_ptr(a.bias.data_ptr<float>()+(T::kRawBias?4096*n_kcol:0))')
        s=s.replace('SmemLayoutBiasHalf{}, make_shape(_256{}, _8{})','SmemLayoutBiasHalf{}, make_shape(_256{}, Int<T::kBiasRows>{})')
        old='n_kcol, a.rowkc0, a.rowkc1};'
        assert s.count(old)==1
        s=s.replace(old,'n_kcol, a.rowkc0, a.rowkc1, float(1.0/a.scale), reinterpret_cast<uint32_t const*>(a.bias.data_ptr<float>()+6144*n_kcol)};')
    return s
