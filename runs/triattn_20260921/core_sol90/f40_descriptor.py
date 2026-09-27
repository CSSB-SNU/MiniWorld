"""Fuse N40 PV with original V TMA: the extra N group points at shared ones.

Major-MN SW64 descriptors have independent N32-group and K8-group strides.
The virtual second N32 group lives in a shared constant tile after all V
stages. Descriptor adjustment accounts for the selected ring stage. No V
global format, TMA shape, or externally visible strides change.
"""
import re

def transform(name,s):
 if name!='triattn_m1_sm90.cuh':return s
 s=s.replace('SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>',
             'SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>')
 s=s.replace('cute::array_aligned<Element, cute::cosize_v<SmemLayoutOnes>, 128> smem_ones;',
             'cute::array_aligned<Element, kBlockN*kHeadDim, 1024> smem_ones;')
 s=s.replace('int(cute::cosize_v<typename T::SmemLayoutOnes>)','T::kBlockN*T::kHeadDim')
 old='return wg_mma_pv.partition_fragment_B(local_tile(sVz, make_shape(Int<kHeadDim>{}, Int<CW>{}), make_coord(_0{}, _)));'
 assert s.count(old)==1
 s=s.replace(old,'''auto vmma=make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{});
        auto operand=vmma.get_slice(0).partition_fragment_B(local_tile(sVz, make_shape(Int<kHeadDim>{}, Int<CW>{}), make_coord(_0{}, _)));
        auto desc=operand.data().desc_;
        uint32_t vaddr=cute::cast_smem_ptr_to_uint(shared.smem_v.data()+cwg*T::kStageElemsV+zoff);
        uint32_t oneaddr=cute::cast_smem_ptr_to_uint(shared.smem_ones.data());
        desc.bitfield.leading_byte_offset_=(oneaddr-vaddr)>>4;
        return make_tensor(GMMA::DescriptorIterator{desc},operand.layout());''')
 pos=s.index('    using OpK =')
 s=s[:pos]+'''    auto fused_v_operand = [&](auto const& operand, auto stage) __attribute__((always_inline)) {
        constexpr int st=decltype(stage)::value;
        auto desc=operand.data().desc_;
        // DescriptorIterator advances the V start address while leaving the
        // N32-group stride fixed. Keep the ones tile fixed across KV stages.
        desc.reg32_[0] -= uint32_t(st*T::kStageElemsV/8)<<16;
        return make_tensor(GMMA::DescriptorIterator{desc},operand.layout());
    };
'''+s[pos:]
 s=s.replace('tV(_, _, kb, c, st), acc_o[hh]', 'fused_v_operand(tV(_, _, kb, c, st),Int<st>{}), acc_o[hh]')
 s=s.replace('tV(_, _, kb, cp, stp), acc_o[hp]', 'fused_v_operand(tV(_, _, kb, cp, stp),Int<stp>{}), acc_o[hp]')
 s=s.replace('Tensor cO = make_identity_tensor(make_shape(_64{}, Int<kHeadDim>{}));','Tensor cO = make_identity_tensor(make_shape(_64{}, _40{}));')
 s=s.replace('partition_fragment_C(tiled_mma_pv, make_shape(_64{}, Int<kHeadDim>{}))','partition_fragment_C(tiled_mma_pv, make_shape(_64{}, _40{}))')
 s=s.replace('decltype(size(AccO{}))::value == 16','decltype(size(AccO{}))::value == 20')
 s=s.replace('    AccL acc_l[2];','''    auto l_view0=make_tensor(acc_o[0].data()+16,AccL{}.layout());
    decltype(l_view0) acc_l[2]={l_view0,make_tensor(acc_o[1].data()+16,AccL{}.layout())};''')
 s=re.sub(r'(?m)^ *#pragma unroll\n *for \(int kb = 0; kb < kNKB; \+\+kb\) \{ cute::gemm\(tiled_mma_l,[^\n]+\}[^\n]*\n','',s)
 assert 'cute::gemm(tiled_mma_l,' not in s
 return s
