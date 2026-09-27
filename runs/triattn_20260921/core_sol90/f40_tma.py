"""N40 PV experiment: TMA transposes D8 groups, immutable shared ones tail.

No global V padding/copy. The first prototype accepts B=1; arbitrary S/N/H
strides remain represented by the five-dimensional TMA descriptor.
"""
import re

def transform(name,s):
 if name=='triattn_m1_sm90.cuh':
  s=s.replace('SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>',
              'SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>')
  a=s.index('    using SmemLayoutV =');b=s.index('    // all-ones B tile',a)
  s=s[:a]+'''    // Physical order [D/8][key][D%8]. The fifth D/8 block is
    // immutable ones, initialized once and never touched by V TMA.
    using SmemLayoutVt = Layout<Shape<Shape<_8,_5>,Int<kBlockN>,Int<kStagesKV>>,
                               Stride<Stride<_1,Int<kBlockN*8>>,_8,Int<kBlockN*40>>>;
    using SmemLayoutV = SmemLayoutVt;
    using SmemLayoutVTma = Layout<Shape<_8,Int<kBlockN>,_4>,Stride<_1,_8,Int<kBlockN*8>>>;
    using SmemLayoutVTmaStages = Layout<Shape<_8,Int<kBlockN>,_4,Int<kStagesKV>>,Stride<_1,_8,Int<kBlockN*8>,Int<kBlockN*40>>>;
    using ShapeVP = Shape<_8,int32_t,_4,int32_t,int32_t>;
    using StrideVP = Stride<_1,int64_t,_8,int64_t,int64_t>;
    static constexpr int kStageElemsV = kBlockN * 40;
'''+s[b:]
  s=s.replace('using TMA_V = TMA_K;', '''using TMA_V = decltype(make_tma_copy(SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeVP{}, StrideVP{}), SmemLayoutVTma{}, Shape<_8,Int<kBlockN>,_4>{}, _1{}));''')
  # kBytesV remains the original 128*32*2, only copied values count.
  old='        cutlass::arch::fence_view_async_shared();'
  assert s.count(old)==1
  s=s.replace(old,'''        for (int ix=tid-128;ix<T::kStagesKV*kBlockN*8;ix+=T::kNumMmaThreads) {
            int st=ix/(kBlockN*8),local=ix%(kBlockN*8);
            shared.smem_v[st*T::kStageElemsV+kBlockN*32+local]=Element(1.f);
        }
'''+old)
  s=s.replace('Tensor sV = make_tensor(make_smem_ptr(shared.smem_v.data()), typename T::SmemLayoutV{});',
              'Tensor sV = make_tensor(make_smem_ptr(shared.smem_v.data()), typename T::SmemLayoutVTmaStages{});')
  s=s.replace('Tensor mV = params.tma_v.get_tma_tensor(params.shape_qk)(_, _, h, _, b);',
              'Tensor mV = params.tma_v.get_tma_tensor(make_shape(_8{},params.S,_4{},params.H,params.N))(_,_,_,h,_);')
  a=s.index('            Tensor gV = local_tile(');b=s.index('\n',a)
  s=s[:a]+'''            Tensor gV = local_tile(domain_offset(make_coord(_0{},key0,_0{},_0{}),mV),Shape<_8,Int<kBlockN>,_4>{},make_coord(_0{},_,_0{},_));'''+s[b:]
  s=s.replace('group_modes<0, 3>(block_tma_v.partition_S(gV))','group_modes<0, 4>(block_tma_v.partition_S(gV))')
  s=s.replace('group_modes<0, 3>(block_tma_v.partition_D(sV))','group_modes<0, 4>(block_tma_v.partition_D(sV))')
  s=s.replace('make_shape(Int<kHeadDim>{}, Int<CW>{}), make_coord(_0{}, _)','make_shape(_40{}, Int<CW>{}), make_coord(_0{}, _)')
  s=s.replace('Tensor cO = make_identity_tensor(make_shape(_64{}, Int<kHeadDim>{}));','Tensor cO = make_identity_tensor(make_shape(_64{}, _40{}));')
  s=s.replace('partition_fragment_C(tiled_mma_pv, make_shape(_64{}, Int<kHeadDim>{}))','partition_fragment_C(tiled_mma_pv, make_shape(_64{}, _40{}))')
  s=s.replace('decltype(size(AccO{}))::value == 16','decltype(size(AccO{}))::value == 20')
  s=s.replace('    AccL acc_l[2];','''    auto l_view0 = make_tensor(acc_o[0].data()+16, AccL{}.layout());
    decltype(l_view0) acc_l[2] = {l_view0, make_tensor(acc_o[1].data()+16, AccL{}.layout())};''')
  s=re.sub(r'(?m)^ *#pragma unroll\n *for \(int kb = 0; kb < kNKB; \+\+kb\) \{ cute::gemm\(tiled_mma_l,[^\n]+\}[^\n]*\n','',s)
  assert 'cute::gemm(tiled_mma_l,' not in s
 if name=='launch_m1.cuh':
  s=s.replace('    TORCH_CHECK(D == T::kHeadDim,','    TORCH_CHECK(B==1,"f40 TMA prototype requires B=1");\n    TORCH_CHECK(D == T::kHeadDim,')
  s=s.replace('Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.v.data_ptr())), shape_qk, stride_of(a.v));',
              'Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.v.data_ptr())), make_shape(_8{},S,_4{},H,N), typename T::StrideVP{_1{},a.v.stride(3),_8{},a.v.stride(2),a.v.stride(1)});')
  s=s.replace('take<0, 2>(typename T::SmemLayoutV{}), make_shape(Int<T::kBlockN>{}, Int<T::kHeadDim>{})',
              'typename T::SmemLayoutVTma{}, make_shape(_8{},Int<T::kBlockN>{},_4{})')
 return s

def sw32(name,s):
 """Use D16 groups and 32-byte swizzle; pad N40 to a physical D48 tile."""
 s=transform(name,s)
 if name=='triattn_m1_sm90.cuh':
  a=s.index('    using SmemLayoutVt =');b=s.index('    // all-ones B tile',a)
  s=s[:a]+'''    using SmemLayoutVt = ComposedLayout<Swizzle<1,4,3>,smem_ptr_flag_bits<16>,
       Layout<Shape<Shape<_16,_4>,Int<kBlockN>,Int<kStagesKV>>,
              Stride<Stride<_1,Int<kBlockN*16>>,_16,Int<kBlockN*48>>>>;
    using SmemLayoutV = SmemLayoutVt;
    using SmemLayoutVTma = ComposedLayout<Swizzle<1,4,3>,smem_ptr_flag_bits<16>,
       Layout<Shape<_16,Int<kBlockN>,_2>,Stride<_1,_16,Int<kBlockN*16>>>>;
    using SmemLayoutVTmaStages = ComposedLayout<Swizzle<1,4,3>,smem_ptr_flag_bits<16>,
       Layout<Shape<_16,Int<kBlockN>,_2,Int<kStagesKV>>,Stride<_1,_16,Int<kBlockN*16>,Int<kBlockN*48>>>>;
    using ShapeVP = Shape<_16,int32_t,_2,int32_t,int32_t>;
    using StrideVP = Stride<_1,int64_t,_16,int64_t,int64_t>;
    static constexpr int kStageElemsV = kBlockN * 48;
'''+s[b:]
  old='return wg_mma_pv.partition_fragment_B(local_tile(sVz, make_shape(_40{}, Int<CW>{}), make_coord(_0{}, _)));'
  assert s.count(old)==1
  s=s.replace(old,'auto descmma=make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::MN>{}); return descmma.get_slice(0).partition_fragment_B(local_tile(sVz, make_shape(_32{}, Int<CW>{}), make_coord(_0{}, _)));')
  s=s.replace('Shape<_8,Int<kBlockN>,_4>','Shape<_16,Int<kBlockN>,_2>')
  s=s.replace('make_shape(_8{},params.S,_4{},params.H,params.N)','make_shape(_16{},params.S,_2{},params.H,params.N)')
  s=s.replace('ix<T::kStagesKV*kBlockN*8','ix<T::kStagesKV*kBlockN*16')
  s=s.replace('ix/(kBlockN*8),local=ix%(kBlockN*8)','ix/(kBlockN*16),local=ix%(kBlockN*16)')
 if name=='launch_m1.cuh':
  s=s.replace('make_shape(_8{},S,_4{},H,N)','make_shape(_16{},S,_2{},H,N)')
  s=s.replace('a.v.stride(3),_8{}','a.v.stride(3),_16{}')
  s=s.replace('make_shape(_8{},Int<T::kBlockN>{},_4{})','make_shape(_16{},Int<T::kBlockN>{},_2{})')
 return s
