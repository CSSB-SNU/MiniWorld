"""Experimental N40 PV: append eight ones columns to packed V, share the MMA.

Packing is included in the timed wrapper. No serving code imports this file.
"""
def transform(name,s):
 if name=='triattn_m1.py':
  pos=s.index('    ext.fwd(')
  s=s[:pos]+'''    assert S % 128 == 0
    packed_v = torch.ones((B, N, H, S // 8, 5, 8, 8), dtype=v.dtype, device=v.device)
    packed_v[..., :4, :, :].copy_(v.reshape(B, N, H, S // 8, 8, 4, 8).transpose(-2, -3))
    qkv[2] = packed_v.view(B, N, H, S, 40)
'''+s[pos:]
 if name=='triattn_m1_sm90.cuh':
  s=s.replace('SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>', 'SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>')
  a=s.index('    using SmemLayoutV =');b=s.index('    // all-ones B tile',a)
  s=s[:a]+'''    using SmemLayoutVt = decltype(tile_to_shape(GMMA::Layout_MN_INTER_Atom<Element>{}, Shape<_40, Int<kBlockN>, Int<kStagesKV>>{}));
    using SmemLayoutV = SmemLayoutVt;
    using SmemLayoutVTma = Layout<Shape<_64,_5,Int<kBlockN/8>>,Stride<_1,_64,_320>>;
    using ShapeVP = Shape<_64,_5,int,int>;
    using StrideVP = Stride<_1,_64,_320,int64_t>;
    static constexpr int kStageElemsV = kBlockN * 40;
'''+s[b:]
  s=s.replace('using TMA_V = TMA_K;', '''using TMA_V = decltype(make_tma_copy(SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeVP{}, StrideVP{}), SmemLayoutVTma{}, Shape<_64,_5,Int<kBlockN/8>>{}, _1{}));''')
  s=s.replace('kBytesV = kBytesK;', 'kBytesV = kBlockN * 40 * sizeof(Element);')
  s=s.replace('Tensor sV = make_tensor(make_smem_ptr(shared.smem_v.data()), typename T::SmemLayoutV{});', '''Tensor sV = make_tensor(make_smem_ptr(shared.smem_v.data()), Layout<Shape<_64,_5,Int<kBlockN/8>,Int<T::kStagesKV>>,Stride<_1,_64,_320,Int<T::kStageElemsV>>>{});''')
  s=s.replace('Tensor mV = params.tma_v.get_tma_tensor(params.shape_qk)(_, _, h, _, b);', '''Tensor mV = params.tma_v.get_tma_tensor(make_shape(_64{},_5{},params.S/8,params.N*params.H*get<4>(params.shape_qk)));''')
  a=s.index('            Tensor gV = local_tile(');b=s.index('\n',a)
  s=s[:a]+'''            Tensor gV = local_tile(domain_offset(make_coord(_0{},_0{},key0/8,_0{}),mV),Shape<_64,_5,Int<kBlockN/8>>{},make_coord(_0{},_0{},_,_));'''+s[b:]
  s=s.replace('group_modes<0, 3>(block_tma_v.partition_S(gV))','group_modes<0, 4>(block_tma_v.partition_S(gV))')
  s=s.replace('group_modes<0, 3>(block_tma_v.partition_D(sV))','group_modes<0, 4>(block_tma_v.partition_D(sV))')
  s=s.replace('tVgV(_, j, i)', 'tVgV(_, j, (b * params.N + i) * params.H + h)')
  s=s.replace('make_shape(Int<kHeadDim>{}, Int<CW>{}), make_coord(_0{}, _)', 'make_shape(_40{}, Int<CW>{}), make_coord(_0{}, _)')
  s=s.replace('Tensor cO = make_identity_tensor(make_shape(_64{}, Int<kHeadDim>{}));','Tensor cO = make_identity_tensor(make_shape(_64{}, _40{}));')
  s=s.replace('partition_fragment_C(tiled_mma_pv, make_shape(_64{}, Int<kHeadDim>{}))','partition_fragment_C(tiled_mma_pv, make_shape(_64{}, _40{}))')
  s=s.replace('decltype(size(AccO{}))::value == 16','decltype(size(AccO{}))::value == 20')
  s=s.replace('    AccL acc_l[2];', '''    auto l_view0 = make_tensor(acc_o[0].data()+16, AccL{}.layout());
    decltype(l_view0) acc_l[2] = {l_view0, make_tensor(acc_o[1].data()+16, AccL{}.layout())};''')
  # The final four registers of N40 are the same row/column layout as N8.
  # Preserve the existing rescale, validation, and output-store loops.
  import re
  s=re.sub(r'(?m)^ *#pragma unroll\n *for \(int kb = 0; kb < kNKB; \+\+kb\) \{ cute::gemm\(tiled_mma_l,[^\n]+\}[^\n]*\n', '', s)
  assert 'cute::gemm(tiled_mma_l,' not in s
 if name=='launch_m1.cuh':
  s=s.replace('Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.v.data_ptr())), shape_qk, stride_of(a.v));', '''Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.v.data_ptr())), make_shape(_64{},_5{},S/8,B*N*H), typename T::StrideVP{_1{},_64{},_320{},int64_t(S)*40});''')
  s=s.replace('take<0, 2>(typename T::SmemLayoutV{}), make_shape(Int<T::kBlockN>{}, Int<T::kHeadDim>{})', 'typename T::SmemLayoutVTma{}, make_shape(_64{},_5{},Int<T::kBlockN/8>{})')
 if name=='m1_binding.cu':
  # Uniform rows still use the original D32 output; only V physical indexing changes.
  s=s.replace('int64_t(key) * vs + d', 'int64_t(key / 8) * 320 + (d / 8) * 64 + (key % 8) * 8 + d % 8')
 return s.replace('_320', 'Int<320>')
