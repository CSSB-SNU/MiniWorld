"""Use the same BF16 P for tensor-core numerator and denominator accumulation."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'hot6r/fused.cu').read_text()
s=s.replace('  union alignas(1024) Scratch {', '''  using Sum=decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K,GMMA::Major::K>{}));
  using Ones=decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<Element>{},Shape<_8,_16>{}));
  union alignas(1024) Scratch {''')
s=s.replace('    int unsafe[Consumers][4];',
            '    int unsafe[Consumers][4];\n    array_aligned<Element,128,128> ones;')
pos=s.index('  __syncthreads();')
s=s[:pos]+'''  if(tid<128)s.ones[tid]=Element(1.f);
  cutlass::arch::fence_view_async_shared();
'''+s[pos:]
s=s.replace('      constexpr bool Safe=decltype(safe_tag)::value;', '''      constexpr bool Safe=decltype(safe_tag)::value;
      typename C::Sum sum_mma;
      auto sums=partition_fragment_C(sum_mma,Shape<_64,_8>{});clear(sums);
      auto ones=make_tensor(make_smem_ptr(s.ones.data()),typename C::Ones{});
      auto ob=sum_mma.get_slice(lane).partition_fragment_B(ones);''')
s=s.replace('if constexpr(Safe) ls[mi]+=score(x); else ls[mi]+=float(Element(score(x)));',
            'if constexpr(Safe) ls[mi]+=score(x);')
s=s.replace('          l[mi]=l[mi]*alpha[mi]+ls[mi];',
            '          if constexpr(Safe) l[mi]=l[mi]*alpha[mi]+ls[mi];')
s=s.replace('        flash::gemm<false,0>(pmma,pr,vb,out);', '''        if constexpr(Safe) flash::gemm<false,0>(pmma,pr,vb,out);
        else {
          warpgroup_fence_operand(pr);warpgroup_fence_operand(out);warpgroup_fence_operand(sums);
          warpgroup_arrive();
          pmma.accumulate_=GMMA::ScaleOut::One;
          sum_mma.accumulate_=GMMA::ScaleOut::One;
          #pragma unroll
          for(int kk=0;kk<size<2>(pr);++kk) {
            cute::gemm(pmma,pr(_,_,kk),vb(_,_,kk),out);
            cute::gemm(sum_mma,pr(_,_,kk),ob(_,_,0),sums);
          }
          warpgroup_commit_batch();warpgroup_wait<0>();
          warpgroup_fence_operand(pr);warpgroup_fence_operand(out);warpgroup_fence_operand(sums);
        }''')
start=s.index('      if constexpr(!Safe) {\n        #pragma unroll')
end=s.index('      bias_epoch+=',start)
s=s[:start]+'''      if constexpr(!Safe) { l[0]=sums(0);l[1]=sums(2); }
'''+s[end:]
path=r/'hot6t'
assert not (path/'build-ready.json').exists()
path.mkdir(exist_ok=True)
(path/'fused.cu').write_text(s)
