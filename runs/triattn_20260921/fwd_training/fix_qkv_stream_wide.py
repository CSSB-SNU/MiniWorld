"""Load N64 producer weights as four M32/K64 boxes in the actual N64 layout."""
from pathlib import Path
r=Path(__file__).resolve().parent
for c in (2,4,6):
    s=(r/('qkv_stream_wide%d/fused.cu'%c)).read_text()
    mark='  using TB='
    s=s.replace(mark,'''  using WKL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_32,_64>{}));
  using TWKV=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),WS{},WStride{}),WKL{},Shape<_32,_64>{},_1{}));
'''+mark,1)
    s=s.replace('TW wq,wk,wv,wg;', 'TW wq,wg;TWKV wk,wv;')
    b=s.index('      for(int which=0;which<2;++which) {',s.index('    auto load_z='))
    e=s.index('\n    }\n    s.wfull.wait(1);',b)
    s=s[:b]+'''      auto ws=make_tensor(make_smem_ptr(s.scratch.attn.w[0].data()),typename C::ZL{});
      #pragma unroll
      for(int which=0;which<2;++which) {
        auto const& wt=which==0?p.wk:p.wv;
        auto wg=wt.get_tma_tensor(make_shape(_128{},_128{}));
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk) {
          auto src=local_tile(wg,Shape<_32,_64>{},make_coord(h,chunk));
          auto dst=local_tile(ws,Shape<_32,_64>{},make_coord(which,chunk));
          tma_load(wt,src,dst,s.wfull);
        }
      }'''+s[e:]
    mark='  auto bg=make_tensor(make_gmem_ptr((Element const*)b.data_ptr())'
    assert mark in s
    s=s.replace(mark,'''  auto weightkv=[&](torch::Tensor const& w) {
    auto g=make_tensor(make_gmem_ptr((Element const*)w.data_ptr()),typename C::WS{},typename C::WStride{});
    return make_tma_copy(SM90_TMA_LOAD{},g,typename C::WKL{},Shape<_32,_64>{},_1{});
  };
'''+mark,1)
    s=s.replace('p{tz,weight(wq),weight(wk),weight(wv),weight(wg),tb,',
                'p{tz,weight(wq),weight(wg),weightkv(wk),weightkv(wv),tb,')
    s=s.replace('// Two adjacent N32 head weights are one swizzled N64 weight tile.',
                '// Four M32/K64 TMA boxes populate the N64/K128 swizzled weight tile.')
    d=r/('qkv_stream_wide%db'%c);assert not (d/'build-ready.json').exists()
    d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
