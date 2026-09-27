"""One Z read and N128 WGMMA produces K,V,Q,gate together inside each CTA."""
from pathlib import Path
r = Path(__file__).resolve().parent
s = (r/'splitq4/fused.cu').read_text()
s = s.replace('using WPL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));',
              'using WPL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_128,_128>{}));')
s = s.replace('using Proj=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_64,_64>>()));',
              'using Proj=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_128,_64>>()));')
s = s.replace('array_aligned<Element,8192,1024> w;', 'array_aligned<Element,16384,1024> w;')
s = s.replace('      array_aligned<Element,2048,1024> q[2][Consumers];\n', '')
s = s.replace('    Scratch scratch;', '    array_aligned<Element,2048,1024> q[2][Consumers];\n    Scratch scratch;')
s = s.replace('s.scratch.attn.q', 's.q')
s = s.replace('s.wfull.arrive_and_expect_tx(8192*sizeof(Element))', 's.wfull.arrive_and_expect_tx(16384*sizeof(Element))')
s = s.replace('which<2', 'which<4')
s = s.replace('auto const& wt=kv?(which==0?p.wk:p.wv):(which==0?p.wq:p.wg);',
              'auto const& wt=which==0?p.wk:(which==1?p.wv:(which==2?p.wq:p.wg));')
start = s.index('    // Q/gate are produced only')
end = s.index('    if(qt<nt) {\n      auto bg=', start)
s = s[:start]+s[end:]
start = s.index('    typename C::Proj mma;')
end = s.index('  __syncthreads();\n  {\n    int batch=',start)
proj = s[start:end].replace('Shape<_64,_64>{}', 'Shape<_64,_128>{}')
# Z tiles still have K64; only the output N dimension expands.
proj = proj.replace('local_tile(zg(_,_,row),Shape<_64,_128>{}', 'local_tile(zg(_,_,row),Shape<_64,_64>{}')
proj = proj.replace('local_tile(ws,Shape<_64,_128>{}', 'local_tile(ws,Shape<_128,_64>{}')
proj = proj.replace('store_pair(cast_smem_ptr_to_uint(s.kv[nd/32][qt].data())+off,acc(x),acc(x+1));',
'''if(nd<64) store_pair(cast_smem_ptr_to_uint(s.kv[nd/32][qt].data())+off,acc(x),acc(x+1));
        else if(qt/Consumers==int(blockIdx.z))
          store_pair(cast_smem_ptr_to_uint(s.q[nd/32-2][c].data())+off,acc(x),acc(x+1));''')
s = s[:start]+proj+s[end:]
path=r/'joint4'
assert not (path/'build-ready.json').exists()
path.mkdir(exist_ok=True)
(path/'fused.cu').write_text(s)
