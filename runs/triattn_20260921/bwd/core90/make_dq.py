from pathlib import Path
root=Path(__file__).resolve().parent.parent/'dq'
s=(root/'vector_bias/fused.cu').read_text()
s=s.replace(' using KT=', ' using KL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_128,_32>{}));\n using KT=')
s=s.replace('Shape<_32,_64>', 'Shape<_32,_128>').replace('Shape<_64,_64>', 'Shape<_64,_128>')
s=s.replace('Shape<_64,_64,_32>', 'Shape<_64,_128,_32>').replace('Shape<_64,_32,_64>', 'Shape<_64,_32,_128>')
s=s.replace(' using TB=', ' using TK=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),QS{},QStride{}),KL{},Shape<_128,_32>{},_1{}));\n using TB=')
s=s.replace('array_aligned<Element,2048,1024> q,dout,k[2],v[2];', 'array_aligned<Element,2048,1024> q,dout;\n  array_aligned<Element,4096,1024> k[2],v[2];')
s=s.replace('array_aligned<Element,4096,1024> bias[2],ds;', 'array_aligned<Element,8192,1024> bias[2],ds;')
s=s.replace('struct Params {TQ q,k,v,dout;', 'struct Params {TQ q;TK k,v;TQ dout;')
s=s.replace('__launch_bounds__(256,3)', '__launch_bounds__(256,2)').replace('warpgroup_reg_alloc<128>', 'warpgroup_reg_alloc<232>')
s=s.replace('kt<L/64', 'kt<(L+127)/128').replace('(2*2048+4096)*sizeof(Element)', '(2*4096+8192)*sizeof(Element)')
s=s.replace('local_tile(kg(_,_,h,row),Shape<_64,_32>', 'local_tile(kg(_,_,h,row),Shape<_128,_32>').replace('local_tile(vg(_,_,h,row),Shape<_64,_32>', 'local_tile(vg(_,_,h,row),Shape<_128,_32>')
s=s.replace('s.k[slot].data()),Config::QL{}', 's.k[slot].data()),Config::KL{}').replace('s.v[slot].data()),Config::QL{}', 's.v[slot].data()),Config::KL{}')
s=s.replace('dp(x+u)=pr*', 'if(kt*128+kr+u>=L)pr=0.f;dp(x+u)=pr*')
pos=s.index(' auto bg=make_tensor',s.index('torch::Tensor backward'))
s=s[:pos]+''' auto make_k=[&](torch::Tensor const& x){auto src=make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),shape,Config::QStride{x.stride(3),_1{},_32{},x.stride(2)});return make_tma_copy(SM90_TMA_LOAD{},src,Config::KL{},Shape<_128,_32>{},_1{});};
'''+s[pos:]
s=s.replace('p{make_q(q),make_q(k),make_q(v),make_q(dy)', 'p{make_q(q),make_k(k),make_k(v),make_q(dy)')
d=root/'key128';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
