from pathlib import Path

root=Path(__file__).resolve().parent.parent/'dq'
for k,ctas in ((32,5),(128,3)):
    s=(root/'rs_coop_tma/fused.cu').read_text()
    s=s.replace(' using KT=',f' using KL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{{}},Shape<_{k},_32>{{}}));\n using KT=')
    s=s.replace('Shape<_32,_64>',f'Shape<_32,_{k}>').replace('Shape<_64,_64>',f'Shape<_64,_{k}>')
    s=s.replace('Shape<_64,_64,_32>',f'Shape<_64,_{k},_32>').replace('Shape<_64,_32,_64>',f'Shape<_64,_32,_{k}>')
    if k==32:
        # A 32-element contiguous bias row requires SW64 rather than SW128.
        s=s.replace('GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_32>', 'GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>')
    s=s.replace(' using TB=',f' using TK=decltype(make_tma_copy(SM90_TMA_LOAD{{}},make_tensor(make_gmem_ptr((Element const*)nullptr),QS{{}},QStride{{}}),KL{{}},Shape<_{k},_32>{{}},_1{{}}));\n using TB=')
    s=s.replace('array_aligned<Element,2048,1024> q,dout,k[2],v[2];', f'array_aligned<Element,2048,1024> q,dout;\n  array_aligned<Element,{k*32},1024> k[2],v[2];')
    s=s.replace('array_aligned<Element,4096,1024> bias[2];',f'array_aligned<Element,{k*64},1024> bias[2];')
    s=s.replace('struct Params {TQ q,k,v,dout;', 'struct Params {TQ q;TK k,v;TQ dout;')
    s=s.replace('__launch_bounds__(128,4)',f'__launch_bounds__(128,{ctas})')
    s=s.replace('kt<L/64',f'kt<(L+{k-1})/{k}').replace('kt+1<L/64',f'kt+1<(L+{k-1})/{k}')
    s=s.replace('(2*2048+4096)*sizeof(Element)',f'(2*{k*32}+{k*64})*sizeof(Element)')
    for v in ('kg','vg'):
        s=s.replace(f'local_tile({v}(_,_,h,row),Shape<_64,_32>',f'local_tile({v}(_,_,h,row),Shape<_{k},_32>')
    for v in ('k','v'):
        s=s.replace(f's.{v}[slot].data()),Config::QL{{}}',f's.{v}[slot].data()),Config::KL{{}}')
    if k>64:
        s=s.replace('dp(x+u)=pr*',f'if(kt*{k}+kr+u>=L)pr=0.f;dp(x+u)=pr*')
    pos=s.index(' auto bg=make_tensor',s.index('torch::Tensor backward'))
    s=s[:pos]+f''' auto make_k=[&](torch::Tensor const& x){{auto src=make_tensor(make_gmem_ptr((Element const*)x.data_ptr()),shape,Config::QStride{{x.stride(3),_1{{}},_32{{}},x.stride(2)}});return make_tma_copy(SM90_TMA_LOAD{{}},src,Config::KL{{}},Shape<_{k},_32>{{}},_1{{}});}};
'''+s[pos:]
    s=s.replace('p{make_q(q),make_q(k),make_q(v),make_q(dy)', 'p{make_q(q),make_k(k),make_k(v),make_q(dy)')
    d=root/f'rs_coop_k{k}';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
