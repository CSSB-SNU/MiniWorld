from pathlib import Path
root=Path(__file__).resolve().parent.parent
s=(root/'bias_fusion/rs_double_vec4/grouped.cu').read_text()
s=s.replace('  using ScoreMMA=', '  using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),ShapeQ{},StrideQ{}),KL{},Shape<_64,_32>{},_1{}));\n  using ScoreMMA=')
s=s.replace('    int L;\n  };','    int L; TO outk,outv;\n  };')
a=s.index('  #pragma unroll\n  for(int rr=0;rr<C::RP;++rr) {',s.index('void grouped_dkdv'))
b=s.index('\n}\n\n__global__ void reduce_bias',a)
s=s[:a]+'''  #pragma unroll
  for(int rr=0;rr<C::RP;++rr) {
    int r=wg*C::RP+rr,row=group*R+r;
    uint32_t skp=cast_smem_ptr_to_uint(s.k[r].data()),svp=cast_smem_ptr_to_uint(s.v[r].data());
    #pragma unroll
    for(int x=0;x<NG;x+=2) {
      int key=get<0>(gcoords(x)),d=get<1>(gcoords(x));
      int off=as_position_independent_swizzle_layout(typename C::KL{})(make_coord(key,d))*2;
      sm_store_pair(skp+off,dkbuf[rr][x]*SCALE,dkbuf[rr][x+1]*SCALE);
      sm_store_pair(svp+off,dvbuf[rr][x],dvbuf[rr][x+1]);
    }
    cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,wg+1);
    if(lane==0){
      auto shape=make_shape(L,_32{},_4{},L);
      auto gk=p.outk.get_tma_tensor(shape);auto gv=p.outv.get_tma_tensor(shape);
      auto ktile=local_tile(gk(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
      auto vtile=local_tile(gv(_,_,h,row),Shape<_64,_32>{},make_coord(kt,0));
      auto ks=make_tensor(make_smem_ptr(s.k[r].data()),typename C::KL{});
      auto vs=make_tensor(make_smem_ptr(s.v[r].data()),typename C::KL{});
      auto ck=p.outk.get_slice(_0{});auto cv=p.outv.get_slice(_0{});
      copy(p.outk,ck.partition_S(ks),ck.partition_D(ktile));
      copy(p.outv,cv.partition_S(vs),cv.partition_D(vtile));
      tma_store_arrive();tma_store_wait<0>();
    }
  }
'''+s[b:]
a=s.index('  C10_CUDA_CHECK(cudaFuncSetAttribute(grouped_dkdv')
s=s[:a]+'''  auto make_out=[&](torch::Tensor const& x){auto g=make_tensor(make_gmem_ptr((Element*)x.data_ptr()),shape,typename C::StrideQ{128,_1{},_32{},int64_t(L)*128});return make_tma_copy(SM90_TMA_STORE{},g,typename C::KL{},Shape<_64,_32>{},_1{});};
  p.outk=make_out(dk);p.outv=make_out(dv);
'''+s[a:]
s=s.replace('part.data_ptr<float>(),L};','part.data_ptr<float>(),L,{},{}};')
d=root/'bias_fusion/rs_tma_store';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)
# dQ: compare static 4-CTA compact producer and the safe original 3-CTA roles.
s=(root/'dq/rs4/fused.cu').read_text().replace('  cutlass::arch::warpgroup_reg_dealloc<24>();\n','').replace(' cutlass::arch::warpgroup_reg_alloc<112>();\n','')
d=root/'dq/rs4_static';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
s=(root/'dq/rs3/fused.cu').read_text()
s=s.replace(' using Score=', ' using TO=decltype(make_tma_copy(SM90_TMA_STORE{},make_tensor(make_gmem_ptr((Element*)nullptr),QS{},QStride{}),QL{},Shape<_64,_32>{},_1{}));\n using Score=')
s=s.replace('Element* dq;int L;};','Element* dq;int L;TO out;};')
a=s.index(' #pragma unroll\n for(int x=0;x<size(dq);++x)');b=s.index('\n}\ntorch::Tensor backward',a)
s=s[:a]+''' uint32_t op=cast_smem_ptr_to_uint(s.q.data());
 #pragma unroll
 for(int x=0;x<size(dq);x+=2){int qr=get<0>(gc(x)),d=get<1>(gc(x));int off=as_position_independent_swizzle_layout(Config::QL{})(make_coord(qr,d))*2;spair(op+off,dq(x)*SCALE,dq(x+1)*SCALE);}
 cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,1);
 if(lane==0){auto og=p.out.get_tma_tensor(make_shape(L,_32{},_4{},L));auto tile=local_tile(og(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));auto src=make_tensor(make_smem_ptr(s.q.data()),Config::QL{});auto c=p.out.get_slice(_0{});copy(p.out,c.partition_S(src),c.partition_D(tile));tma_store_arrive();tma_store_wait<0>();}
'''+s[b:]
a=s.index(' C10_CUDA_CHECK(cudaFuncSetAttribute(dq_tma')
s=s[:a]+''' auto og=make_tensor(make_gmem_ptr((Element*)result.data_ptr()),shape,Config::QStride{128,_1{},_32{},int64_t(L)*128});p.out=make_tma_copy(SM90_TMA_STORE{},og,Config::QL{},Shape<_64,_32>{},_1{});
'''+s[a:]
s=s.replace('(Element*)result.data_ptr(),L};','(Element*)result.data_ptr(),L,{}};')
d=root/'dq/rs_tma_store';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
