"""Reuse each Z chunk for all three projections; retain three small weight tiles."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'resident_qkv/fused.cu').read_text()
s=s.replace('using ZL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_128>{}));',
            'using ZL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>{}));')
s=s.replace('ZL{},Shape<_64,_128>', 'ZL{},Shape<_64,_64>')
s=s.replace('Shape<_64,_32,_128>', 'Shape<_64,_32,_64>')
s=s.replace('array_aligned<Element,8192,1024> z; array_aligned<Element,4096,1024> w;',
            'array_aligned<Element,4096,1024> z; array_aligned<Element,4096,1024> w[3];')
start=s.index('  // Projection producer.')
end=s.index('  // QKV saves are complete.')
s=s[:start]+r'''
  // A single input Z chunk feeds all three projections. Full head weights stay resident.
  if(c==0) {
    typename C::Proj mma; auto mt=mma.get_slice(lane);
    auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
    auto aq=partition_fragment_C(mma,Shape<_64,_32>{});
    auto ak=partition_fragment_C(mma,Shape<_64,_32>{});
    auto av=partition_fragment_C(mma,Shape<_64,_32>{});
    auto zg=p.z.get_tma_tensor(make_shape(L,_128{},L));
    auto zs=make_tensor(make_smem_ptr(s.scratch.proj.z.data()),typename C::ZL{});
    auto qa=mt.partition_fragment_A(zs);
    if(lane==0) {
      s.wfull.arrive_and_expect_tx(3*4096*sizeof(Element));
      #pragma unroll
      for(int which=0;which<3;++which) {
        auto const& wt=(which==0?p.wq:(which==1?p.wk:p.wv));
        auto wg=wt.get_tma_tensor(make_shape(_128{},_128{}));
        auto tile=local_tile(wg,Shape<_32,_128>{},make_coord(h,0));
        auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w[which].data()),typename C::WL{});
        tma_load(wt,tile,ws,s.wfull);
      }
    }
    s.wfull.wait(0);
    for(int qt=0;qt<nt;++qt) {
      #pragma unroll
      for(int chunk=0;chunk<2;++chunk) {
        if(lane==0) {
          s.zfull.arrive_and_expect_tx(4096*sizeof(Element));
          auto tile=local_tile(zg(_,_,row),Shape<_64,_64>{},make_coord(qt,chunk));
          tma_load(p.z,tile,zs,s.zfull);
        }
        s.zfull.wait(chunk);
        auto wq=make_tensor(make_smem_ptr(s.scratch.proj.w[0].data()),typename C::WL{});
        auto wk=make_tensor(make_smem_ptr(s.scratch.proj.w[1].data()),typename C::WL{});
        auto wv=make_tensor(make_smem_ptr(s.scratch.proj.w[2].data()),typename C::WL{});
        auto bq=mt.partition_fragment_B(local_tile(wq,Shape<_32,_64>{},make_coord(0,chunk)));
        auto bk=mt.partition_fragment_B(local_tile(wk,Shape<_32,_64>{},make_coord(0,chunk)));
        auto bv=mt.partition_fragment_B(local_tile(wv,Shape<_32,_64>{},make_coord(0,chunk)));
        if(chunk==0) {
          flash::gemm<true,-1>(mma,qa,bq,aq);
          flash::gemm<true,-1>(mma,qa,bk,ak);
          flash::gemm<true,-1>(mma,qa,bv,av);
        } else {
          flash::gemm<false,-1>(mma,qa,bq,aq);
          flash::gemm<false,-1>(mma,qa,bk,ak);
          flash::gemm<false,-1>(mma,qa,bv,av);
        }
        warpgroup_wait<0>();
        warpgroup_fence_operand(aq);warpgroup_fence_operand(ak);warpgroup_fence_operand(av);
        cutlass::arch::NamedBarrier::sync(128,0);
      }
      auto store=[&](auto const& acc,int which) {
        uint32_t dst=cast_smem_ptr_to_uint(s.qkv[which][qt].data());
        #pragma unroll
        for(int x=0;x<size(acc);x+=2) {
          int mr=get<0>(coord(x)),nd=get<1>(coord(x));
          int off=as_position_independent_swizzle_layout(typename C::QL{})(make_coord(mr,nd))*2;
          store_pair(dst+off,acc(x),acc(x+1));
        }
      };
      store(aq,0);store(ak,1);store(av,2);
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(128,0);
      if(lane==0) {
        #pragma unroll
        for(int which=0;which<3;++which) {
          auto const& save=(which==0?p.saveq:(which==1?p.savek:p.savev));
          auto sg=save.get_tma_tensor(make_shape(L,_32{},_4{},L));
          auto tile=local_tile(sg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
          auto src=make_tensor(make_smem_ptr(s.qkv[which][qt].data()),typename C::QL{});
          auto cp=save.get_slice(_0{});
          copy(save,cp.partition_S(src),cp.partition_D(tile));
        }
        tma_store_arrive();tma_store_wait<0>();
      }
    }
  }
''' + s[end:]
p=r/'resident_qkv_reuse';p.mkdir(exist_ok=True);(p/'fused.cu').write_text(s)
