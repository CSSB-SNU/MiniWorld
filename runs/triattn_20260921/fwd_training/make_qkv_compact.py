"""Compact resident QKV/gate: reuse one N64 accumulator and fewer projection WGs."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'qkv_five/fused.cu').read_text()
s=s.replace('int Capacity,int Consumers,int Stages','int Capacity,int Consumers,int Stages,int Projectors')
s=s.replace('Config<Capacity,Consumers,Stages>','Config<Capacity,Consumers,Stages,Projectors>')
s=s.replace('qkv_attention_resident','qkv_attention_compact')
s=s.replace('qkv_attention_compact<Capacity,Consumers,Stages>','qkv_attention_compact<Capacity,Consumers,Stages,Projectors>')
s=s.replace('Shape<_64,_64>{}));\n  using WL=','Shape<_64,_128>{}));\n  using WL=',1)
s=s.replace('Shape<_32,_128>{}));\n  using ZS=', 'Shape<_32,_64>{}));\n  using ZS=',1)
s=s.replace('ZL{},Shape<_64,_64>{},_1{}','ZL{},Shape<_64,_128>{},_1{}')
s=s.replace('WL{},Shape<_32,_128>{},_1{}','WL{},Shape<_32,_64>{},_1{}')
s=s.replace('Shape<_64,_32,_64>>()));\n  using Score=', 'Shape<_64,_64,_128>>()));\n  using Score=')
s=s.replace('array_aligned<Element,4096,1024> z[Consumers]; array_aligned<Element,4096,1024> w[3];',
            'array_aligned<Element,8192,1024> z[Projectors]; array_aligned<Element,8192,1024> w[2];')
s=s.replace('TW wq,wk,wv;', 'TW wq,wg,wk,wv;').replace('TO saveq,savek,savev,out;', 'TO saveq,saveg,savek,savev,out;')
s=s.replace('    prefetch_tma_descriptor(p.wq.get_tma_descriptor());','    prefetch_tma_descriptor(p.wq.get_tma_descriptor());\n    prefetch_tma_descriptor(p.wg.get_tma_descriptor());',1)
b=s.index('  // Each warpgroup independently projects');e=s.index('  // QKV saves are complete.',b)
s=s[:b]+r'''  // Fewer projection WGs keep full K128 Z tiles without increasing attention scratch.
  if(c<Projectors) {
    typename C::Proj mma;auto mt=mma.get_slice(lane);
    auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_64>{}));
    auto acc=partition_fragment_C(mma,Shape<_64,_64>{});
    auto zg=p.z.get_tma_tensor(make_shape(L,_128{},L));
    auto zs=make_tensor(make_smem_ptr(s.scratch.proj.z[c].data()),typename C::ZL{});
    auto za=mt.partition_fragment_A(zs);
    if(tid==0) {
      s.wfull.arrive_and_expect_tx(4*4096*sizeof(Element));
      #pragma unroll
      for(int which=0;which<4;++which) {
        auto const& wt=which==0?p.wq:(which==1?p.wg:(which==2?p.wk:p.wv));
        auto wg=wt.get_tma_tensor(make_shape(_128{},_128{}));
        auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w[which/2].data()),typename C::ZL{});
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk) {
          auto src=local_tile(wg,Shape<_32,_64>{},make_coord(h,chunk));
          auto dst=local_tile(ws,Shape<_32,_64>{},make_coord(which%2,chunk));
          tma_load(wt,src,dst,s.wfull);
        }
      }
    }
    s.wfull.wait(0);
    for(int qt=c;qt<nt;qt+=Projectors) {
      if(lane==0) {
        s.zfull[c].arrive_and_expect_tx(8192*sizeof(Element));
        auto zt=local_tile(zg(_,_,row),Shape<_64,_128>{},make_coord(qt,0));
        tma_load(p.z,zt,zs,s.zfull[c]);
      }
      s.zfull[c].wait((qt/Projectors)%2);
      #pragma unroll
      for(int pair=0;pair<2;++pair) {
        auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w[pair].data()),typename C::ZL{});
        auto wb=mt.partition_fragment_B(ws);
        flash::gemm<true,0>(mma,za,wb,acc);
        #pragma unroll
        for(int x=0;x<size(acc);x+=2) {
          int mr=get<0>(coord(x)),nd=get<1>(coord(x)),which=nd/32;
          Element* dst=pair==0?s.q[c+which*Projectors].data():s.kv[which][qt].data();
          int off=as_position_independent_swizzle_layout(typename C::QL{})(make_coord(mr,nd%32))*2;
          store_pair(cast_smem_ptr_to_uint(dst)+off,acc(x),acc(x+1));
        }
        cutlass::arch::fence_view_async_shared();cutlass::arch::NamedBarrier::sync(128,c+1);
        if(lane==0) {
          #pragma unroll
          for(int which=0;which<2;++which) {
            auto const& save=pair==0?(which==0?p.saveq:p.saveg):(which==0?p.savek:p.savev);
            auto sg=save.get_tma_tensor(make_shape(L,_32{},_4{},L));
            auto tile=local_tile(sg(_,_,h,row),Shape<_64,_32>{},make_coord(qt,0));
            auto src=make_tensor(make_smem_ptr(pair==0?s.q[c+which*Projectors].data():s.kv[which][qt].data()),typename C::QL{});
            auto cp=save.get_slice(_0{});copy(save,cp.partition_S(src),cp.partition_D(tile));
          }
          tma_store_arrive();
        }
      }
      // Complete global Q saves before the attention phase reloads them.
      if(lane==0)asm volatile("cp.async.bulk.wait_group 0;":::"memory");
      cutlass::arch::NamedBarrier::sync(128,c+1);
    }
  }
'''+s[e:]
s=s.replace('torch::Tensor wv,torch::Tensor b,', 'torch::Tensor wv,torch::Tensor wg,torch::Tensor b,')
s=s.replace('torch::Tensor v,torch::Tensor out,', 'torch::Tensor v,torch::Tensor gate,torch::Tensor out,')
s=s.replace('typename C::ZL{},Shape<_64,_64>{}', 'typename C::ZL{},Shape<_64,_128>{}')
s=s.replace('typename C::WL{},Shape<_32,_128>{}', 'typename C::WL{},Shape<_32,_64>{}')
s=s.replace('p{tz,weight(wq),weight(wk),weight(wv),tb,tq,save(q),save(k),save(v),',
            'p{tz,weight(wq),weight(wg),weight(wk),weight(wv),tb,tq,save(q),save(gate),save(k),save(v),')
s=s.replace('auto gate=at::linear(z,wg);', 'auto gate=torch::empty_like(z);')
b=s.index('  if(L<=128)');e=s.index('  C10_CUDA_KERNEL_LAUNCH_CHECK();',b)
s=s[:b]+'''  if(L<=128)launch<128,2,2,1>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
  else if(L<=384)launch<384,6,2,3>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
  else if(L==768)launch<768,6,2,3>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
  else launch<1024,6,1,2>(z,wq,wk,wv,wg,b,q,k,v,gate,o,lse);
'''+s[e:]
s=s.replace('Config<1024,5,1>::Shared','Config<1024,6,1,2>::Shared')
d=r/'qkv_compact';assert not (d/'build-ready.json').exists()
d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
