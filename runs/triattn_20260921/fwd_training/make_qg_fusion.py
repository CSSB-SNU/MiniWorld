"""Reuse normalized input for gate and Q; preserve all training saves."""
import argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--weights',choices=('serial','preload'),required=True)
a=ap.parse_args();r=Path(__file__).resolve().parent
s=(r/'q_only_head4/fused.cu').read_text()
s=s.replace('qkv_attention_resident_qonly','qg_attention_fused')
s=s.replace('TW w; TQ k,v; TB bias; TO saveq,out;', 'TW w,gweight; TQ k,v; TB bias; TO saveq,saveg,out;')
s=s.replace('    prefetch_tma_descriptor(p.w.get_tma_descriptor());','    prefetch_tma_descriptor(p.w.get_tma_descriptor());\n    prefetch_tma_descriptor(p.gweight.get_tma_descriptor());')
start=s.index('  {\n    auto zg=p.z.get_tma_tensor')
end=s.index('  {\n    int r=0, lane=tid, row=rg;',start)
proj=r'''
  {
    // Z is loaded once. Gate and Q retain the original BF16 rounding boundaries.
    auto zg=p.z.get_tma_tensor(make_shape(L,_128{},L));
    auto zs=make_tensor(make_smem_ptr(s.scratch.proj.z.data()),Config::ZL{});
    auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w.data()),Config::WL{});
    if(tid==0) {
      s.qfull.arrive_and_expect_tx((8192+4096)*sizeof(Element));
      auto zz=local_tile(zg(_,_,rg),Shape<_64,_128>{},make_coord(qt,0));
      auto wg=p.gweight.get_tma_tensor(make_shape(_128{},_128{}));
      auto ww=local_tile(wg,Shape<_32,_128>{},make_coord(h,0));
      tma_load(p.z,zz,zs,s.qfull);tma_load(p.gweight,ww,ws,s.qfull);
    }
    #pragma unroll
    for(int which=0;which<2;++which) {
      if(which==1 && tid==0) {
        s.qfull.arrive_and_expect_tx(4096*sizeof(Element));
        auto wg=p.w.get_tma_tensor(make_shape(_128{},_128{}));
        auto ww=local_tile(wg,Shape<_32,_128>{},make_coord(h,0));
        tma_load(p.w,ww,ws,s.qfull);
      }
      // Q weights are issued only after the gate save consumes the shared source.
      s.qfull.wait(which);
      Config::Proj mma;auto mt=mma.get_slice(tid);
      auto acc=partition_fragment_C(mma,Shape<_64,_32>{});
      auto za=mt.partition_fragment_A(zs);auto wb=mt.partition_fragment_B(ws);
      flash::gemm<true,0>(mma,za,wb,acc);
      auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
      uint32_t dst=cast_smem_ptr_to_uint(s.q[0].data());
      #pragma unroll
      for(int x=0;x<size(acc);x+=2) {
        int mr=get<0>(coord(x)),nd=get<1>(coord(x));
        int off=as_position_independent_swizzle_layout(Config::QL{})(make_coord(mr,nd))*2;
        store_pair(dst+off,acc(x),acc(x+1));
      }
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(128,0);
      if(tid==0) {
        auto const& save=which==0?p.saveg:p.saveq;
        auto sg=save.get_tma_tensor(shape);
        auto tile=local_tile(sg(_,_,h,rg),Shape<_64,_32>{},make_coord(qt,0));
        auto src=make_tensor(make_smem_ptr(s.q[0].data()),Config::QL{});
        auto cp=save.get_slice(_0{});
        copy(save,cp.partition_S(src),cp.partition_D(tile));
        tma_store_arrive();tma_store_wait<0>();
      }
      // Retire every projection reader and the TMA save before scratch/output reuse.
      cutlass::arch::NamedBarrier::sync(128,0);
    }
    if(tid==0) {load(0);if(nt>1)load(1);}
  }
'''
if a.weights=='preload':
    s=s.replace('array_aligned<Element,4096,1024> w;} proj;', 'array_aligned<Element,4096,1024> w[2];} proj;')
    proj=proj.replace('    auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w.data()),Config::WL{});','')
    proj=proj.replace('(8192+4096)*sizeof(Element)', '(8192+2*4096)*sizeof(Element)')
    proj=proj.replace('      tma_load(p.z,zz,zs,s.qfull);tma_load(p.gweight,ww,ws,s.qfull);', '''      auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w[0].data()),Config::WL{});
      auto wqs=make_tensor(make_smem_ptr(s.scratch.proj.w[1].data()),Config::WL{});
      auto wqg=p.w.get_tma_tensor(make_shape(_128{},_128{}));
      auto wqt=local_tile(wqg,Shape<_32,_128>{},make_coord(h,0));
      tma_load(p.z,zz,zs,s.qfull);tma_load(p.gweight,ww,ws,s.qfull);tma_load(p.w,wqt,wqs,s.qfull);''')
    begin=proj.index('      if(which==1 && tid==0) {')
    finish=proj.index('      Config::Proj mma;',begin)
    proj=proj[:begin]+'''      s.qfull.wait(0);
      auto ws=make_tensor(make_smem_ptr(s.scratch.proj.w[which].data()),Config::WL{});
'''+proj[finish:]
start=s.index('  {\n    auto zg=p.z.get_tma_tensor')
end=s.index('  {\n    int r=0, lane=tid, row=rg;',start)
s=s[:start]+proj+s[end:]
s=s.replace('torch::Tensor wv,torch::Tensor b)', 'torch::Tensor wv,torch::Tensor wg,torch::Tensor b)')
s=s.replace('for(auto const& w:{wq,wk,wv})','for(auto const& w:{wq,wk,wv,wg})')
s=s.replace('auto q=torch::empty_like(z),k=', 'auto gate=torch::empty_like(z);\n  auto q=torch::empty_like(z),k=')
needle='  auto shape=make_shape(L,_32{},_4{},L);'
pos=s.index(needle,s.index('std::vector<torch::Tensor> launch_forward'))
s=s[:pos]+'''  auto gwg=make_tensor(make_gmem_ptr((Element const*)wg.data_ptr()),Config::WS{},Config::WStride{});
  auto gtw=make_tma_copy(SM90_TMA_LOAD{},gwg,Config::WL{},Shape<_32,_128>{},_1{});
'''+s[pos:]
s=s.replace('  Config::Params p{tz,tw,make_q(k),make_q(v),tb,saveq,to,lse.data_ptr<float>(),L};', '''  auto gg=make_tensor(make_gmem_ptr((Element*)gate.data_ptr()),shape,Config::QStride{128,_1{},_32{},int64_t(L)*128});
  auto saveg=make_tma_copy(SM90_TMA_STORE{},gg,Config::QL{},Shape<_64,_32>{},_1{});
  Config::Params p{tz,tw,gtw,make_q(k),make_q(v),tb,saveq,saveg,to,lse.data_ptr<float>(),L};''')
s=s.replace('view(k),view(v)};', 'view(k),view(v),gate};')
s=s.replace('auto wg=make_tensor(make_gmem_ptr((Element const*)wq.data_ptr())',
            'auto wq_global=make_tensor(make_gmem_ptr((Element const*)wq.data_ptr())')
s=s.replace('make_tma_copy(SM90_TMA_LOAD{},wg,Config::WL{}', 'make_tma_copy(SM90_TMA_LOAD{},wq_global,Config::WL{}')
assert 'auto wg=make_tensor' not in s
assert 's.bfull[stage]);\n  };\n\n  {' in s
d=r/('qg_'+a.weights+'3');assert not (d/'build-ready.json').exists(),'published artifact is immutable'
d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
