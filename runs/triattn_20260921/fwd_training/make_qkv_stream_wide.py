"""Fuse producer K/V into N64 WGMMA and prefetch two normalized-input tiles."""
import argparse
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--consumers',type=int,choices=(2,4),required=True)
a=ap.parse_args();r=Path(__file__).resolve().parent
base='qkv_stream_c2_late88' if a.consumers==2 else 'qkv_stream_c4_static'
s=(r/base/'fused.cu').read_text()
s=s.replace('  using Score=', '  using KVProj=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_64,_128>>()));\n  using Score=',1)
s=s.replace('producer_z;', 'producer_z[2];')
s=s.replace('zfull[Consumers+1]', 'zfull[Consumers+2]').replace('cc<=Consumers;', 'cc<Consumers+2;')
begin=s.index('  if(c==Consumers) {\n',s.index('// All Q/gate readers'))
end=s.index('  } else {\n',begin)
producer=s[begin:end]
producer=producer.replace('    auto zs=make_tensor(make_smem_ptr(s.producer_z.data()),typename C::ZL{});\n','')
producer=producer.replace('    if(lane==0) {\n', '''    auto load_z=[&](int kt) {
      int slot=kt%2;
      s.zfull[Consumers+slot].arrive_and_expect_tx(8192*sizeof(Element));
      auto zs=make_tensor(make_smem_ptr(s.producer_z[slot].data()),typename C::ZL{});
      auto ztile=local_tile(zg(_,_,row),Shape<_64,_128>{},make_coord(kt,0));
      tma_load(p.z,ztile,zs,s.zfull[Consumers+slot]);
    };
    if(lane==0) {
      load_z(0);if(nt>1)load_z(1);
''',1)
begin_mma=producer.index('    typename C::Proj mma;');end_mma=producer.index('    for(int kt=',begin_mma)
producer=producer[:begin_mma]+'''    typename C::KVProj mma;auto mt=mma.get_slice(lane);
    auto coord=mt.partition_C(make_identity_tensor(Shape<_64,_64>{}));
    // Two adjacent N32 head weights are one swizzled N64 weight tile.
    auto ws=make_tensor(make_smem_ptr(s.scratch.attn.w[0].data()),typename C::ZL{});
    auto wb=mt.partition_fragment_B(ws);
    auto acc=partition_fragment_C(mma,Shape<_64,_64>{});
'''+producer[end_mma:]
b=producer.index('      if(lane==0) {\n        s.zfull[c].arrive_and_expect_tx');e=producer.index('      cutlass::arch::fence_view_async_shared();',b)
producer=producer[:b]+'''      s.zfull[Consumers+st].wait((kt/2)%2);
      auto zs=make_tensor(make_smem_ptr(s.producer_z[st].data()),typename C::ZL{});
      auto za=mt.partition_fragment_A(zs);
      flash::gemm<true,0>(mma,za,wb,acc);
      #pragma unroll
      for(int x=0;x<size(acc);x+=2) {
        int mr=get<0>(coord(x)),nd=get<1>(coord(x));
        int which=nd/32;
        int off=as_position_independent_swizzle_layout(typename C::QL{})(make_coord(mr,nd%32))*2;
        store_pair(cast_smem_ptr_to_uint(s.kv[which][st].data())+off,acc(x),acc(x+1));
      }
'''+producer[e:]
producer=producer.replace('      if(lane==0) {\n        // Only one', '      if(lane==0) {\n        if(kt+2<nt)load_z(kt+2);\n        // Only one',1)
s=s[:begin]+producer+s[end:]
d=r/('qkv_stream_wide%d'%a.consumers);assert not (d/'build-ready.json').exists()
d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
