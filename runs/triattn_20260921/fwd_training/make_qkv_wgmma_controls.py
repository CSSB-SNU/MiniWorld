"""Isolate eager PV retirement and shared-P lifetime in the aliased QKV kernel."""
from pathlib import Path
R = Path(__file__).resolve().parent
base = (R / 'qkv_compact_alias/fused.cu').read_text()
retire = base.replace('flash::gemm<false,-1>(pmma,pr,vb,out);',
                      'flash::gemm<false,0>(pmma,pr,vb,out);')
assert retire != base
four = retire.replace('launch<384,6,2,6>', 'launch<384,4,2,4>')
four = four.replace('launch<768,6,2,6>', 'launch<768,4,2,4>')
four = four.replace('launch<1024,6,1,4>', 'launch<1024,4,2,4>')
four = four.replace('Config<1024,6,1,4>::Shared', 'Config<1024,4,2,4>::Shared')
shared = base.replace('GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_64>',
                      'GMMA::ss_op_selector<Element,Element,float,Shape<_64,_32,_64>')
old = '''        auto ar=make_tensor(score.data(),flash::convert_layout_acc_Aregs<typename C::PV>(score.layout()));
        auto pr=make_tensor_like<Element>(ar); flash::convert_type_out(ar,pr);
'''
new = '''        // Current bias is dead. QK completion at the next iteration retires
        // this PV before any TMA refill of the same bias slot.
        auto pc=st.partition_C(make_identity_tensor(Shape<_64,_64>{}));
        #pragma unroll
        for(int x=0;x<size(score);x+=2) {
          int qr=get<0>(pc(x)),kr=get<1>(pc(x));
          int off=as_position_independent_swizzle_layout(typename C::BL{})(make_coord(qr,kr))*2;
          store_pair(bp+off,score(x),score(x+1));
        }
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(128,c+1);
        auto sp=make_tensor(make_smem_ptr(s.scratch.attn.bias[c][stage].data()),typename C::BL{});
        auto pr=pt.partition_fragment_A(sp);
'''
assert old in shared
shared = shared.replace(old, new, 1)
# One-stage bias refill previously occurred immediately after reading bias.
# Defer it until the next iteration after QK has also retired previous PV.
old = '''        if constexpr(Stages==1) {
          cutlass::arch::NamedBarrier::sync(128,c+1);
          if(lane==0 && kt+1<nt)load_bias(kt+1);
        }
'''
assert old in shared
shared = shared.replace(old, '', 1)
needle = '''      s.bfull[c][stage].wait(((qt/Consumers)*(nt/Stages)+kt/Stages)%2);'''
assert needle in shared
shared = shared.replace(needle, '''      if constexpr(Stages==1) {
        if(lane==0 && kt>0)load_bias(kt);
      }
''' + needle, 1)
for name, code in [('qkv_compact_retire6', retire), ('qkv_compact_retire4', four),
                   ('qkv_compact_pshared', shared)]:
    target = R / name
    assert not (target / 'build-ready.json').exists(), name
    target.mkdir(exist_ok=True)
    (target / 'fused.cu').write_text(code)
