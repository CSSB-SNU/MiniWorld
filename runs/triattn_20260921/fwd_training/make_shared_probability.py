"""Reuse consumed bias SMEM for P; SS PV removes register-resident P operands."""
from pathlib import Path
r = Path(__file__).resolve().parent

for name, base, bounds in (
    ('cooperative_pshared', 'cooperative_s2_retire', '__launch_bounds__(128,6)'),
    ('producer_warp_pshared', 'producer_warp_relaxed', '__launch_bounds__(160,5)'),
):
    import re
    s = (r / base / 'fused.cu').read_text()
    s = re.sub(r'__launch_bounds__\(\d+,\d+\)', bounds, s)
    s = s.replace('GMMA::rs_op_selector<Element,Element,float,Shape<_64,_32,_64>',
                  'GMMA::ss_op_selector<Element,Element,float,Shape<_64,_32,_64>')
    old = '''        auto ar=make_tensor(score.data(),flash::convert_layout_acc_Aregs<Config::PV>(score.layout()));
        auto pr=make_tensor_like<Element>(ar); flash::convert_type_out(ar,pr);'''
    assert old in s
    s = s.replace(old, '''        // Bias is dead. Reuse its entire tile for BF16 P, keeping FP32 sums in l.
        cutlass::arch::NamedBarrier::sync(128,1);
        #pragma unroll
        for(int x=0;x<size(score);x+=2) {
          int qr=get<0>(sc(x)), kr=get<1>(sc(x));
          int off=as_position_independent_swizzle_layout(Config::BL{})(make_coord(qr,kr))*2;
          store_pair(bp+off,score(x),score(x+1));
        }
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(128,1);
        auto sp=make_tensor(make_smem_ptr(s.bias[stage].data()),Config::BL{});
        auto pr=pt.partition_fragment_A(sp);''')
    p = r / name
    p.mkdir(exist_ok=True)
    (p / 'fused.cu').write_text(s)
