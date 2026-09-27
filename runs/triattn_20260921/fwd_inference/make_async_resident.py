"""Overlap outstanding PV with the next QK, keeping RS registers live until the wait."""
from pathlib import Path
r=Path(__file__).resolve().parent
for consumers in (4,6):
    s=(r/('resident%d'%consumers)/'fused.cu').read_text()
    begin=s.index('    auto score=')
    end=s.index('    int ep_tid,',begin)
    part=s[begin:end]
    part=part.replace('    for(int kt=0;kt<nt;++kt) {', '''    auto ar=make_tensor(score.data(),flash::convert_layout_acc_Aregs<typename C::PV>(score.layout()));
    auto pr=make_tensor_like<Element>(ar);clear(pr);
    for(int kt=0;kt<nt;++kt) {''')
    part=part.replace('      flash::gemm<true,0>(smma,qa,kb,score);',
                      '      flash::gemm<true,0>(smma,qa,kb,score);\n      warpgroup_fence_operand(pr);')
    part=part.replace('        auto ar=make_tensor(score.data(),flash::convert_layout_acc_Aregs<typename C::PV>(score.layout()));\n        auto pr=make_tensor_like<Element>(ar); flash::convert_type_out(ar,pr);',
                      '        flash::convert_type_out(ar,pr);')
    part=part.replace('flash::gemm<false,0>(pmma,pr,vb,out);', 'flash::gemm<false,-1>(pmma,pr,vb,out);')
    part=part.replace('    warpgroup_wait<0>(); warpgroup_fence_operand(out);',
                      '    warpgroup_wait<0>(); warpgroup_fence_operand(out);warpgroup_fence_operand(pr);')
    s=s[:begin]+part+s[end:]
    path=r/('async%d'%consumers)
    assert not (path/'build-ready.json').exists()
    path.mkdir(exist_ok=True)
    (path/'fused.cu').write_text(s)
