"""Keep QK accumulator registers read-only outside WGMMA; process a separate copy."""
from pathlib import Path
import argparse

p=argparse.ArgumentParser()
p.add_argument('--artifact',required=True)
p.add_argument('--width',type=int,choices=(32,64),default=64)
p.add_argument('--consumers',type=int,choices=(2,4),default=2)
p.add_argument('--drain-before-copy',action='store_true')
a=p.parse_args();r=Path(__file__).resolve().parent
s=(r/'pipe32_c4/fused.cu').read_text()
lo=s.index('    } else {\n      auto next_score=')
hi=s.index('    warpgroup_wait<0>(); warpgroup_fence_operand(out);',lo)
part=s[lo:hi]
part=part.replace('auto next_score=partition_fragment_C(smma,Shape<_64,_32>{});',
                  'auto work=partition_fragment_C(smma,Shape<_64,_32>{});')
part=part.replace('auto step=[&](auto& score,auto& next,int kt){',
                  'auto step=[&](int kt){')
part=part.replace('        if(kt+1<nt*2)issue_qk(next,kt+1);',
                  '        cute::copy(score,work);\n        if(kt+1<nt*2)issue_qk(score,kt+1);')
rs=part.index('        uint32_t bp=');re=part.index('        auto vv=',rs)
body=part[rs:re].replace('size(score)','size(work)').replace('score(x)','work(x)').replace('score.data()','work.data()')
part=part[:rs]+body+part[re:]
part=part.replace('for(int kt=0;kt<nt*2;kt+=2){step(score,next_score,kt);step(next_score,score,kt+1);}',
                  'for(int kt=0;kt<nt*2;++kt)step(kt);')
s=s[:lo]+part+s[hi:]
if a.drain_before_copy:
    s=s.replace('if(kt==0)warpgroup_wait<0>();else warpgroup_wait<1>();','warpgroup_wait<0>();')
    s=s.replace('// Queue invariant on entry: current QK, then previous PV (except the first step).',
                '// Drain both groups before reading QK registers; next QK overlaps this step softmax.')
if a.width==64:
    s=s.replace('  using KHL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_32,_32>{}));\n','')
    s=s.replace('GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_32>', 'GMMA::Layout_MN_SW64_Atom<Element>{},Shape<_32,_64>')
    s=s.replace('using BL=decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{},Shape<_64,_32>',
                'using BL=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{},Shape<_64,_64>')
    s=s.replace('BS{},BStride{}),BL{},Shape<_64,_32>', 'BS{},BStride{}),BL{},Shape<_64,_64>')
    s=s.replace('ss_op_selector<Element,Element,float,Shape<_64,_32,_32>', 'ss_op_selector<Element,Element,float,Shape<_64,_64,_32>')
    s=s.replace('rs_op_selector<Element,Element,float,Shape<_64,_32,_32>', 'rs_op_selector<Element,Element,float,Shape<_64,_32,_64>')
    s=s.replace('array_aligned<Element,2048,1024> bias[Consumers][Stages]', 'array_aligned<Element,4096,1024> bias[Consumers][Stages]')
    s=s.replace('s.bfull[c][stage].arrive_and_expect_tx(2048*sizeof(Element))', 's.bfull[c][stage].arrive_and_expect_tx(4096*sizeof(Element))')
    s=s.replace('bg(_,_,h),Shape<_64,_32>', 'bg(_,_,h),Shape<_64,_64>')
    s=s.replace('bg,typename C::BL{},Shape<_64,_32>', 'bg,typename C::BL{},Shape<_64,_64>')
    s=s.replace('partition_fragment_C(smma,Shape<_64,_32>{})', 'partition_fragment_C(smma,Shape<_64,_64>{})')
    s=s.replace('nt*2','nt')
    s=s.replace('s.kv[0][kt/2].data()+(kt%2)*1024),typename C::KHL{}', 's.kv[0][kt].data()),typename C::QL{}')
    s=s.replace('s.kv[1][kt/2].data()+(kt%2)*1024),typename C::VT{}', 's.kv[1][kt].data()),typename C::VT{}')
for cap in (384,768,1024):
    s=s.replace(f'launch<{cap},4,2,4>',f'launch<{cap},{a.consumers},2,{a.consumers}>')
s=s.replace('Config<1024,4,2,4>::Shared',f'Config<1024,{a.consumers},2,{a.consumers}>::Shared')
s=s.replace('qkv_attention_n32_pipeline',f'qkv_attention_n{a.width}_copy_pipeline')
folder=r/a.artifact;assert not (folder/'build-ready.json').exists()
folder.mkdir(exist_ok=True);(folder/'fused.cu').write_text(s)
