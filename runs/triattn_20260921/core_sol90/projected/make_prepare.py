"""Preserve installed LN/G/bias arithmetic; emit rounded LN instead of Q/K/V."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'oc/opt_core/kernels/triattn_surround_tma'
s = (BASE / 'prologue.cu').read_text()
s = s.replace('triattn_tma_pro', 'triattn_projected_prepare')
s = s.replace('prologue_cuda', 'projected_prepare_cuda')
s = s.replace('float eps;int L;', 'float eps;int L;Elem* norm;')
s = s.replace('  load_weight(0,0);', '  load_weight(3,0);')
s = s.replace('  *reinterpret_cast<uint4*>(&a(m,lane*8))=output_bits;',
'''  *reinterpret_cast<uint4*>(&a(m,lane*8))=output_bits;
  *reinterpret_cast<uint4*>(p.norm+((int64_t(i)*p.L+j+m)*128)+lane*8)=output_bits;''')
s = s.replace('for(int chunk=0;chunk<4;chunk++)', 'for(int chunk=3;chunk<4;chunk++)')
s = s.replace('s.ready_w[stage].wait(chunk&1)', 's.ready_w[stage].wait(0)')
s = s.replace('float(eps),L};', 'float(eps),L,(Elem*)q.data_ptr()};')
assert 'load_weight(0,0)' not in s
assert 'wait(chunk&1)' not in s
(HERE / 'prepare.cu').write_text(s)
