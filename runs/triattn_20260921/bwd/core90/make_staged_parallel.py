"""Separate producer coupling from the earlier FP32 probability staging trial."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'bias_fusion'
base = (root / 'rs8_producer32/grouped.cu').read_text()
s = (root / 'rs_r8_staged_p/grouped.cu').read_text()
helper = base[base.index('// Two independent MMAs'):base.index('template<int R> struct Config')]
s = s.replace('template<int R> struct Config', helper + 'template<int R> struct Config', 1)
start = s.index('  if(tid<128) {')
end = s.index('  typename C::ScoreMMA', start)
a = base.index('  if(tid<128) {')
b = base.index('  typename C::ScoreMMA', a)
producer = base[a:b].replace('warpgroup_reg_alloc<232>()', 'warpgroup_reg_alloc<112>()')
s = s[:start] + producer + s[end:]
s = s.replace('s.kv_full.init(1);', 's.kv_full.init(C::NWG);')
old = '''      flash::gemm<false,-1>(gmma,pa,dbt,dv);
      flash::gemm<false,0>(gmma,dsa,qbt,dk);'''
assert s.count(old) == 1
s = s.replace(old, '      gemm_pair<false>(gmma,pa,dbt,dv,dsa,qbt,dk);')
s = s.replace('m.def("backward",&backward);', 'm.attr("row_group")=8;m.def("backward",&backward);')
s = s.replace('// One TMA producer, four consumer warpgroups; each WG owns two outer rows.',
              '// Four independent TMA producer warps; four consumers, two rows each. FP32 P stays in shared memory.')
target = root / 'rs8_staged_parallel'
target.mkdir(exist_ok=True)
(target / 'grouped.cu').write_text(s)
