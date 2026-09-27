"""Keep query-group K/V reuse without a dedicated producer or register split."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'dq'
for g in (2, 4):
    s = (root / ('rs_query_group%d' % g) / 'fused.cu').read_text()
    s = s.replace('__launch_bounds__((G+1)*128,(G==2?2:1))',
                  '__launch_bounds__(G*128,(G==2?2:1))')
    s = s.replace('wg=tid/128-1', 'wg=tid/128')
    start = s.index(' if(tid<128){')
    end = s.index(' Config::Score smma;', start)
    resident_start = s.index('   auto shape=', start)
    resident_end = s.index('   auto kg=', resident_start)
    resident = s[resident_start:resident_end]
    loader_start = resident_end
    loader_end = s.index('\n   }\n  }\n  return;', loader_start)
    loader = s[loader_start:loader_end]
    loader = loader.replace('   for(int kt=0;kt<L/64;++kt){', '   auto load=[&](int kt){')
    loader = loader.replace('    s.empty[slot].wait(phase^1);\n', '')
    # Keep tensor descriptor state local to lane zero's issue path, as in the
    # installed cooperative kernel. The full CTA barrier protects stage reuse.
    loader = loader.replace('   auto load=[&](int kt){', '')
    s = s[:start] + ' if(tid==0){\n' + resident + '\n }\n auto load=[&](int kt){\n   auto shape=make_shape(L,_32{},_4{},L);\n' + loader + '\n };\n if(tid==0)load(0);\n' + s[end:]
    # Restrict to the grouped body; single fallback already prefetches.
    a = s.index('void dq_tma_grouped')
    body = s[a:].replace('s.full[slot].wait(phase);',
                         's.full[slot].wait(phase);\n  if(tid==0 && kt+1<L/64)load(kt+1);')
    s = s[:a] + body
    s = s.replace('cutlass::arch::NamedBarrier::sync(128,wg+1);if(lane==0)s.empty[slot].arrive();',
                  '__syncthreads();')
    s = s.replace('cutlass::arch::NamedBarrier::sync(128,wg+1);', '__syncthreads();')
    s = s.replace('(G+1)*128,sizeof(QueryShared<G>)', 'G*128,sizeof(QueryShared<G>)')
    assert 'warpgroup_reg_' not in s
    target = root / ('rs_query_coop%d' % g)
    target.mkdir(exist_ok=True)
    (target / 'fused.cu').write_text(s)
