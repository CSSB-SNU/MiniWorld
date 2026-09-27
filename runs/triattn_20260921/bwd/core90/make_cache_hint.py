"""Isolated TMA L2 retention policies; no math, layout or barrier changes."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'dq'
base = (root / 'rs_softmax_overlap/fused.cu').read_text()
old = 'cutlass::arch::ClusterTransactionBarrier& ready){auto c=tma.get_slice(_0{});copy(tma.with(reinterpret_cast<uint64_t&>(ready)),'
new = 'cutlass::arch::ClusterTransactionBarrier& ready,cute::TMA::CacheHintSm90 hint=cute::TMA::CacheHintSm90::EVICT_NORMAL){auto c=tma.get_slice(_0{});copy(tma.with(reinterpret_cast<uint64_t&>(ready),0,hint),'
assert base.count(old) == 1
base = base.replace(old, new)
for name, targets in [('rs_kv_last', ('k', 'v')), ('rs_bias_last', ('bias',))]:
    source = base
    for target in targets:
        if target == 'bias':
            old = 'Config::SL{}),s.full[slot]);'
        else:
            old = 's.%s[slot].data()),Config::QL{}),s.full[slot]);' % target
        assert source.count(old) == 1
        source = source.replace(old, old[:-2] + ',cute::TMA::CacheHintSm90::EVICT_LAST);')
    path = root / name
    path.mkdir(exist_ok=True)
    (path / 'fused.cu').write_text(source)
