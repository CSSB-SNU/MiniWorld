"""Add on-chip Q/dO prefetch depth to the async R8 reduction pipeline."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'bias_fusion'
base = (root / 'rs8_async_bias64/grouped.cu').read_text()
for stages in (4, 8):
    s = base.replace('static constexpr int NWG=2, RP=R/NWG;',
                     'static constexpr int NWG=2, RP=R/NWG, QS=%d;' % stages)
    s = s.replace('q[NWG][2],dout[NWG][2]', 'q[NWG][QS],dout[NWG][QS]')
    s = s.replace('lse[NWG][2],delta[NWG][2]', 'lse[NWG][QS],delta[NWG][QS]')
    s = s.replace('q_full[NWG][2]', 'q_full[NWG][QS]')
    s = s.replace('q_empty[NWG][2]', 'q_empty[NWG][QS]')
    old = '      for(int w=0;w<C::NWG;++w) {s.q_full[w][st].init(1);s.q_empty[w][st].init(1);}\n    }'
    new = '    }\n    for(int st=0;st<C::QS;++st)\n      for(int w=0;w<C::NWG;++w) {s.q_full[w][st].init(1);s.q_empty[w][st].init(1);}'
    assert s.count(old) == 1
    s = s.replace(old, new)
    s = s.replace('st=it%2,phase=(it/2)%2', 'st=it%C::QS,phase=(it/C::QS)%2')
    s = s.replace('slot=it%2,phase=(it/2)%2', 'slot=it%C::QS,phase=(it/C::QS)%2')
    target = root / ('rs8_async_q%d' % stages)
    target.mkdir(exist_ok=True)
    (target / 'grouped.cu').write_text(s)
