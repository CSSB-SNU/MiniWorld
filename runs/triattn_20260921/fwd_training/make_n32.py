"""Smaller key tiles reduce register/shared storage per consumer CTA."""
from pathlib import Path

r = Path(__file__).resolve().parent
for name, baseline, bounds in (
    ('cooperative_n32', 'cooperative_s2_retire', '__launch_bounds__(128,8)'),
    ('producer_warp_n32', 'producer_warp_relaxed', '__launch_bounds__(160,6)'),
):
    s = (r / baseline / 'fused.cu').read_text()
    import re
    s = re.sub(r'__launch_bounds__\(\d+,\d+\)', bounds, s)
    s = s.replace('nt=L/64', 'nt=L/32')
    s = s.replace('Shape<_64,_64,_32>', 'Shape<_64,_32,_32>')
    s = s.replace('Shape<_64,_32,_64>', 'Shape<_64,_32,_32>')
    s = s.replace('Shape<_32,_64>', 'Shape<_32,_32>')
    s = s.replace('Shape<_64,_64>', 'Shape<_64,_32>')
    s = s.replace('Layout_K_SW128_Atom<Element>', 'Layout_K_SW64_Atom<Element>')
    ql = next(line for line in s.splitlines() if 'using QL=' in line)
    kvl = ql.replace('using QL=', 'using KVL=').replace('Shape<_64,_32>', 'Shape<_32,_32>')
    s = s.replace(ql, ql + '\n' + kvl)
    tq = next(line for line in s.splitlines() if 'using TQ=' in line)
    tk = tq.replace('using TQ=', 'using TK=').replace('QL{}', 'KVL{}').replace('Shape<_64,_32>', 'Shape<_32,_32>')
    s = s.replace(tq, tq + '\n' + tk)
    s = s.replace('TQ q,k,v;', 'TQ q; TK k,v;')
    s = s.replace('array_aligned<Element,2048,1024> q[R], k[2][R], v[2][R];',
                  'array_aligned<Element,2048,1024> q[R];\n    array_aligned<Element,1024,1024> k[2][R], v[2][R];')
    s = s.replace('array_aligned<Element,4096,1024> bias[2];',
                  'array_aligned<Element,2048,1024> bias[2];')
    s = s.replace('2*2048*sizeof(Element)', '2*1024*sizeof(Element)')
    s = s.replace('4096*sizeof(Element)', '2048*sizeof(Element)')
    lines = []
    for line in s.splitlines():
        if 'local_tile(kg' in line or 'local_tile(vg' in line:
            line = line.replace('Shape<_64,_32>', 'Shape<_32,_32>')
        if ('s.k[' in line or 's.v[' in line) and 'Config::QL{}' in line:
            line = line.replace('Config::QL{}', 'Config::KVL{}')
        lines.append(line)
    s = '\n'.join(lines) + '\n'
    start = s.index('  auto make_q=[&]')
    end = s.index('  auto bg=', start)
    make_k = s[start:end].replace('auto make_q=', 'auto make_k=').replace('Config::QL{}', 'Config::KVL{}').replace('Shape<_64,_32>', 'Shape<_32,_32>')
    s = s[:end] + make_k + s[end:]
    s = s.replace('Config::Params p{make_q(q),make_q(k),make_q(v)',
                  'Config::Params p{make_q(q),make_k(k),make_k(v)')
    folder = r / name
    folder.mkdir(exist_ok=True)
    (folder / 'fused.cu').write_text(s)
