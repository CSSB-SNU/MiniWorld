"""Schedule adjacent projection-layout heads together to reuse 128B cache lines."""
from pathlib import Path
r = Path(__file__).resolve().parent
base = (r / 'cooperative_q1_s2/fused.cu').read_text()
for heads in (2, 4):
    s = base.replace('int tid=threadIdx.x, qt=blockIdx.x, rg=blockIdx.y, h=blockIdx.z, L=p.L, nt=L/64;',
        'int tid=threadIdx.x, qt=blockIdx.x/%d, rg=blockIdx.y, h=blockIdx.z*%d+blockIdx.x%%%d, L=p.L, nt=L/64;' % (heads, heads, heads))
    assert s != base
    s = s.replace('dim3(L/64,L/Config::R,4)', 'dim3(L/64*%d,L/Config::R,%d)' % (heads, 4//heads))
    p = r / ('cooperative_head%d' % heads)
    p.mkdir(exist_ok=True)
    (p / 'fused.cu').write_text(s)
