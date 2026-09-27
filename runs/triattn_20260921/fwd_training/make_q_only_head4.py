"""Reuse the complete normalized input among four adjacent Q projection heads."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'q_only_fused/fused.cu').read_text()
assert 'qt=blockIdx.x/2' in s and 'h=blockIdx.z*2+blockIdx.x%2' in s
s=s.replace('qt=blockIdx.x/2','qt=blockIdx.x/4')
s=s.replace('h=blockIdx.z*2+blockIdx.x%2','h=blockIdx.x%4')
s=s.replace('dim3(L/64*2,L/Config::R,2)','dim3(L/64*4,L/Config::R,1)')
d=r/'q_only_head4';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
