"""Recover the six-CTA register budget lost by the first Q projection fusion."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'q_only_fused/fused.cu').read_text()
assert s.count('__launch_bounds__(128,5)')==1
s=s.replace('__launch_bounds__(128,5)','__launch_bounds__(128,6)')
d=r/'q_only_cta6';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
