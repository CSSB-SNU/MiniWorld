"""Cluster-wide ownership of projection/bias scratch, in addition to per-slot readers."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'cluster2/fused.cu').read_text()
old='    __syncthreads();\n    if(qt<nt) {\n      auto bg='
assert old in s
s=s.replace(old,'    cute::cluster_arrive();cute::cluster_wait();\n    if(qt<nt) {\n      auto bg=')
old='    // Output TMA source reads retire before projection overwrites the union.\n    __syncthreads();'
assert old in s
s=s.replace(old,'''    // Both CTAs must retire multicast traffic before either reuses projection/bias scratch.
    cute::cluster_arrive();cute::cluster_wait();''')
path=r/'cluster2s'
assert not (path/'build-ready.json').exists()
path.mkdir(exist_ok=True)
(path/'fused.cu').write_text(s)
