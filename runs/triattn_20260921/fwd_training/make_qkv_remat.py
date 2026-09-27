"""Recompute epilogue coordinates after softmax to avoid keeping them in local memory."""
from pathlib import Path
import re
r=Path(__file__).resolve().parent
s=(r/'qkv_compact_async4/fused.cu').read_text()
needle='''  auto sc=st.partition_C(make_identity_tensor(Shape<_64,_64>{}));
  auto oc=pt.partition_C(make_identity_tensor(Shape<_64,_32>{}));
'''
assert needle in s
s=s.replace(needle,'',1)
b=s.index('    float inv[2];');e=s.index('\n}\n\ntemplate<int Capacity',b)
epilogue=s[b:e]
for before,after in [('lane','ep_lane'),('c','ep_c'),('h','ep_h'),('row','ep_row'),('L','ep_L')]:
    epilogue=re.sub(r'\b'+before+r'\b',after,epilogue)
head='''    int ep_tid,ep_h,ep_row,ep_nt;
    asm volatile("mov.u32 %0, %%tid.x;":"=r"(ep_tid));
    asm volatile("mov.u32 %0, %%ctaid.x;":"=r"(ep_h));
    asm volatile("mov.u32 %0, %%ctaid.y;":"=r"(ep_row));
    asm volatile("mov.u32 %0, %1;":"=r"(ep_nt):"r"(nt));
    int ep_lane=ep_tid%128,ep_c=ep_tid/128,ep_L=ep_nt*64;
    auto sc=smma.get_slice(ep_lane).partition_C(make_identity_tensor(Shape<_64,_64>{}));
    auto oc=pmma.get_slice(ep_lane).partition_C(make_identity_tensor(Shape<_64,_32>{}));
'''
s=s[:b]+head+epilogue+s[e:]
d=r/'qkv_compact_remat';assert not (d/'build-ready.json').exists()
d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
