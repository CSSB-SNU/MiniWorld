"""Two resident CTAs, two M64 consumers each, three scores and two P buffers."""
from pathlib import Path
def transform(s):
 from make_m64_prefetch import transform as prefetch
 s=prefetch(s)
 v9=(Path(__file__).resolve().parent/'overlap_v9.cu').read_text()
 s=s.replace(' auto next_score=partition_fragment_C(qk,Shape<_64,Int<N>>{});',
             ' auto next_score=partition_fragment_C(qk,Shape<_64,Int<N>>{});\n auto third_score=partition_fragment_C(qk,Shape<_64,Int<N>>{});')
 marker=' bool missing_seed=false;'
 s=s.replace(marker,' auto pp_next=make_tensor_like<Element>(pp);\n'+marker)
 a=s.index(' auto issue_pv=');b=s.index(' if constexpr(Safe) {\n  for(int seq=',a)
 va=v9.index(' auto pack=');vb=v9.index(' if constexpr(Safe) {\n  for(int seq=',va)
 s=s[:a]+v9[va:vb]+s[b:]
 s=s.replace('exponentiate(score,cute::true_type{});issue_pv(score,seq);','exponentiate(score,cute::true_type{});pack(score,pp);issue_pv(pp,seq);')
 a=s.index(' } else {\n  // Prologue');b=s.index('\n auto id=',a)
 va=v9.index(' } else {\n  // Three score');vb=v9.index('\n auto id=',va)
 s=s[:a]+v9[va:vb]+s[b:]
 s=s.replace('int wg=tid/128,t=tid%128;','int wg=int(__reduce_max_sync(0xffffffffu,unsigned(tid)/128u)),t=tid%128;')
 s=s.replace('auto tq=qk.get_slice(t);auto tp=pv.get_slice(t);auto tl=ls.get_slice(t);','auto tq=qk.get_slice(0);auto tp=pv.get_slice(0);auto tl=ls.get_slice(0);')
 s=s.replace('auto id=tp.partition_C(', 'auto id=pv.get_slice(t).partition_C(')
 return s
