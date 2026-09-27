"""Overlap dP WGMMA with ex2 on the16 logits retained in registers."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'dq'
s = (root / 'rs_half_p5/fused.cu').read_text()
old = 'float pr=ex2(logit*(SCALE*LOG2E)-((x%4)/2?m1:m0));score(x)=pr;'
assert s.count(old) == 1
s = s.replace(old, 'float normalized=logit*(SCALE*LOG2E)-((x%4)/2?m1:m0);score(x)=normalized;')
s = s.replace('float v=score(x);uint32_t ptr=', 'float v=ex2(score(x));uint32_t ptr=')
s = s.replace('  flash::gemm<true,0>(smma,da,vb,dp);', '''  flash::gemm<true,-1>(smma,da,vb,dp);
  #pragma unroll
  for(int x=0;x<16;++x)kept[x]=ex2(kept[x]);
  warpgroup_wait<0>();warpgroup_fence_operand(dp);''')
p = root / 'rs_half_p_overlap'
p.mkdir(exist_ok=True)
(p / 'fused.cu').write_text(s)
