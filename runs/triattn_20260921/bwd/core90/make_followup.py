"""Isolated follow-ups to the qualified R8/TMA and dQ/ldmatrix installation."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
bias = (root / 'bias_fusion/rs8_tma_barrier/grouped.cu').read_text()
for registers in (32, 40):
    # Initial 168*384 register allocation covers 128*40 + 256*232.
    # Keep all qualified shared-memory reader/reuse barriers unchanged.
    source = bias.replace('warpgroup_reg_dealloc<24>()',
                          'warpgroup_reg_dealloc<%d>()' % registers)
    target = root / ('bias_fusion/rs8_producer%d' % registers)
    target.mkdir(exist_ok=True)
    (target / 'grouped.cu').write_text(source)

source = (root / 'dq/rs_tma_ldmatrix/fused.cu').read_text()
old = 'flash::gemm<true,-1>(smma,qa,kb,score);flash::gemm<true,0>(smma,da,vb,dp);'
assert source.count(old) == 1
source = source.replace(old, '''flash::gemm<true,-1>(smma,qa,kb,score);flash::gemm<true,-1>(smma,da,vb,dp);
  // Score is the older committed group. Overlap softmax with the dP group.
  warpgroup_wait<1>();warpgroup_fence_operand(score);''')
old = 'dp(x)=pr*(dp(x)-((x%4)/2?d1:d0));'
assert source.count(old) == 1
source = source.replace(old, 'score(x)=pr;')
old = '  auto acc_a=make_tensor(dp.data(),'
assert source.count(old) == 1
source = source.replace(old, '''  warpgroup_wait<0>();warpgroup_fence_operand(dp);
  #pragma unroll
  for(int x=0;x<size(dp);++x)dp(x)=score(x)*(dp(x)-((x%4)/2?d1:d0));
  auto acc_a=make_tensor(dp.data(),''')
target = root / 'dq/rs_softmax_overlap'
target.mkdir(exist_ok=True)
(target / 'fused.cu').write_text(source)
