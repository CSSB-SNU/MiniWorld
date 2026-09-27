"""Rejected experiment: racecheck16866 still fails after __syncwarp().

Retain full warpgroup barriers in any installed implementation.
"""
from pathlib import Path

# wgmma.wait_group synchronizes a warp, not all four warps. Keep the release
# proof explicit: each warp arrives after its MMA completes, and the producer
# cannot recycle shared storage until all four arrivals have completed.
# https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group
root=Path(__file__).resolve().parent.parent
s=(root/'bias_fusion/rs_r8_pair/grouped.cu').read_text()
s=s.replace('s.q_empty[w][st].init(1);','s.q_empty[w][st].init(4);')
old='''      cutlass::arch::NamedBarrier::sync(128,wg+1);
      if(lane==0) s.q_empty[wg][slot].arrive();'''
new='''      __syncwarp();
      if(lane%32==0) s.q_empty[wg][slot].arrive();'''
assert old in s;s=s.replace(old,new)
d=root/'bias_fusion/rs_r8_warp_release';d.mkdir(exist_ok=True);(d/'grouped.cu').write_text(s)

s=(root/'dq/rs_ldmatrix/fused.cu').read_text()
s=s.replace('s.empty[i].init(1);','s.empty[i].init(4);')
s=s.replace('   int slot=kt%2;','   int slot=kt%2;\n   s.empty[slot].wait(((kt/2)%2)^1);')
old='flash::gemm<false,0>(gmma,dsa,ktb,dq);__syncthreads();'
assert old in s;s=s.replace(old,'flash::gemm<false,0>(gmma,dsa,ktb,dq);__syncwarp();if(lane%32==0)s.empty[slot].arrive();')
d=root/'dq/rs_warp_release';d.mkdir(exist_ok=True);(d/'fused.cu').write_text(s)
