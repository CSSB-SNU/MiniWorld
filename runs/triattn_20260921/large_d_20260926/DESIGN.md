# TriangleAttention wide-channel continuation

The C128/H4/head32 campaign is closed at the user's request. The selected
installed training checkpoint18246 and the inference19157 entry are preserved;
the last dQ reuse experiment remains explicit. The corrected comparison against
the actual original Triton engine is in
`../bwd/baseline_comparison_20260926/TRITON_COMPARISON.md`.

Initial assumption pending the dimension clarification: pair width and total
QKV width are both256/512, H=4, hence per-head dimensions64/128. BF16 activations
and projection weights, FP32 LN affine, B=1, L384/768 initially, both directions.
Measure full training FWD, BWD and F+B, not just isolated kernel timing.

## Fusion boundaries and tensor lifetime

The existing native dispatch requires C128/head32. At wider shapes it falls
back to original Triton attention and ordinary projection gradients. Start by
extending the native attention boundary without changing outer-module math:

- FWD consumes projected Q/K/V and bias and emits projection-layout O plus FP32
  base2 LSE. Online softmax keeps probabilities on chip.
- BWD keeps delta preprocessing, grouped dK/dV plus dBias partial production,
  and query-owned dQ. No L-cubed probability or dS tensor is written.
- Preserve the grouped dK/dV + dBias fusion; use R4 at head64 and R2 at head128
  initially. Larger heads require more resident dK/dV accumulators and Q/K/V
  shared storage, so keeping the head32 R8 grouping would exceed H100 resources.
- At head64/R4, FP32 grouped dBias scratch is half of the original per-row BF16
  scratch. At head128/R2 its byte count equals the original scratch. Do not claim
  HBM savings for that case; WGMMA/TMA scheduling must earn any speedup.
- TMA stages Q/K/V/dO/bias, WGMMA computes score and gradient products. Preserve
  producer/consumer barriers and final buffer retirement. Accumulators remain
  FP32; final gradients are BF16 with FP32 grouped bias reduction.

Projection/LN/gate fusion is a separate follow-up guided by measured attribution.
Do not blindly fuse QKV projection at wider head sizes: recomputation and shared
memory/accumulator pressure grow with channel width. Use complete-module results
to select any candidate.

Experiments stay under this directory. No serving edits or baseline relabeling.
Source and binary hashes, actual kernel profiles, all gradients, edge masks,
changed-input CUDA Graph replay and all three sanitizer checks are required
before treating a native candidate as qualified.
