# D256 backward optimization, stage 2 (in progress)

The SOL90 goal **has not been achieved**. Work has resumed on node02 under
the user's latest instruction; low QoS (`normal_h100`) is unchanged. This directory
continues the validated stage-1 D256 work with D128 producer/consumer and
packed-epilogue techniques. These are MiniWorld kernels. No engine dispatch or
autograd registration is changed.

## Current qualified paired Triton results (2026-09-26)

D384 uses checkpoint23, qualified at both lengths (**18898**) and directly
compared in **18910/18911**. D512 uses checkpoint24, qualified in **19351/19352**
and directly compared in **19353/19391**. All five strict fixtures, graph replay,
independent PyTorch, three sanitizers and both D512 overflow branches pass.
Checkpoint24 adds a dedicated loader warp and whole TMA maps only for D512 short.
Short affine initialization moves to the output gate. D512 short output LN
uses two ordered channel owners with disjoint prefix/final shared sums.
Checkpoint22 was rejected for a shared WAR warning; checkpoint23 fixes it.
Long kernel code is unchanged.
D256 uses `d256_pool_checkpoint.py`, qualified in **19247** and directly
compared in **19249/19254**. Short source dW uses N256 WGMMA and a verified
32/208 dynamic register split with120 initial registers, two resident CTAs
and zero spills. The cubin machine code is unchanged by the initial-allocation
metadata adjustment; its SASS register lifetimes and all sanitizer gates pass. Short affine initialization moves to the preceding
gate, removing both LN grid barriers; input-weight finishing uses packed stores.
Its short contraction uses a full N384 output tile with N256+N128 WGMMA
and ordered K. It retains the late-output-weight schedule.
Both output dW GEMMs overlap source work with separate workspaces and an explicit
graph join. Its long path is unchanged. Wide checkpoint20 retains the checkpoint17 D384 long contraction and
checkpoint18 D512 long input-weight overlap.

| D | L | 기존 BWD → 우리 BWD | BWD 배율 | 기존 F+B → 우리 F+B | F+B 배율 | 1.5배 충족 |
|---:|---:|---:|---:|---:|---:|:---:|
| 128 | 384 | 1.089 → 0.715 | 1.5232× | 1.634 → 1.018 | 1.6061× | 충족 |
| 128 | 768 | 4.342 → 2.874 | 1.5106× | 6.539 → 4.075 | 1.6047× | 충족 |
| 256 | 384 | 2.220 → 1.751 | 1.2683× | 3.546 → 2.621 | 1.3531× | 미달 |
| 256 | 768 | 12.112 → 7.128 | 1.6992× | 17.463 → 10.506 | 1.6621× | 충족 |
| 384 | 384 | 4.016 → 2.886 | 1.3914× | 6.370 → 4.585 | 1.3895× | 미달 |
| 384 | 768 | 18.892 → 12.333 | 1.5318× | 28.820 → 19.096 | 1.5092× | 충족 |
| 512 | 384 | 5.900 → 4.474 | 1.3187× | 9.506 → 7.048 | 1.3487× | 미달 |
| 512 | 768 | 28.003 → 18.621 | 1.5038× | 43.382 → 28.665 | 1.5134× | 충족 |

All three L768 widths now meet 1.5x in BWD and directly measured F+B.
Latest direct ratios at L768 are1.5318x BWD/1.5092x F+B for D384 and
1.5038x/1.5134x for D512. Both long paths retain the preceding kernel code.
All three L384 widths remain below
the threshold; close-to-threshold long results do not establish large margins.
The six-shape goal is **unmet**. These are qualified explicit entry points;
engine dispatch is unchanged. D128 reference remains paired jobs 17323/17324.
Wide paired graphs are scoped separately to avoid retaining six graph pools.
F+B is measured directly, never inferred by summing component timings.
The machine-readable authority is `qualified_speedups.json` with raw evidence
links in `qualified_speedups.md`.

Continuation on 2026-09-26:

- **17970** fully qualifies `wide_checkpoint9.py` at D384/512 and both
  lengths: five strict fixtures, graph replay, PyTorch, memcheck, racecheck,
  synccheck. It combines the overlapping preactivation/AB forward stores
  from **17885** with direct output-gate gradient publication into the dX
  prefix from corrected **17929**. The DWgate Lt algorithms are frozen in
  `prefix_gate_choices.json`; D256/D384 retain the tanh sigmoid headers.
- **17974** directly compares checkpoint9 to the existing Triton with scoped
  paired graphs. All four qualified results are in the table above. D384 long reaches
  1.469x BWD / 1.464x full; D512 long reaches 1.286x / 1.380x.
- **17975 / 17977** fully qualify and directly compare
  `d256_gate_checkpoint.py`, retaining the current D256 forward. All five
  strict fixtures, graphs, PyTorch and three sanitizers pass. Long reaches
  1.655x BWD / 1.652x full; short remains 1.186x / 1.294x.
- **17950** resident-dNorm/output-LN fusion preserves dNorm/dTriangle bits
  and strict gradients but regresses at every shape. D512 long combined
  stage 3.366 -> 6.380 ms, full 32.175 -> 35.044 ms. Rejected.
- **17955** independent WGMMA slot credits preserve GP bits and strict
  gradients but do not improve the paired workload. Rejected.

New architectural pilots, serialized after qualification/comparison:

- **17982** `wide_parallel_dual_norm.py`: separate warpgroups compute the
  exact packed forward and scalar backward normalization orders from one
  TMA input tile. Shared gamma/beta; saved scalar norm removes backward
  normalization. Both numerical equivalence and full time are required.
- **17984** current checkpoint9 component timings (diagnostic only).
- **17986** `check_joint_input_dw.py`: one nine-plane split GEMM replaces
  separate output-gate and eight-plane input weight-gradient GEMMs. The
  additional fixed-order gate reduction shares the input-LN finish.
- **17988** `wide_prefetch_ln.py`: two protected input/output slots allow
  next-tile TMA reads during current-tile output-LN arithmetic. Row8/16 and
  register occupancy tradeoffs are paired against checkpoint9.
- **17993** `wide_compact_contract_gp.py`: two K slots and direct cached
  mask loads reduce shared storage to 65,664 bytes; test two/three resident
  CTAs with the original accumulation order.
- **18001** `prefix_gate_dn.py`: gate derivatives and dNorm GEMM share one
  resident DP tile, while still publishing DP for dWproj and DG for dX.

- **18008** `d256_aliased_pipe_source.py`: two compute warpgroups overlap
  projection/GLU and full-D dW. Both input slots reuse dead dL/dR storage for
  dGate, fitting 115,072 shared bytes and targeting two CTAs per SM. Split
  counts, K accumulation and published GP order remain unchanged.

Completed pilot evidence after checkpoint9:

- **17982 / 18010**: parallel dual normalization is strict at both lengths
  but slower: short full 7.766 -> 7.882 ms, long 32.312 -> 33.251 ms.
  Rejected; extra arithmetic concurrency does not remove its cost.
- **17984**: actual checkpoint9 components confirm D512 long fused
  contraction/GP 5.668 ms, input dW 3.464 ms and prefix dX 3.787 ms. Output
  LN 2.544 ms and recomputed norm 1.438 ms remain separate costs. These are
  diagnostic timings, not additive substitutes for full graph measurement.
- **17986**: joint gate/input dW passes standard strict checks everywhere.
  D384 short full 5.024 -> 4.976 ms; long regresses 20.397 -> 20.502 ms.
  D512 short is neutral/slower 8.046 -> 8.051 ms, long 32.588 -> 32.391 ms.
  D512 weight outputs are bitwise; D384 dWgate differs by 2.26e-4 / 3.30e-4
  versus checkpoint9. No universal selection or qualification yet.
- **17988**: two-slot output-LN passes strict checks with bitwise dTriangle.
  Row16 D384 short full 4.841 -> 4.804 ms, but long regresses
  20.078 -> 20.174 ms. D512 short also regresses; row8 is slower everywhere.
  No universal selection.

Further completed architectural checks:

- **17993**: compact N128 contraction is strict/bitwise at two-CTA bounds
  but 30-40% slower in the fused stage. Three-CTA compilation fails because
  PTXAS needs at least 90 registers, versus its target of 80. The JSON records
  retain the two-CTA timings with `complete=false`; there is no three-CTA
  performance result.
- **18001**: shared-DP gate/dNorm fusion is bitwise in DP/DG/dNorm and strict
  at all six shapes, but slower everywhere. D256 short full 2.747 -> 2.803 ms;
  D512 long 32.176 -> 33.428 ms. Rejected.
- **18008**: aliased D256 two-compute-group pipeline has zero stack/spills,
  two resident CTAs and bitwise GP/partials. The best 96/160-register split
  remains slower: source 0.564 -> 0.587 / 2.448 -> 2.551 ms. Rejected.
- **18030**: N64 contraction achieves the intended four CTAs (two input slots)
  or three CTAs (three slots), with bitwise GP/strict gradients. Both regress
  at every shape; e.g. D512 long source 5.726 -> 6.175 ms for three slots.
- **18034**: a 32-thread loader plus one compute warpgroup fits two CTAs, but
  introduces a 152-byte stack at 168 registers. N128 and N256 both regress.
- **18038** profiles the current D256 source and those two alternatives.
  Current: 609 us, 412.6 MB reads / 600.9 MB writes, tensor activity 66.7%.
  Aliased: 631 us, 410.6 / 601.1 MB, tensor 64.1%, long-scoreboard ratio
  6.08 versus 1.64. Small loader: 921 us, 445.4 / 695.5 MB, tensor 43.7%.
  These are profile diagnostics; graph timings govern selection.
- **18040**: N192 contraction retains two CTAs and strict/bitwise GP but
  regresses all shapes, including D512 long 5.660 -> 6.651 ms fused.
- **18046**: explicit 192/200/208-register caps remove the small-loader
  occupancy assumption: all three have only one resident CTA and remain
  slower. No selection.

**18048** qualifies `wide_checkpoint10.py` only at its changed shapes:
D384 L384 combines joint input/gate dW and row16 prefetched output LN;
D512 L768 uses joint dW alone. **18050** is the dependent direct Triton
comparison. Both shapes passed all five stress cases, graphs, PyTorch,
memcheck, racecheck and synccheck. D384 short directly improves from 1.285x
to 1.318x BWD and 1.315x to 1.338x F+B. D512 long absolute times improve
to 21.424 ms BWD / 31.145 ms full, but the paired Triton also speeds up:
1.285863x / 1.380148x, effectively unchanged from checkpoint9 ratios.
All other shapes retain checkpoint9 behavior.

Additional completed pilots:

- **18053**: single compute warpgroup issues next-tile TMA during projection
  WGMMA. N128/N256 have zero stack/spills and two CTAs, but both source and
  full times regress at both D256 lengths. Rejected.
- **18055**: two-warpgroups gate/dNorm fusion is bitwise in DP/DG/dNorm and
  strict at all six shapes, but all full times regress. Rejected.
- **18059**: full-D resident input dW reuses each GP tile across three/four
  compute warpgroups. All weight outputs are bitwise, but isolated dW is
  slower everywhere. No selection.
- **18077**: first sparse dual norm records just 584 / 2217 changed BF16
  values, with bitwise norm/stats and strict gradients, including forced
  overflow fallback. Its extra shared buffer and comparison cost regress
  full time 7.735 -> 7.860 / 32.008 -> 32.597 ms. Rejected.
- **18081**: two outstanding WGMMA groups for resident input dW are strict
  and bitwise in GP/weights but regress at all shapes, including D512 long
  dW 3.436 -> 4.364 ms. Rejected.
- **18092**: in-register packed/scalar comparison avoids the extra full
  norm buffer. Lossless 32-bit patches restore scalar backward norm and
  flagged projection recomputes only affected 64-row tiles. Norm, stats,
  projection, dX and all weights are bitwise; forced-overflow fallback also
  passes. Changed tiles: 518/2304 short, 1927/9216 long. Paired full times
  improve 7.773 -> 7.650 / 32.143 -> 31.703 ms. Standard pilot only;
  stress/graph/sanitizer qualification remains pending.
- **18099**: cached gamma/beta and explicit warp-uniform fast/slow rows
  retain strict/bitwise norm, projection, dX and weights. Best paired full
  7.802 -> 7.493 / 32.449 -> 31.022 ms; forcing six CTAs regresses. The
  selected variant retains two-block launch bounds.
- **18102** gathers unique changed rows for the exact original Lt
  projection algorithm, scattering only those rows back. Both patch-list
  and compact-row overflow have exact GPU fallback paths and forced tests.
  Both overflow tests and the compact path pass strict with bitwise
  norm/projection/dX/weights. Unique changed rows: 575 / 2189. With cached
  affine and the earlier branch layout, full 7.828 -> 7.464 / 32.175 ->
  30.897 ms. No production change.
- **18108** fully qualifies `wide_checkpoint11.py` at both D512 lengths.
  It combines cached/unswitched lossless norm patches, unique-row compact
  projection and the previous checkpoint10 choices. All five stress cases,
  poisoned saved buffers, graphs and PyTorch pass. Normal, forced norm-list
  overflow, and forced compact-row overflow each pass memcheck, racecheck
  and synccheck with zero errors/hazards/warnings. The exact original Lt
  projection algorithm is reused for the compact matrices. **18110** completes
  the dependent direct scoped Triton comparison: D512 short 1.260684x BWD /
  1.314241x full; long 1.430172x / 1.446425x. FWD is 1.440326x / 1.481939x,
  including the extra exact statistics and patch detection. Both shapes
  remain below the required 1.5x BWD/full target.
- **18115** tests narrow forward norm tiles with cached affine parameters
  at D256/D384, preserving each original scalar reduction order.
- **18118** tests two cooperating warps per output-LN row. This reduces
  channel state per thread but changes reduction order, so strict gradient
  checks must pass before any timing is considered.
- **18122** measures a bitwise channel-major dNorm GEMM plus row-statistics
  LN/reconstruction prototype. This is a diagnostic for removing dTriangle
  traffic; consumer reuse and repeated statistics reads must be accounted for.
- **18130** tests a trailing 32-thread TMA loader and aligned WGMMA groups
  for pure input dW, with K32 stages and N128/N256 column partitions. This
  avoids the GLU producer register footprint of earlier failed small-loader
  source kernels. Actual occupancy, strict gradients and full time are gated.
- **18133** replaces GP sigmoid evaluation with a 65,536-entry FP32 lookup,
  initialized by the same sigmoid implementation for every BF16 input bit
  pattern. It introduces no interpolation or reduced precision; bitwise GP
  and paired full-workload checks decide whether cache traffic is worthwhile.

Completed follow-ups after checkpoint11:

- **18115**: cached forward norm passes standard strict checks at all four
  D256/D384 shapes. Row32 is beneficial at D384: full 4.819 -> 4.789 /
  20.063 -> 19.910 ms. D256 short gains only about 10 us; no D256 promotion.
- **18118**: paired warps change dTriangle by only about 5.3e-6, but final dX
  changes by 1.08e-4 to 1.34e-4, beyond the 2e-5 limit. All rejected before
  timing. Original LN reduction order remains mandatory for these candidates.
- **18122**: channel-major dNorm and deferred LN statistics/reconstruction
  preserve dNorm/dTriangle bits and strict gradients, but the complete
  reconstruction path is slower everywhere. D384 long stats alone 0.909 ms,
  versus original LN 1.565 ms; reconstruction raises it to 3.452 ms. This
  is diagnostic only, not a selected fusion or a performance gain.
- **18130 / 18169**: K32 and K64 small-loader input dW both achieve their
  requested two/three/four CTA residency and preserve strict gradients, but
  regress every full workload. K64 has REG90/STACK0; occupancy alone did not
  recover throughput. Rejected.
- **18133**: exact sigmoid lookup preserves GP bits but regresses fused GP
  by about 35-40% and full workloads at every shape. Rejected.
- **18151 / 18156**: separate affine worker with register-held output is
  bitwise in dTriangle and strict but spills and regresses. Full unrolling
  still has STACK224 / STACK688 at D384 / D512. Rejected.
- **18164 / 18166**: checkpoint12 fully qualifies the D384 row32 cached
  forward norm (all five fixtures, graphs, PyTorch, full memcheck and
  racecheck/synccheck of the sole new kernel). Inherited kernels retain their
  preceding qualifications. Direct Triton short: 1.316405x BWD / 1.341086x
  full; long: 1.467914x / 1.467330x. The goal remains unmet.
- **18168**: a separate shared output tile eliminates affine-worker spills
  (REG128/STACK0) and allows two CTAs at D512. D384 regresses, D512 short is
  neutral. D512 long LN 2.534 -> 2.261 ms, full 31.024 -> 30.450 ms, with
  bitwise dTriangle and strict gradients. Select only the long D512 shape.
- **18178**: streamed DP/weight dNorm, shared dNorm, channel-owned affine
  partials and a two-stage final reduction are strict and bitwise in
  dNorm/dTriangle. Full time regresses everywhere; D512 long combined stage
  3.376 -> 6.765 ms. Rejected.
- **18182** fully qualifies checkpoint13's D512 L768 affine-output LN:
  all five stress cases, graph replay, PyTorch and three sanitizers pass.
  Other shapes inherit checkpoint12 behavior. **18183** completes the direct
  Triton comparison: 1.452662x BWD / 1.456671x full (FWD 1.479360x).
  BWD still needs about 0.604 ms and F+B about 0.858 ms reduction to reach
  1.5x against those paired baseline times. No goal completion is claimed.
- **18184** tests a 192-thread LN organization (four norm warps, two affine
  warps). Caching all but the final 32 gamma values fits exactly within the
  shared-memory budget for three D384 CTAs. Actual residency, register use,
  strict gradients and full timings remain to be measured.
- **18188** applies the affine-worker variants to the unchanged D256 gate
  checkpoint; both lengths are required before selection.
- **18191** tests row16/256-thread and row32/128-or-256-thread versions of
  the exact lossless delta norm, retaining compact projection and the
  selected long D512 affine-output LN. It targets the forward overhead of
  preserving both packed and scalar normalization orders.

Later completed/pending measurements:

- **18184**: mixed affine LN achieves three D384 CTAs and two D512 CTAs,
  with strict gradients and bitwise dTriangle, but all full workloads regress.
  **18188** confirms both affine-worker organizations also regress D256 at
  both lengths. Preserve the original D256 LN and qualified D512-long-only
  checkpoint13 selection.
- **18191**: all delta-norm tile variants are strict and bitwise in restored
  norm. Row32/128 threads improves full 7.289 -> 7.272 / 30.390 -> 29.991 ms;
  its REG204 limits residency to two CTAs. 256-thread variants spill 216 bytes
  and are not selected.
- **18199**: separate packed/scalar-statistic warpgroups remain bitwise and
  strict. Both register splits regress short full time. Long isolated norm
  is also slower (1.774 -> 1.984 ms), although full medians drift favorably.
  Do not select from that inconsistent, one-length result.
- **18205**: explicit shared input reloads plus row32/three-block bounds are
  strict/bitwise and beneficial at both lengths. Norm 0.440 -> 0.351 /
  1.783 -> 1.334 ms; full 7.339 -> 7.264 / 30.498 -> 29.831 ms. REG168 with
  STACK32, three resident CTAs. Row16/four-block variants spill 200 bytes and
  are rejected. This remains a pilot pending full qualification.
- **18212**: additionally volatile shared gamma/beta gives no clear advantage
  over 18205. Row32/three-block remains beneficial (full 7.281 -> 7.198 /
  30.631 -> 29.938 ms); register/stack usage is still 168/32. No independent
  selection over the simpler explicit-input-reload variant.
- **18217** tests joint gate/input dW at 32/64 split counts with a dedicated
  partial buffer and each shape's supported Lt algorithms. Standard strict
  checks gate complete paired timing; no candidate is selected yet.
- **18224** bounds delta-norm loop unrolling at four/eight iterations to
  reduce register live ranges without changing scalar accumulation order.
  Tests row16/four-block and row32/three-block configurations at both lengths.

All pending pilots are unqualified; no production dispatch is modified.

New follow-up pilots:

- **17700 / 17726**: ordered Lt prefix dX plus cached TMA input LN improves D256
  full workload 2.859 -> 2.823 / 12.057 -> 11.762 ms versus the mask checkpoint.
  Standard strict passes both lengths; full qualification follows separately.
- **17711**: dual packed/scalar D512 output normalization is strict, but short
  full time regresses 8.462 -> 8.497 ms; long improves only 34.930 -> 34.804 ms.
  No paired-length selection.
- **17694 / 17695**: deferred TMA store completion is strict and GP/partials
  bitwise. Wide results are neutral except a small D512 long gain; D256 does
  not benefit. Current qualified checkpoints retain their original waits.

Completed new pilots and qualification (2026-09-26):

- **17752** qualifies D256 prefix Lt dX and cached input TMA LN at both lengths:
  all five strict fixtures, graphs, PyTorch, memcheck/racecheck/synccheck pass.
  **17754 / 17758** direct Triton results are in the table above. Frozen Lt
  index 3 and opaque algorithm are asserted in `d256_prefix_checkpoint.py`.
- **17757 / 17763**: direct packed GP loads plus two input slots and rank-major
  tile ordering are bitwise in GP. D512 GP 1.130 -> 1.054 / 4.540 -> 4.287 ms;
  paired full 8.317 -> 8.229 / 34.198 -> 33.606 ms. Full qualification pending.
- **17760 / 17769**: actual checkpoint6 component timings, diagnostic only.
  D384 short source 1.552 ms dominates; D512 short saved GP 1.136 ms and input
  dW 0.872 ms, prefix dX GEMM 0.952 ms. Component sums are not full timings.
- **17768 / 17770 / 17771**: explicit independent-weight streams with separate
  Lt workspaces preserve strict gradients/graphs. D256 short gains only ~14 us;
  D384/512 are neutral or slower. No selection.
- **17775 / 17776**: channel-major forward preactivation stores and coalesced
  global GP are bitwise, but no better than the simpler rank-major TMA pipe.
  4xSM grid is slower; 8/16xSM gives only small gains. No selection.
- **17777**: four-warpgroups split TMA issue from projection and two dW
  consumers. D384 short synchronous dW improves source 1.514 -> 1.419 ms,
  full 5.330 -> 5.236 ms. Two outstanding dW groups regress markedly even with
  the independent loader; reject that variant. Long-length pilot follows.

Pending continuation chain on node02 / normal_h100:

- **17778** confirms synchronous four-warpgroup D384 long: source
  6.272 -> 5.879 ms, full 21.261 -> 20.803 ms. GP/partials are bitwise;
  asynchronous consumers still regress and are rejected.
- **17780 / 17781** D256 cached tiled output LN: 16 rows improves short
  0.229 -> 0.207 ms but regresses long 0.865 -> 1.005 ms; no global change.
- **17782** qualifies `wide_checkpoint7.py`: D384 uses the independent TMA
  loader, D512 uses rank-major two-slot GP. Full five-stress/graph/PyTorch and
  three sanitizers at each shape. **17784** directly compares it to Triton.
- **17791** pilots one N192 WGMMA dW instruction per D384 consumer instead
  of three N64 instructions; retains four warpgroups and exact K order.
- **17793** tunes bitwise-preserving D256 output/contraction GEMMs while
  keeping the already frozen prefix dX. No production dispatch changes.
- **17796** pilots N192/N256 dW on the three-warpgroup recompute source.
  D512 restores the original forward and 32-way input reduction for this
  comparison, so its full timing includes removal of preactivation saves.
- **17798** tests output LN occupancy: cached D512 LN currently uses 206
  registers (two CTAs/SM). Explicit 3/4-block bounds and an 8-row TMA tile
  try to raise occupancy; `tile_transpose8.cuh` handles its unswizzled input.

Latest completed continuation evidence:

- **17782 / 17784**: wide checkpoint7 is fully qualified at all four shapes;
  current table includes completed direct Triton comparisons (all scoped pairs).
- **17791**: N192 on four warpgroups is bitwise but offers only ~0.3-1%.
  **17796** N192/N256 on three warpgroups is strict but regresses both short
  widths versus checkpoint7. No promotion of these full-N variants.
- **17793 / 17805**: frozen D256 Lt choices have negligible short benefit,
  but long full 11.940 -> 11.483 ms. **17809** passes all five stress cases,
  graph, independent PyTorch and three sanitizers at both lengths. **17812**
  direct Triton: long BWD 1.616x and F+B 1.622x, short F+B 1.271x.
- **17798 / 17807**: compact output LN passes strict, including the 8-row
  transpose. D384 16-row/minblocks3 improves short 0.390 -> 0.337 ms; long
  follow-up **17814** regresses the LN 1.543 -> 1.770 ms and full is neutral.
  D512 tighter bounds/8-row tiles regress strongly. No global selection.
- **17815 / 17818**: cached gamma with memory fences is strict, but does not
  fix the D512 regression and does not materially improve the D384 result.
- **17819**: per-two-rank ring waits preserve strict dX (bitwise) and weights,
  but ring finish still 1.49-1.56 ms versus current 1.00 ms. Rejected.
- **17820**: sequential L2-sized GP buffers plus ordered Lt dX are bitwise
  in dX; FP32 weight carry passes strict. Chunk costs dominate: finish
  1.31-1.99 ms versus ~1.00 ms over 4K/8K/16K rows. Rejected.
- **17825**: row-major GP, prefix and all dX/weight outputs are bitwise;
  source rises 0.582 -> 0.665 ms, GEMM is neutral, full regresses. Rejected.
- **17826 / 17829**: independent FP32 split output-weight gradients pass
  standard strict. Short gains are negligible; D384 long DWgate saves ~36 us,
  D512 long DWproj saves ~28 us. Full gains are only ~29 / 85 us. Not selected.
- **17835**: D256 paired-rank XN reuse with bulk masks and N256 dW is bitwise
  in GP and partials, but all variants regress full time at both lengths.
- **17840**: first contraction-fusion launch stopped during Python setup
  because PREFIX_COPY was not set. No GPU candidate result from this job.
  Corrected setup is explicit in **17843** and later scripts.
- **17843 / 17845 / 17848**: native contraction plus GP fusion is bitwise
  in GP and all dX/weights. Direct global preactivation access is slow;
  channel-major saves and then TMA epilogue reduce the regression, but the
  64x64 GEMM still loses to checkpoint7, especially at L768.
- **17850**: 384-thread/two-block launch bounds cannot encode N128 WGMMA
  (80 registers available versus 90 required). Compile failure only.
  **17853** uses one block and passes strict with bitwise GP/dX/weights;
  128x128 fusion is neutral short, slower long. Not selected.
- **17855**: two compute warpgroups, 128x128 and three input slots use
  90 registers without spills. Short full improves 8.329 -> 8.036 ms;
  long regresses 33.793 -> 35.890 ms. Pilot only, not qualified.
- **17857**: D512 four-warpgroup recompute removes preactivation saves,
  passes standard strict, but full regresses 8.365 -> 8.638 / 34.207 ->
  35.020 ms. Not selected.
- **17859** profiles fused contraction at L768. **17862** tests channel
  grouping of 1/2/4/8/16 instead of 32 to improve input-matrix L2 reuse.
  The original group32 kernel reads 23.016 GB and writes 4.820 GB in
  9.167 ms; tensor utilization is 25.86%, long-scoreboard stalls 12.28.
  Group1 cuts fused contraction+GP to 5.853 ms versus the separate path's
  7.169 ms. Pilot full improves 34.406 -> 32.754 ms, short 8.336 -> 7.899 ms.
- **17865** adapts the fusion to D384 with new channel-major preactivation
  saves and 8/16-way input dW. GP/dX are bitwise; weight errors stay below
  3.55e-4. Full improves 5.294 -> 4.944 / 21.260 -> 20.305 ms.
- **17868** found a synccheck error in the first checkpoint8: two template
  consumer branches reached different `__syncthreads` sites. Five stress
  fixtures, graph, PyTorch, memcheck and racecheck had passed, but this is
  **not qualification**. Remaining tasks and dependent 17872/17873 were
  cancelled. TwoGroupContractGP now runs all threads through one consumer
  function with a runtime warpgroup index and uniform CTA barrier sites.
- **17874** qualifies the corrected checkpoint8 at all four shapes: five
  strict stress fixtures, graph replay, independent PyTorch, memcheck,
  racecheck and synccheck all pass. **17876** comparison failed before
  timing because VALIDATION_ARRAY_JOB was missing. Corrected direct Triton
  comparison is **17906** with both explicit comparison environment values.
  All four comparisons completed; current qualified table is above.
- Pending follow-ups, serialized on node02: **17877** four compute groups
  at 128x256; **17879** D256 saved-front contraction fusion; **17885**
  independent 8KB preactivation / 4KB AB staging to overlap forward stores;
  **17893** emit output-gate gradient directly into the dX prefix and remove
  its later transpose/copy, with bitwise-preserving DWgate Lt selection.
- **17895** tests streaming/retained TMA cache priorities; **17897** tests
  explicit volatile shared-memory LN reloads to shorten register lifetimes.
- **17877** completed: 128x256 four-compute-group GP is bitwise but regresses
  all four shapes against corrected checkpoint8. No selection.
- **17917** tests D256 three/four-group recompute pipelines with its original
  8/16 weight splits, keeping the current front and prefix dX unchanged.

Further pilots after qualification:

- **17570** profiles the actual checkpoint4 backward. D512 L384 source
  takes 2.928 ms under NCU, tensor-pipe active 54.51%, and has substantial
  long-scoreboard stalls (5.575 per issue-active). Output/input TMA LN
  take 0.548 / 0.321 ms. Profiling is not a paired Triton speedup.
- **17572** tests 8/16 input dW splits without changing GP/dX arithmetic.
  8 splits pass short shapes but fail long weight tolerance (~5.8e-4).
  16 splits pass standard strict cases at both lengths. Short full times
  improve modestly; D512 L768 full regresses, so not globally selected.
- **17577** uses D128's multicast protocol for shared input tiles.
  Derivatives and dW partials match bitwise, but 2/4-CTA clusters are
  substantially slower at both tested widths; rejected.
- **17579** uses register-input projection WGMMA with balanced dynamic
  register budgets. Both strict pilots pass, both full workloads regress;
  rejected.
- **17585** tests narrow forward-normalization tiles. Its first task
  (17593) stalled in the 256-thread D384 transpose: channel tile count
  was not divisible by warp count, causing unequal barrier participation.
  Only that task was cancelled after confirming it remained running.
  `tile_transpose.cuh` now keeps barrier rounds uniform and predicates
  only loads/stores; remaining tasks run the corrected implementation.
  The partial failed-task record is retained and is not qualification.
- **17590** checks bitwise-preserving cuBLASLt algorithms for all dense
  output/dX/contraction GEMMs and measures their combined workload.
- **17594** tests early global mask prefetch versus bulk-TMA mask staging
  to address the measured source memory stalls.

Further work (2026-09-25, checkpoint5):

- Completed **17585**: D512 forward norm tile16/128 is strict and reduces
  output 1.027 -> 0.836 ms / 4.102 -> 3.321 ms, full 9.119 -> 8.938 /
  37.202 -> 36.506 ms. D384 tile changes add no material benefit.
- Completed **17590**: fixed bitwise-preserving Lt choices reduce paired
  full workloads by ~0.7-2.8%. Exact indices and opaque algorithm data are
  frozen in `wide_checkpoint5_lt.json`; this is not a Triton comparison.
- Completed **17594**: early mask prefetch is faster than bulk staging.
  GP and dW partials match bitwise; standard strict passes all four shapes.
  D384 source 1.619 -> 1.522 / 6.563 -> 6.056 ms; D512 short
  2.727 -> 2.573 ms. Full timings also improve.
- `wide_checkpoint5.py` combines early mask loads, the D512 tile output,
  and fixed Lt GEMMs. Full qualification **17620**, direct Triton paired
  timing **17632**. Qualified results must be read from completed artifacts.
- **17623**: two outstanding dW WGMMA groups; move input prefetch after GP
  publication and preload both slots to avoid cyclic credit waits.
- **17627**: port early/bulk mask loading to the selected D256 source.
- **17630**: tile D512 backward's scalar output normalization while
  retaining its legacy reduction order, separately from packed forward LN.
- **17634**: wide native ordered dX GEMM fused with input LN. Two consumer
  warpgroups cover the output width, eliminating prefix transpose and the
  dXn HBM write/read. Await correctness and measured full workload results.
- **17639**: test exact-partial Lt algorithms for saved-preactivation
  input dW; earlier untuned saved-front results remain rejected.

Completed follow-up evidence:

- **17623** async dW groups are bitwise/strict but slower: source
  1.460 -> 2.174 ms (D384), 2.439 -> 3.469 ms (D512), full also regresses.
  Rejected; original prefetch timing is materially better.
- **17627** D256 early mask reads are neutral; bulk mask staging improves
  source 0.566 -> 0.552 / 2.398 -> 2.332 ms and paired full by ~0.5%.
  The bulk candidate passes full strict/graph/PyTorch/three-sanitizer gates
  at both lengths in **17646**. Direct Triton comparison **17660** pending.
- **17630** D512 backward scalar norm tiling is bitwise but no material
  paired-length benefit; rejected.
- **17634** the native wide fused dX/LN is bitwise in dX/weights and strict
  overall, but slower: 0.860 -> 1.302 ms (D384), 1.366 -> 1.912 ms (D512).
  No promotion. This eliminates intermediates but loses enough GEMM
  throughput to regress the complete workload.
- **17639** choosing exact input-dW Lt algorithms still leaves saved-front
  full workloads slower/neutral; reject that global-store version.
- **17650** pilots shared gamma caching in wide input/output LN.
- **17654** reuses the existing forward staging allocation for TMA
  preactivation stores, addressing the global-store instruction cost.
- **17657** combines that store path with 8 FP32 input-dW splits at short
  lengths (16 for long lengths); reduced gradients retain strict gates.

Further source/lifetime work:

- **17650/17680**: shared gamma is strict at all four wide shapes.
  D384 output LN drops 0.418 -> 0.377 / 1.716 -> 1.520 ms; D512 long
  drops 2.669 -> 2.475 ms. Input gamma caching also improves the kernel.
- **17654**: forward TMA preactivation saves (existing 8KB/WG staging)
  pass strict and improve D512 short full 8.888 -> 8.560 ms. Unlike the
  rejected scalar global-store version, the extra front cost is 0.299 ms.
- **17657/17674**: short S8 / long S16 Lt input dW with those saves is
  strict. D512 full 8.972 -> 8.522 / 35.357 -> 34.482 ms. D384 long is
  neutral, so its selected path continues recomputation without saves.
- **17667** failed before launch because descriptor rewriting still
  expected the removed preactivation MMA. Correct transformation order;
  rerun **17672** is strict. Immediate native dW from saved preactivation
  has no clear advantage over Lt; no promotion.
- **17678**: split dW columns between two resident CTAs, with only one
  publishing common GP. Zero spills, strict, but no advantage over Lt.
- `wide_checkpoint6.py` combines shared input/output LN gamma at both
  widths and saved TMA preactivation + S8/S16 Lt input dW only at D512.
  Full validation **17683**, direct paired Triton **17689**. Frozen input
  dW algorithm data: `wide_checkpoint6_dw.json`. Native forward control
  temporarily restores the original front; saved preactivations are
  poisoned before the candidate during strict/graph fixtures.
- **17688**: two 64x128 dW consumers share X; direct BF16 preactivation
  reads avoid the FP32 expansion and keep two CTAs resident.
- **17692**: test the same dual source at D256 with TMA preactivation
  saves while retaining the existing input-statistics saves.
- **17694/17695**: defer TMA global-store completion until kernel exit;
  per-tile waits protect only shared-buffer reads. Test wide/D256.
- **17700**: revisit D256 prefix dX with the new TMA gamma-cached input
  LN and fixed bitwise-preserving Lt choices; the old global LN reducer
  was the large cost in the rejected 17408-era prefix implementation.

All follow-ups are serial on node02 / normal_h100.

## Wide checkpoint 3 qualified; checkpoint 4 in validation (2026-09-25)

Array **17496** passed all five strict cases, graph replay, independent
PyTorch, and memcheck/racecheck/synccheck on D384/512 x L384/768.
Paired comparison **17498** remains below the 1.5x target:

| D | L | Triton BWD → ours (ms) | BWD speedup | Triton F+B → ours (ms) | F+B speedup |
|---:|---:|---:|---:|---:|---:|
| 384 | 384 | 3.936 → 4.464 | 0.882x | 6.251 → 6.522 | 0.958x |
| 384 | 768 | 17.996 → 17.545 | 1.026x | 27.606 → 25.746 | 1.072x |
| 512 | 384 | 5.791 → 7.192 | 0.805x | 9.491 → 10.312 | 0.920x |
| 512 | 768 | 26.342 → 27.973 | 0.942x | 41.637 → 40.877 | 1.019x |

Subsequent node02 / normal_h100 pilots:

- **17513**, `wide_layout_ln.py`: direct channel-major shared reads remove
  output-LN transposes. Strict/dTri bitwise passed. D384 short LN fell
  0.742 → 0.512 ms; D512 gains were small.
- **17516**: N128 dW instructions combined with prefetch passed bitwise
  derivatives/partials but yielded only small full-workload gains.
- **17519**, `wide_tile_ln.py`: 16/32-row TMA tiles and TMA dNorm loads
  remove output-LN spills. D384 short LN 0.742 → 0.396 ms, full
  6.370 → 6.026 ms; D512 0.787 → 0.495 ms, full 10.052 → 9.733 ms.
  Selected rows: 32 at D384, 16 at D512, 128 threads, no compiler fences.
- **17524**, `wide_dense_output.py`: keep exact forward LN and epilogue,
  use dense GEMMs between them, and reuse the saved-product allocations.
  All four strict pilots passed. D384 full 6.397 → 5.882 /
  25.526 → 23.143 ms with 128 threads. D512 selected 256 threads.
- **17542**: unroll D512's standalone packed LN without changing its sum
  order. Unroll16/256 threads passed both lengths; output stage
  1.287 → 1.009 / 5.112 → 3.963 ms versus the initial dense candidate.
- **17533**: channel-owned affine sums were strict but slower than the
  TMA row-warp LN; rejected.
- **17539**, `wide_tma_input.py`: TMA x/dXn/residual, CTA affine aggregate
  and exact-order weight reduction. All short-shape strict pilots passed.
  Selected 16 rows/128 threads/minblocks4: D384 input stage
  0.428 → 0.221 ms, D512 0.585 → 0.317 ms.
- **17548**, `wide_saved_front.py`: preactivation saves plus exact 32-split
  cuBLASLt dW pass strict/bitwise dX and weights at all four shapes, but
  extra forward stores offset the source savings. Full times regress;
  rejected, no additional activation allocation in selected candidates.
- **17554** applies dense forward and TMA output LN to D256. Dense128
  alone passes both strict pilots and gives full 3.068 → 2.847 /
  12.848 → 11.723 ms. TMA output LN adds no useful D256 gain; excluded.

`wide_checkpoint4.py` combines dense output, TMA output LN and TMA input
LN with checkpoint3. Full validation is **17558**, direct paired Triton
comparison **17560** depends on its success. `d256_dense_checkpoint.py`
keeps the selected D256 backward and changes only the output schedule;
validation **17563**, successor direct comparison tracked in Slurm.
All numbers above are preceding-CUDA vs candidate pilots unless explicitly
labeled Triton. They are not a 1.5x claim. No production dispatch changes.

## Latest validated checkpoint: saved input stats and output products

Construct `selected_current.Training` with the configuration below, then call
`shared_candidate.attach(plan)` with `SHARED_CANDIDATE=saved_products`:

```text
GP_OFF=1 GP_WARP=1 DX_LN=1 DX_WARP=1 SAVE_NORM=1
DX_N256=0 DX_PIPE=0 DX_SPLIT=0 DX_IN_STATS=0 DX_GAMMA_CACHE=0
LN_THREADS=128 LN_MINBLOCKS=3 LN_AGG=1 LN_CACHE=0
LN_STATS_TMA=0 LN_STORE_READ=0 LN_GAMMA_SMEM=0 DN_SLIM=0
SAVE_EPI_THREADS=256 SAVE_EPI_GRID=1056
```

Attachment enables input-LN FP32 statistics saves, dX shared gamma caching,
and output projection/gate BF16 saves. The backward preparation stage skips
two GEMMs and uses a 1,056-CTA pointwise epilogue. The optimized forward
math and output are preserved. Input statistics add 1.125 / 4.5 MiB at
L384 / L768. Saved output products reuse existing allocations but extend
144 / 576 MiB of tensor lifetimes. There are no production dispatch edits.

Array **17255**, node02 / normal_h100, compares the new checkpoint directly
against the preceding saved-norm/LN128 checkpoint on the same buffers:

| L | Forward before → new | Backward before → new | F+B before → new | Fixed SOL model |
|---:|---:|---:|---:|---:|
| 384 | 1.043 → 1.087 ms | 2.111 → 1.957 ms | 3.145 → 3.045 ms | 36.1% → 38.9% |
| 768 | 4.347 → 4.524 ms | 8.701 → 8.136 ms | 12.950 → 12.613 ms | 40.4% → 43.2% |

Backward time falls **7.29% / 6.49%**; F+B falls **3.19% / 2.60%**.
Results: `result-checkpoint-saves-L384-17256.json` and
`result-checkpoint-saves-L768-17255.json`. The original recomputing-policy
SOL model is intentionally fixed for comparison. Saving products removes
`6*M*D^2` actually executed backward FLOPs; the reported ratio is **not**
a hardware-utilization measurement. SOL90 remains unmet.

Array **17250** passes all strict stress fixtures, CUDA graph replay and
independent PyTorch comparisons at both lengths. Memcheck and racecheck
both report zero errors/hazards/warnings. Follow-up array **17272** also
passes synccheck with zero errors at both lengths. Evidence:
`validation-saved-products-g1056-node02-L*.json`,
`memcheck-saved-products-g1056-node02-L*.log`,
`racecheck-saved-products-g1056-node02-L*.log`,
`synccheck-saved-products-g1056-node02-L*.log`.
Standard dX differs from the preceding checkpoint by 5.34e-6 / 6.93e-6
relative L2 (limit 2e-5); all weight gradients match bit for bit in the
direct comparison. The input-statistics reduction explains the dX change.

NCU job **17260**, L384 (`node02-saved-products-g1056-L384.*`), verifies
the active kernels: pointwise epilogue 126.816 us, output LN 219.680 us,
source/dW 534.784 us and dX/LN 477.568 us. Source/dW reads/writes
402.0 / 601.0 MB; dX/LN 858.4 / 74.1 MB. These profile durations are
diagnostics, not replacements for paired graph timings.

The checkpoint's exact configuration, cubins, validation records and paired
timings are collected in `checkpoint-saved-products-node02.json`.

### Speedup over existing Triton

Array **17323** compares each width against the existing Triton implementation
on node02 / normal_h100. Each length uses one process and the same GPU, with
150 alternating CUDA graph samples per scope. Backward is measured directly.
The baseline uses the existing heuristic-24 configuration, fixed 25% dropout,
BF16 B1 bidirectional TriMul and H=2D. Speedup is Triton time / our time.

| D | L | Backward Triton → ours (ms) | Backward speedup | Forward speedup | F+B speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 384 | 1.089 → 0.715 | 1.523x | 1.769x | 1.606x |
| 256 | 384 | 2.212 → 1.954 | 1.132x | 1.219x | 1.164x |
| 128 | 768 | 4.342 → 2.874 | 1.511x | 1.805x | 1.605x |
| 256 | 768 | 12.192 → 8.066 | 1.512x | 1.214x | 1.413x |

Backward time reductions at L384 / L768 are **34.35% / 33.80%** for D128
and **11.65% / 33.84%** for D256. These gains use the common Triton baseline;
the 7.29% / 6.49% above instead compare successive D256 CUDA checkpoints.
D128 retains its validated B1/B7 cubin hashes. Output and gradient comparisons
against Triton pass at both widths and lengths, as do the graph replay checks.
Retained backward graphs disable AOT buffer donation and capture on their
saved-forward stream; replay gradient error versus full Triton is below 2e-6.
Our kernel's separate strict validation contract remains unchanged.

Evidence: [L384](baseline-D128-D256-L384-17324.json) and
[L768](baseline-D128-D256-L768-17323.json).

### Follow-up toward 1.5x over Triton (2026-09-25)

The active target covers D256/384/512 at L384/768. Do not infer completion
from D256 L768 backward alone: D256 L384 backward and both D256 F+B scopes
are below 1.5x, and D384/512 have not received the D256 checkpoint's changes.
All experiments below use node02 / normal_h100, one GPU at a time.

`saved_front_products.py` saves K1's BF16 projection/gate preactivations and
replaces B7's recomputing GEMM with a TMA load. Array **17347** passes standard
strict gradients, poison coverage and graph replay at both lengths; output,
derivatives, dX and weight gradients are bitwise equal. It adds 576 / 2304 MiB
of saved tensors, however, and is rejected on paired complete-workload timing:

| L | Backward control → saved preactivations (ms) | F+B control → candidate (ms) |
|---:|---:|---:|
| 384 | 1.934 → 2.014 | 2.980 → 3.538 |
| 768 | 8.033 → 8.277 | 12.351 → 14.499 |

Evidence: `result-saved-front-products-L384-17348.json` and
`result-saved-front-products-L768-17347.json`. Neither saved-front candidate
was promoted to full sanitizer validation.

`cute_contract.py` tests eight batched GEMM schedules with the existing CuTe
SM90 implementation. Array **17349** finds seven supported schedules with
strict gradients at each length; 128x256 ping-pong is unsupported and is
recorded as a compile exception. All L384 candidates regress. The 128x128,
two-CTA M cluster reduces L768 backward 8.279 → 8.224 ms and F+B
13.137 → 13.047 ms in its paired comparison. This small pilot result remains
unselected and has no full stress/sanitizer qualification.

`lt_contract.py` queries the installed cuBLASLt heuristics and measures each
of the eight returned algorithms per contraction, selecting only bitwise
matching results. Array **17351** is effectively unchanged at L384; L768
contractions fall 1.560 → 1.442 ms, backward 8.228 → 8.146 ms, and F+B
13.055 → 12.977 ms. These are comparisons with the preceding CUDA checkpoint,
not the original Triton baseline. The explicit Lt runner remains a pilot.

`wide_hybrid_b1.py` extends the D256 cuBLAS GEMMs and native TMA output-LN
schedule to D384/512. Array **17353** checks the two widths and both lengths
against their preceding B1, preserving our existing forward and B7. The
per-shape records are `result-wide-hybrid-D*-L*-173*.json`; compilation or
an isolated B1 gain alone does not establish a validated 1.5x full workload.

Both D384 standard comparisons pass, with bitwise dX, input weight gradients
and all four B1 intermediate tensors. Array **17359** then passes all five
stress cases, graph replay and independent PyTorch comparison at both lengths,
plus memcheck, racecheck and synccheck with zero errors/hazards/warnings.
The largest D384 stress dWproj error is 4.84e-4, below the unchanged 5e-4 limit.

D512 initially fails because its preceding B1 uses ex2/rcp sigmoid, while
D384's preceding streamed B1 uses tanh. Setting `TMN_SIGMOID_TANH=0` in 17357
does not disable an `#ifdef` branch. Removing the define in **17361** restores
bitwise dp/dg/dTri/dX and input weight gradients. D512 L768 dWproj still differs
by 5.68e-4, so `ExactProjectionWeight` retains the preceding native 32-way
FP32 split-K accumulation for that shape. Job **17366** passes strict gradients
with bitwise dWproj. D512 L384 retains cuBLAS dWproj after its strict pass.

Paired pilot timings against the preceding CUDA B1, in milliseconds:

| D | L | B1 before → candidate | Backward before → candidate | F+B before → candidate |
|---:|---:|---:|---:|---:|
| 384 | 384 | 3.099 → 1.754 | 7.432 → 6.089 | 9.206 → 7.873 |
| 384 | 768 | 12.751 → 6.639 | 30.468 → 24.420 | 37.819 → 31.744 |
| 512 | 384 | 4.488 → 2.281 | 11.297 → 9.079 | 14.178 → 11.980 |
| 512 | 768 | 19.472 → 10.292 | 48.052 → 38.633 | 59.652 → 50.463 |

These are **not Triton speedups**. D512 stress and all three sanitizer tools
run in array **17370**; Triton comparison array **17371** depends on its success.
D384 Triton comparison **17363** fails during graph capture because the
autograd stream differs from the stream used to create its leaf graph nodes;
no timings from it are accepted. **17367** clones the baseline leaves and
initializes/captures the complete Triton path on the same dedicated stream,
with an explicit graph-gradient comparison. L384 F+B is 6.178 ms Triton versus
7.998 ms CUDA (**0.772x**); L768 is 27.459 versus 32.395 ms (**0.848x**).
Thus D384 remains below the target despite the B1 gain. Both records from
17367 are complete and retain the Triton replay-gradient checks.
The unchanged wide B7 is the next large remaining stage; no engine dispatch
or autograd registration is modified.

D512 array **17370** subsequently passes both lengths' five stress cases,
graph replay, independent PyTorch and all three sanitizers, with zero
errors/hazards/warnings. The records include cubin and source SHA-256 values.
Paired Triton array **17371** is complete: L384 full is 9.312 ms Triton /
12.074 ms B1-optimized CUDA (**0.771x**); L768 is 41.119 / 51.939 ms
(**0.792x**). These results establish that B1 alone is insufficient.

### Wide B7 derivative/dW reuse

`wide_source.cu` computes each 32-channel projection/gate pair once, publishes
the derivatives with TMA, and immediately consumes the same shared derivative
tile in two dW warpgroups. The two groups partition input-weight columns.
It retains the existing 32 split-K intervals and FP32 reduction order. Shared
storage is 163,968 / 213,120 bytes at D384 / D512. `wide_b7.py` follows it with
the existing dX accumulation and input-LN/reduction code in a separate kernel.
The full derivative workspace is still needed by dX; it is not eliminated.

Array **17374** passes standard strict gradients, poisoned-buffer checks and
graph replay at all four shapes. Derivative planes, dXn, dX and all weight
gradients match bit for bit. Compared with the qualified B1-only checkpoint:

| D | L | Backward before → new B7 (ms) | F+B before → new B7 (ms) |
|---:|---:|---:|---:|
| 384 | 384 | 6.241 → 5.412 | 8.089 → 7.261 |
| 384 | 768 | 25.117 → 21.425 | 32.593 → 28.887 |
| 512 | 384 | 9.199 → 8.272 | 12.207 → 11.268 |
| 512 | 768 | 38.755 → 34.217 | 51.141 → 46.467 |

These are pilot comparisons against CUDA, not validated Triton speedups.
Full stress and sanitizers for this B7 are still required before selection.
Array **17378** tests dX N128 with one/two warpgroups and N64 with one group.
All preserve strict gradients; D384 rejects all three schedules. D512's
N128/two-group finish improves backward 8.230 → 7.945 / 33.778 → 32.896 ms
and full 11.189 → 10.927 / 46.135 → 45.114 ms. It remains a pilot.

`prefix_dx.py` next builds a single dX GEMM whose K dimension starts with
the output-gate derivatives, followed by the four input projection/gate
derivative planes in their original accumulation order. The source publishes
directly into that suffix, so only the D-channel gate prefix is copied;
weights are packed from their live tensors. This differs from the earlier
rejected CuTe dX that accumulated a separate FP32 gate GEMM in the epilogue.
Array **17387** starts with L384 at D256/384/512. Strict gradients, not algebraic
equivalence alone, determine whether this accumulation change is acceptable.

### Prefix transpose and combined wide checkpoint

Arrays **17387** (CuTe) and **17391** (cuBLAS) establish strict correctness
for the single ordered dX GEMM. The original prefix `torch.copy_` is too slow:
phase diagnostics **17403** measure 4.561 / 9.528 ms for the copy alone at
D256 / D512 L768, versus 1.045 / 3.549 ms for the respective GEMM.
`prefix_transpose.cu` replaces it with 64x64 TMA tiles and shared-memory
ldmatrix/stmatrix transposes. Array **17408** passes strict gradients and
bitwise prefix checks at all six shapes. Paired times, relative to the
preceding CUDA path (not Triton), are:

| D | L | Backward control -> TMA prefix (ms) | F+B control -> candidate (ms) |
|---:|---:|---:|---:|
| 256 | 384 | 1.943 -> 2.066 | 3.014 -> 3.133 |
| 256 | 768 | 8.130 -> 8.476 | 12.659 -> 12.975 |
| 384 | 384 | 5.390 -> 4.944 | 7.210 -> 6.748 |
| 384 | 768 | 21.373 -> 19.435 | 28.817 -> 26.855 |
| 512 | 384 | 8.269 -> 7.473 | 11.209 -> 10.394 |
| 512 | 768 | 33.753 -> 30.479 | 46.149 -> 42.642 |

The D256 candidate is rejected; its qualified saved-products/fused-dX-LN
checkpoint remains unchanged. The wide variants advance to qualification.

Output-normalization saves in **17399** are bitwise exact at D384 and improve
F+B 7.101 -> 6.951 / 28.594 -> 28.019 ms. D512 needs an extra legacy-order LN
calculation to preserve its strict arithmetic. **17406** passes that exact
save check but regresses F+B 10.988 -> 11.263 / 45.395 -> 46.303 ms; reject it.

`wide_combined.py` combines WideB1, the new derivative/dW source, the ordered
cuBLAS dX GEMM with TMA prefix copy, and D384-only output-normalization saves.
Array **17414** validates all four wide shapes, including poisoned saved
buffers, five stress cases, graph replay, independent PyTorch, and memcheck,
racecheck and synccheck. It records source and cubin hashes. Its sanitizer
filter includes every new native kernel, including the forward save and
prefix transpose. The D256 path is not changed.

Array **17417** separately tests large cuBLAS input preactivation/dW GEMMs.
Array **17419** compares the combined checkpoint directly with existing
Triton in forward, backward and F+B; it requires matching qualified cubin
hashes and successful sanitizer logs. Backward is measured directly with
retained forward tensors on their original CUDA stream. Job **17421** then
tests D512 L768 dWproj using 32 FP32 batched GEMMs and the native split-order
reduction. These job handles are authoritative; pending work is not a pass
or a measured speedup. No production dispatch/autograd changes are made.

Array **17414** is now complete: every shape passes all five strict stress
cases, poisoned-buffer and graph checks, independent PyTorch, and all three
sanitizers with zero errors/hazards/warnings. The four validation records are
`validation-wide-combined-D384-L384-17415.json`, D384/L768-17416,
D512/L384-17418 and D512/L768-17414; sanitizer logs use array ID 17414.

Array **17417** rejects the all-cuBLAS input-source alternative. Derivatives
and dX match bitwise at every shape, but dW relative errors exceed 5e-4 for
D384 L768 (up to 8.79e-4) and both D512 lengths (up to 6.42e-4 / 1.293e-3).
D384 L384 passes strict but F+B regresses 6.703 -> 6.921 ms. Its derivative
GEMM plus pointwise stage is 1.398 ms, and its dW GEMM is 0.489 ms, versus
1.654 ms for the native fused source/dW. No threshold is relaxed.

Array **17428** tests forward product saves: D384 saves normalized triangle,
projection and gate; D512 saves only gate because the packed forward LN has
a different strict arithmetic order. Each shared output tile is fully drained
before the next column group reuses it. These candidates remain pilots.

Paired Triton array **17419** is complete for the qualified combined path:

| D | L | Backward Triton -> ours (ms) | BWD speedup | FWD speedup | F+B speedup |
|---:|---:|---:|---:|---:|---:|
| 384 | 384 | 3.942 -> 4.813 | 0.819x | 1.191x | 0.931x |
| 384 | 768 | 18.009 -> 18.955 | 0.950x | 1.245x | 1.037x |
| 512 | 384 | 5.764 -> 7.684 | 0.750x | 1.185x | 0.878x |
| 512 | 768 | 26.534 -> 30.858 | 0.860x | 1.258x | 0.950x |

All four targets remain below 1.5x. These are direct paired comparisons,
not ratios assembled from different jobs or improvements over old CUDA.

Job **17421** preserves all 32 native dWproj FP32 partials bit for bit with
cuBLASLt batched GEMM at D512 L768. Selected heuristic index 0 reduces that
stage 2.141 -> 0.876 ms and paired F+B 43.115 -> 41.835 ms. It remains a pilot
pending combined stress/sanitizer checks. Array **17428** also passes standard
strict checks with bitwise saved products; F+B improves 6.593 -> 6.492 /
26.196 -> 25.905 ms for D384 and 10.413 -> 10.361 / 42.823 -> 42.564 ms for
D512. These timings compare CUDA candidates, not Triton.

Array **17435** terminated before kernel validation because the FP32 partial
buffer was sliced along its split dimension instead of flattened. **17439**
corrects that host view and passes bitwise partial/dW checks at all shapes.
The native derivative-only kernel plus separate batched input dW is rejected:
F+B regresses 6.610 -> 6.818 / 26.197 -> 27.134 ms at D384 and
42.900 -> 43.637 ms at D512 L768; D512 L384 is effectively unchanged.

Component diagnostics **17431** identify output LN at 0.739 ms for D384 L384
and 3.171 ms for D512 L768. The D384 original LN cubin uses 255 registers
with a 256-byte stack. **17443** tests recomputing channel values to reduce
the live register set; the first version still spills and has no meaningful
paired-length win. **17447** adds a scalar reduction loop and moves affine
updates to the write pass while retaining their per-row accumulation order.
These LN candidates require strict, full-workload and sanitizer evidence.

`wide_checkpoint2.py` combines only the beneficial forward product saves
with the bitwise FP32 dWproj split at D512 L768. It retains the qualified
native fused input source and original output LN; the split-input-source and
bounded-LN experiments are not included. Array **17451** runs five-case
strict/graph/PyTorch validation and all three sanitizers. Memcheck covers the
entire path, including the cuBLASLt descriptors; racecheck/synccheck cover
all changed native kernels. Array **17452** depends on successful completion
of 17451 and compares all four wide shapes with the existing Triton forward,
backward and F+B. It verifies both cubin hashes and the selected Lt algorithm
against the matching validation record. These remain experimental paths;
the D256 selected-current checkpoint and production dispatch are unchanged.

Array **17451** is complete: checkpoint2 passes all four shapes' five strict
stress cases, poisoned saved products, graph replay, independent PyTorch and
all three sanitizers with zero errors/hazards/warnings. The validator records
its Lt algorithm as well as source/cubin hashes. Paired Triton array **17452**
then measures this exact checkpoint; pending shape records are not results.

`wide_shared_affine_ln.py` moves only the persistent gamma/beta gradient
accumulators to warp-owned shared regions. It preserves per-warp channel and
row sum order, uses no shared-memory atomics, and reduces across warps at the
end. This adds 2*(threads/32)*H*4 shared bytes and reduces register residency;
array **17456** compares 128 and 256 threads at both widths and lengths.

`wide_pipe_source.cu` uses three warpgroups: one producer recomputes projection
and GLU, while two consumers retain their original dW column partitions and
32 split-K intervals. Two input/derivative slots are protected by ready and
credit barriers. Both consumers must finish WGMMA before the producer can
reuse the slot; global derivative TMA stores are drained too. Extra shared
storage is 8 KiB versus the preceding source. Array **17457** verifies
poisoned GP and FP32 partials bit for bit before full-gradient and timing
checks. Job **17461** profiles the qualified D512 L384 source, output LN and
input LN using a limited set of tensor, DRAM and warp-stall metrics. All are
serial on node02/normal_h100, and no candidate is installed from pilot data.

### Qualified checkpoint2 and native follow-ups

Array **17452** completes paired existing-Triton timing for checkpoint2:

| D | L | Backward Triton -> ours (ms) | BWD speedup | F+B speedup |
|---:|---:|---:|---:|---:|
| 384 | 384 | 3.933 -> 4.538 | 0.867x | 0.947x |
| 384 | 768 | 18.008 -> 17.953 | 1.003x | 1.055x |
| 512 | 384 | 5.780 -> 7.550 | 0.766x | 0.887x |
| 512 | 768 | 26.374 -> 29.230 | 0.902x | 0.979x |

These values remain below the 1.5x goal. D256 remains at its qualified
saved-products checkpoint; none of these wider experiments changes it.

**17456** rejects the shared-affine output-LN variants at all four shapes.
They preserve strict gradients and bitwise dTri but retain some register
spills and increase complete-workload time. **17457** similarly rejects the
first three-warpgroup source despite bitwise GP/FP32 partials: its input TMA
load was not overlapped with the current projection.

**17465** applies D256's immediate-offset WGMMA descriptors to the ordinary
wide source. It is bitwise exact, with roughly 1% source gain at D384 and no
meaningful D512 gain. **17472** combines these descriptors with the three-WG
schedule and issues the next input load after current WGMMA commit, waiting
for the old slot's two consumer credits before reuse. Every shape passes
bitwise GP/FP32 partials and strict gradients. Paired CUDA F+B improves:

| D | L | Checkpoint2 -> prefetched source (ms) |
|---:|---:|---:|
| 384 | 384 | 6.490 -> 6.416 |
| 384 | 768 | 25.932 -> 25.395 |
| 512 | 384 | 10.417 -> 10.248 |
| 512 | 768 | 41.657 -> 40.572 |

Profiling job **17461** collected no kernels because ncu uses `regex:` while
compute-sanitizer uses `regex=`. The local installed help confirms both
syntaxes. It also hit a Python user-site encoding error on report import.
**17476** corrects the ncu filter and sets PYTHONNOUSERSITE=1, then succeeds
with three passes for each of the three intended kernels. At D512 L384:
source has 49.7% tensor-pipe activity, 898 MB DRAM reads / 1.458 GB writes;
output LN reads 673 MB / writes 321 MB; input LN plus dW reduction reads
721 MB / writes 157 MB. Input LN/reduction long-scoreboard stalls are
16.43 per issue-active cycle. These profiler counters are not Triton
speedups or a measured SOL90 result.

Array **17477** ports CTA affine aggregation to the wide input LN/reduction.
All configurations pass strict gradients. D384 has only a small/inconsistent
complete-workload benefit and retains its original reduction. D512 selects
256 threads with launch-bounds minblocks=3: reduction is 0.719 -> 0.582 /
2.584 -> 2.044 ms, with F+B 10.284 -> 10.134 / 40.995 -> 40.707 ms.
Array **17489** tests paired N64 dW accumulators as N128 WGMMA instructions
with unchanged K order; this remains a separate pilot and is not included
in the next checkpoint.

`wide_checkpoint3.py` combines checkpoint2, the prefetched immediate-offset
N64 source at all wide shapes, and the selected input reduction only at D512.
Array **17496** performs the full stress/graph/PyTorch and sanitizer gates;
its successor directly remeasures Triton only after those gates succeed.
No production dispatch or autograd registration is changed.

### Matched D128 versus D256 comparison

Array **17297** runs both widths on the same node02 H100 within each length,
under normal_h100 QoS. `compare_d128_d256.py` alternates 150 CUDA graph samples
per forward, backward and F+B scope. Backward is timed directly, not inferred
by subtracting medians. D128 is the strict-corrected single-B7/cache-policy-B1
selection; both B1/B7 cubin SHA-256 values match its recorded validated
selection. D256 is saved_products above. Eager/graph outputs are finite and
agree within 5e-6 at both widths; D128 also passes its strict regression check.
The older D128 module names are resolved locally in this comparison script;
no engine module or dispatch is modified.

| L | Scope | D128 ms | D256 ms | D256 / D128 |
|---:|---|---:|---:|---:|
| 384 | Forward | 0.306512 | 1.078816 | 3.520x |
| 384 | Backward | 0.715392 | 1.946128 | 2.720x |
| 384 | F+B | 1.017824 | 3.021008 | 2.968x |
| 768 | Forward | 1.230784 | 4.530160 | 3.681x |
| 768 | Backward | 2.914048 | 8.194608 | 2.812x |
| 768 | F+B | 4.154176 | 12.735264 | 3.066x |

Using the same fixed recomputing-policy SOL formula, D128 / D256 are
30.72% / 39.14% at L384 and 38.21% / 42.90% at L768. The modeled backward
FLOP ratios are 3.467x / 3.158x. D256 actually removes `6*M*D^2` FLOPs by
saving output products; these fixed-model ratios are not measured hardware
utilization. Evidence: `comparison-D128-D256-L384-17298.json` and
`comparison-D128-D256-L768-17297.json`.

### Follow-up dX and output-LN candidates (rejected)

`dx_separate.cu` / `dx_separate.py` split D256 dX into N128 CTAs followed
by input LN, preserving the GEMM accumulation order. The initial 256-thread
three-CTA producer/consumer build (17263) cannot compile within its
80-register target; no GPU result exists for it. The 128-thread path
(17265) has zero spills and passes strict gradients plus NaN coverage checks,
but its separate input LN costs 313 / 1,219 us. Two-slot complete backward
is 1.946 → 2.154 / 8.011 → 9.061 ms. Three slots are also slower.

`dx_cluster_ln.py` / `dx_cluster_ln.cuh` then keep each CTA's half dX in
shared memory, exchanging only two FP32 row sums through DSM to finish LN.
There is no dXn HBM intermediate. A host tensor-map rank assertion (17268)
was corrected before any kernel measurement. Array **17270** passes strict
gradients and complete dX coverage for both two/three-slot versions, with
zero spills. The altered LN reduction gives dX relative error 5.00e-6 /
5.27e-6 against saved_products. Two slots use 264 resident two-CTA clusters;
nevertheless backward is 1.964 → 2.003 / 8.220 → 8.280 ms and F+B
3.056 → 3.099 / 12.815 → 12.881 ms. Three slots are slower still.
Neither separated nor DSM dX path is selected or fully sanitizer-qualified.

`LN_GAMMA_SMEM=1` caches output-LN gamma in 2 KiB of shared memory without
changing three-CTA occupancy. Array **17274** gives bitwise dt, dX and weight
gradients, passes NaN coverage, and has zero spills. It is still slower:
backward 1.955 → 1.966 / 8.165 → 8.250 ms and F+B 3.037 → 3.049 /
12.695 → 12.792 ms. Keep `LN_GAMMA_SMEM=0` for the checkpoint.

## Previous validated checkpoint: saves and three-CTA output LN

Use `GP_OFF=1 GP_WARP=1 DX_LN=1 DX_WARP=1 SAVE_NORM=1 LN_THREADS=128
LN_MINBLOCKS=3 LN_AGG=1 LN_CACHE=0` with `selected_current.Training`.
This combines forward normalization saves with 128-thread output LN CTAs,
three resident CTAs per SM, and shared-memory affine-gradient reduction.

Same-job paired graph timings against the saved-norm checkpoint below:

| L | Backward before → new LN | Forward+backward before → new LN |
|---:|---:|---:|
| 384 | 2.174 → 2.110 ms | 3.205 → 3.141 ms |
| 768 | 8.939 → 8.659 ms | 13.159 → 12.931 ms |

Jobs 17068/17073 show another 2.9% / 3.1% backward reduction and 2.0% / 1.7%
full-workload reduction. Forward output, dX and all weight gradients exactly
match the saved-norm checkpoint; LN affine gradients satisfy the strict limit.
Validation array 17074 passes all stress, graph-replay and independent PyTorch
checks at both lengths. Both memcheck and racecheck report zero errors or
hazards: `validation-savednorm-ln128-L*.json`,
`memcheck-savednorm-ln128-L*.log`, `racecheck-savednorm-ln128-L*.log`.
New jobs use node02 / normal_h100 QoS and one GPU at a time.

The unchanged SOL model gives **36.1% / 40.6%**, below the unchanged SOL90
objective. No automatic engine dispatch or autograd registration is changed.

Historical resource audit on 2026-09-25: node01 had `AllocTRES ... gres/gpu=8` out of
eight GPUs. Across three consecutive goal turns, 17170 remained PENDING
with reason Resources; 17174, 17176, 17180 and 17181 depend on that serial
chain and remain PENDING. The scheduler estimates 17170 at September 27
02:22:13, subject to change. CPU compilation and validation wiring are done;
the next decision requires actual candidate timings and gradients. The goal
is blocked on this external resource condition, not complete. No queued job
is cancelled or resubmitted, and node01 / normal_h100 remains mandatory.
On resumption, inspect these same job handles and their result files first,
then fully validate any winning candidate at both lengths before selection.

User-authorized node02 resumption: the five jobs were confirmed PENDING,
then their `ReqNodeList` was changed to node02 with `scontrol update`.
Job IDs, dependencies and `normal_h100` QoS are preserved; no duplicate jobs
were submitted. `run.sbatch` and `sanitize_current.sbatch` now default to
node02. The older node01-only instruction below is historical.

All five jobs completed on node02. The four shared-operand/N256 candidates
pass standard strict gradients at both lengths, with bitwise dX and weight
gradients, but all are slower. Same-process graph timings in milliseconds:

| Candidate | L384 backward control → candidate | L768 backward control → candidate |
|---|---:|---:|
| Source rank pairs (17174 / 17180) | 2.112 → 2.127 | 8.764 → 8.813 |
| dX N256 (17176 / 17181) | 2.084 → 2.425 | 8.536 → 10.043 |
| dX row pairs, N128 (17176 / 17181) | 2.084 → 2.165 | 8.536 → 8.945 |
| dX row pairs, N256 (17176 / 17181) | 2.084 → 2.161 | 8.536 → 8.874 |

Forward+backward is also slower for every candidate. Job 17170 independently
measures N128 row pairs at 2.089 → 2.170 ms backward. These candidates remain
rejected and are not promoted to full stress/sanitizer validation. N256's
single-consumer dX stage is 870 versus 531 us at L384, and 3.517 versus
2.037 ms at L768; fewer WGMMA instructions did not offset the changed
occupancy. Source-pair derivative planes are bitwise identical.

`DX_PIPE=1` is the next isolated dX experiment. It preserves N128, the two
shared stages, the CTA launch bound and original K accumulation order, but
keeps two WGMMA groups in flight. The previous slot is released only after
`wait_group 1`; the final slot and accumulators are drained with `wait_group 0`
before LN reuses shared storage. Accumulator register fences bracket the
whole matrix-product loop. Job array 17195 tests both lengths serially on
node02 / normal_h100 via `check_dx_pipeline.py`. Validation selection uses
`SHARED_CANDIDATE=dx_pipe` with initial `DX_PIPE=0 DX_N256=0`.

Node02 follow-ups (all standard strict gradients pass unless specified):

- Array 17195 rejects the rolled two-group pipeline: backward 2.088 → 2.229
  ms / 8.636 → 9.234 ms. Array 17200 unrolls the 36 K steps. Synchronous
  unrolling is only 2.100 → 2.091 / 8.721 → 8.716 ms; unrolled asynchronous
  execution is still slower. Neither is selected. Array 17209 removes only
  the repeated matrix-product CTA barriers while retaining TMA acquire,
  WGMMA completion and all LN barriers. The synchronous candidate gives
  2.101 → 2.093 / 8.743 → 8.704 ms backward, but L768 full workload regresses
  13.048 → 13.053 ms; it is not selected. All these builds have zero spills.
- NCU job 17202 (`node02-baseline-L384.*`) confirms source 538.560 us with
  63.5% active tensor cycles and dX/LN 514.816 us with 35.1%. Source reads /
  writes 399.1 / 601.0 MB; dX/LN 856.8 / 74.4 MB. Output LN is 218.752 us.
  These are profiler observations, not replacement benchmark timings.
- `dx_split.cu` uses 384 threads: one producer, two half-width N64-pair
  consumers and joint input LN. The first 32/112/112 register allocation
  (17214) deadlocks because PTXAS assigns 80 registers/thread, giving a
  30,720-register CTA pool while the roles request 32,768. The known-invalid
  array was cancelled. NVIDIA's PTX `setmaxnreg` semantics define a per-CTA
  pool; SM-wide free registers cannot fill that deficit. The corrected
  32/104/104 allocation exactly fits, with a host attribute check before
  launch. Array 17216 passes strict gradients with bitwise dX, zero spills
  and two CTAs/SM, but gives only 2.114 → 2.098 / 8.854 → 8.818 ms backward.
  Array 17218 repeats the small gain and rejects a shared-code-path variant,
  which is slower at both lengths. These remain experimental.

`saved_input_stats.py` enables the existing K1 SAVE template parameter in
our forward and reuses its FP32 mean/reciprocal-standard-deviation in dX/LN.
It adds 8 bytes per pair row (1.125 / 4.5 MiB at L384 / L768), leaves forward
output bitwise identical and keeps weight gradients bitwise identical.
Array 17220 passes the standard strict test: dX relative L2 5.34e-6 / 6.93e-6,
below 2e-5. Same-process paired results are:

| L | Backward control → input stats | Full control → input stats |
|---:|---:|---:|
| 384 | 2.096 → 2.063 ms | 3.122 → 3.088 ms |
| 768 | 8.723 → 8.588 ms | 13.018 → 12.917 ms |

The forward cost is included (1.029 → 1.027 / 4.360 → 4.361 ms). Array 17222
passes full stress, graph replay and independent PyTorch at both lengths.
The worst strict stress dX relative error is 8.35e-6 / 7.40e-6, graph error
is below 5e-7, and PyTorch error is below 0.00656. Both memcheck and racecheck
report zero errors/hazards/warnings (`*-input-stats-node02-L*.log`). This is
a new fully validated explicit checkpoint. Select it via `SHARED_CANDIDATE=input_stats`;
`DX_IN_STATS=0` during initial plan construction. The original checkpoint
and production dispatch are unchanged.

Array 17225 compares an additional 1 KiB shared gamma cache directly against
that input-stats checkpoint. Backward is 2.070 → 2.056 / 8.626 → 8.569 ms;
full workload is 3.099 → 3.087 / 12.963 → 12.914 ms. Standard dX and weight
gradients are bitwise identical. Array 17228 runs the full validation and
sanitizers before selection of `input_stats_gamma`.

`saved_products.py` reuses K3 projection/gate BF16 outputs to eliminate the
two backward preparation GEMMs. `SavedNormOutput(products=...)` stores both
tiles via TMA after their existing products finish, reusing the weight-ring
shared memory and the preallocated preparation buffers. Only the saved
tensor lifetimes are extended, by 144 / 576 MiB. The remaining preparation
kernel evaluates exactly the existing BF16-rounded pointwise gate gradients.
Array 17230 passes standard gradients at both lengths: saved gate, dp, dg,
dX and weight gradients are all bitwise identical to input_stats_gamma.
Its initial timing overlapped array 17228 because the submitted dependency
used 17227 instead of the returned 17228. Correcting the dependency found
that the tasks had already started/finished. These timings are preliminary
and must not be used for acceptance; a serial rerun after 17228 is queued.

Array 17228 subsequently passes both lengths' full validation and both
sanitizers for `input_stats_gamma`. The serial products rerun, 17233, passes
all standard checks and measures backward 2.052 → 1.971 / 8.563 → 8.210 ms,
full 3.076 → 3.052 / 12.814 → 12.712 ms versus input_stats_gamma.

Array 17236's 128-thread epilogue results are invalid: the old epilogue has
a hardcoded 256-thread stride, so half of dp/dg was left unchanged. Identical
fixtures hid this behind stale valid values. Those timings are moved to
`invalid_times` in the JSON records and explicitly marked invalid; raw logs
are retained. The 256-thread measurements remain valid. `SavedProductsPrepare`
now compiles a blockDim-based variant when 128 threads are requested, and
`check_saved_epi.py` poisons dp/dg before checking each candidate. Array 17242
confirms the old missing half and passes poison-coverage/gradient checks for
every corrected setting. The common 256-thread / 1,056-CTA choice gives
epilogue 156.544 → 129.856 / 622.096 → 522.336 us and backward
1.982 → 1.957 / 8.255 → 8.182 ms, full 3.061 → 3.041 / 12.796 → 12.702 ms
against the original 528-CTA saved-products epilogue. It is the preferred
setting for further validation (`SAVE_EPI_THREADS=256 SAVE_EPI_GRID=1056`).

`fused_saved_b1.py` then combines saved-product pointwise gradients, dNorm
and output LN. It reuses the 64 KiB dNorm output area for initial projection /
gate input and the 32 KiB weight area for dy/dp. After publishing dp/dg for
weight gradients, dp remains in register fragments for dNorm; intermediate
dNorm never reaches HBM. Array 17245 rejects the first build on strict
gradients. Its upstream math header ignored TMN_SIGMOID_TANH, producing a
different sigmoid from our forward/CuTe path. The follow-up explicitly uses
the matching tanh.approx plus rounded fma formula and disables the extra
gamma register cache to reduce the observed spill. No fused candidate is
accepted on the failed results.

Array **17247** validates the corrected fused B1 arithmetic: dp, dg, dt,
dX and weight gradients are bitwise identical to the saved-products control,
with zero spills. It is nevertheless slower: backward 1.937 → 2.037 /
8.073 → 8.477 ms; F+B 3.000 → 3.079 / 12.410 → 12.863 ms. Array **17252**
uses TMA shared-read completion for the initial products' buffer release;
it also regresses (L768 backward 8.069 → 8.496 ms). Both fused versions
are rejected. Fewer intermediate HBM accesses did not improve the complete
workload, so neither was promoted to the full sanitizer suite.

## Validated forward normalization saves

Use `GP_OFF=1 GP_WARP=1 DX_LN=1 DX_WARP=1 SAVE_NORM=1` with
`selected_current.Training`. Forward stores its existing normalized triangle
and LN statistics, and backward skips the redundant normalization pass.
Forward output and all three saves match the recomputing path bit for bit.

Same-job paired measurements against the validated warp checkpoint below:

| L | Backward before → saved norm | Forward+backward before → saved norm |
|---:|---:|---:|
| 384 | 2.315 → 2.164 ms | 3.283 → 3.185 ms |
| 768 | 9.576 → 8.924 ms | 13.585 → 13.232 ms |

Jobs 17035/17047 ran on node01 with normal_h100 QoS. The complete workload
improves 3.0% / 2.6%, including the added forward stores. Job 17048 array tasks
pass every stress/graph/PyTorch comparison and both sanitizer tools at both
lengths: `validation-savednorm-L*.json`, `memcheck-savednorm-L*.log`, and
`racecheck-savednorm-L*.log`. All memcheck errors and racecheck hazards,
errors, and warnings are zero. The new forward kernel has zero stack/spills.

The unchanged compute-dominated SOL bounds give 35.2% / 39.4%, still below
SOL90. Results from different jobs are not ranked as small performance deltas.

## Validated warp-specialized checkpoint

The validated warp configuration is `GP_OFF=1 GP_WARP=1 DX_LN=1
DX_WARP=1` with `selected_current.Training`. It adds producer/consumer
warpgroups for B7 recompute/dW and a full-width dX GEMM fused with input
LayerNorm and the residual gradient. Both kernels use 32 producer / 224
consumer runtime registers and compile without stack frames or spills.
The dX accumulator stores must remain statically expanded; a runtime-indexed
array previously produced a 512-byte local stack.

| L | Backward stage 1 → warp | Forward+backward stage 1 → warp |
|---:|---:|---:|
| 384 | 3.720 → 2.279 ms | 4.658 → 3.214 ms |
| 768 | 15.197 → 9.166 ms | 18.970 → 12.948 ms |

Evidence: paired graph jobs 16861 and 16880. Strict standard gradients pass;
dX and the four input projection weight gradients are exactly equal to stage 1.
Jobs 16871/16872 (`validation-warp-L*.json`) pass the mutation, zero-input,
graph-replay and independent PyTorch checks described below. Job 16873 reports
zero memcheck errors and zero racecheck hazards/errors/warnings at both lengths
(`memcheck-warp-L*.log`, `racecheck-warp-L*.log`). The unchanged SOL model gives
33.4% / 38.4%, so SOL90 remains unachieved.

## Earlier validated checkpoint

`selected_current.Training` uses our wide saved forward, a native TMA output
normalizer, a Hopper projection GEMM with our exact gate-gradient epilogue,
cuBLAS output-gradient GEMMs, native output LN backward, the four contraction
GEMMs, and packed TMA B7 publication with producer-local dW.

The custom projection epilogue is in `cute_prepare.py`, composed with the
installed QuACK/CuTe Hopper GEMM pipeline. It preserves both BF16 roundings,
the tanh sigmoid, dropout scaling, and multiplication order. The projection
intermediate no longer goes through HBM. The B7 epilogue ports D128 packed
ldmatrix/stmatrix operations and publishes the derivative tile by TMA while
its local weight-gradient WGMMA executes.

Same-process alternating CUDA-graph measurements, in milliseconds:

| L | B1 stage 1 → checkpoint | Backward stage 1 → checkpoint | Forward+backward stage 1 → checkpoint |
|---:|---:|---:|---:|
| 384 | 1.813 → 0.898 | 3.686 → 2.509 | 4.624 → 3.439 |
| 768 | 7.518 → 3.473 | 15.300 → 10.114 | 19.089 → 13.899 |

Evidence: jobs 16817 and 16818. The comparisons hold forward and the incoming
gradient fixed, include all eleven gradients, and exclude allocation,
compilation, optimizer, RNG generation and host dispatch. These are comparisons
against our previous D256 training path. Absolute timings from different jobs
are not used to rank small differences.

`validation-current-L384.json` and `validation-current-L768.json` (jobs 16827,
16828) cover standard input, changed x/weights/dy/mask/dropout, zero gamma
subsets, zero mask, zero dropout, and graph replay captured before mutations.
All strict comparisons pass: dX relative L2 < 2e-5, weights < 5e-4, LN affine
gradients < 5e-6. Standard dX and input projection weight gradients are exactly
equal to stage 1. All eleven gradients also pass the separate compiled PyTorch
fixture threshold of 1e-2; its maximum relative L2 is about 0.00655.

Job 16830 checks native and custom CuTe kernels at both lengths: memcheck reports
0 errors and racecheck reports 0 hazards, errors, or warnings. Logs are
`memcheck-current-L*.log` and `racecheck-current-L*.log`.

## SOL accounting

The stage-1 optimistic whole-backward model remains unchanged:

- Necessary recomputing-policy FLOPs: `66*M*D^2 + 8*D*L^3`, `M=L^2`.
- Ideal unique-byte payload: `20*M*D + 44*D^2 + 48*D + 2*M + 2*L*D`.
- Nominal H100 dense BF16 peak: 989.5 TFLOP/s; HBM: 3.35 TB/s.
- Bound: max(compute time, unique-byte time), 0.761765 / 3.515840 ms.
- SOL90 targets: 0.846406 / 3.906489 ms for L384 / L768.

The earlier CuTe checkpoint was approximately 30.4% / 34.8% of this model;
the latest validated saves + LN configuration is 36.1% / 40.6%.
Internal rereads and extra FLOPs are not counted as necessary work to inflate
that percentage. This model is optimistic and does not establish attainability.

## Current bottlenecks and ongoing candidates

`current-L384.json` summarizes NCU job 16829 (profile timings, not benchmark
substitutes). Packed B7 source takes 580.864 us and the old dX/LN finish takes
718.304 us. Source tensor-pipe active-cycle utilization is about 57.7%; finish
is about 24.8%. B7 still writes four complete derivative planes to HBM.

`warp-L384.json` is the refreshed NCU profile (job 17031, node01,
normal_h100). Source is 537.024 us / 63.6% active tensor cycles; fused dX/LN
is 517.312 us / 35.0%. The separate normalization and output-LN backward
take 142.688 and 283.392 us. These percentages describe tensor-pipe activity,
not whole-backward SOL. Source reads/writes about 405/601 MB and dX/LN about
857/74 MB in this profile, motivating further intermediate-traffic removal.

Changes since the earlier checkpoint:

- `dx_ln.py`: dX GEMM and input LN/residual within one shared tile, removing the
  global dX-normalized intermediate. The first two-consumer version reached
  2.389 ms backward at L384 (job 16842).
- `mma_offset.cuh` and `warp_source.py`: compact WGMMA descriptor expressions,
  plus dedicated TMA producer and recompute/dW consumer warpgroups with 32/224
  runtime registers. Standard strict L384 gradients pass. Job 16847 records
  2.337 ms backward together with fused dX/LN; zero source spills.
- `dx_ln_warp.cu`: the corresponding full-width dX/LN consumer and dedicated
  producer, now part of the latest fully validated configuration above.
- `ring3.py`: concurrent producer/consumer CTAs, packed TMA publication and
  zero spills passed strict L384 gradients but took 2.700 ms backward
  (job 16870). The intended L2 reuse did not improve complete backward time.
- `chunk_b7.py`: sequential 32 MiB derivative windows also passed strict
  L384 checks, but backward took 3.215 ms (job 16881, four splits per chunk).
  More launches and weight partials outweighed the saved derivative traffic.
- `saved_norm.py`: forward output-LN saves preserve forward arithmetic and
  remove only normalization recomputation; all backward GEMMs remain. Jobs
  17035/17047 verify bitwise-equal output, normalized values, mean and reciprocal
  standard deviation at both lengths. Against the latest warp checkpoint,
  paired backward is 2.315 → 2.164 / 9.576 → 8.924 ms and full workload is
  3.283 → 3.185 / 13.585 → 13.232 ms. Forward adds 47.8 / 211.5 us; the full
  workload still improves 3.0% / 2.6%. Full stress/sanitizer job 17048 passes
  both lengths with zero sanitizer errors or hazards.
  `SAVE_NORM=1` enables the candidate. The existing normalized buffer has a
  longer lifetime but no additional large allocation. Including the saved
  norm/stat reads in the ideal payload does not change the compute-dominated
  SOL bound; counted GEMM FLOPs are unchanged.
- `DN_STATIC=1 DN_WIDE=1`: statically indexing the N256 accumulator improved
  the earlier wide dNorm/LN experiment, but it still took 438.320 us versus
  the current separate GEMM+LN at 376.896 us (job 17044). Ptxas still reports
  380-byte spill stores/loads. `dn_split.cuh` is a follow-up using two N128
  accumulator arrays and compact descriptor offsets. Job 17052 reduces spills
  to 72 bytes and time to 393.360 us, still slower than the separate path at
  377.792 us. A further candidate reuses the idle triangle shared-memory tile
  for affine sums while WGMMA runs (`dn_shared_sums.py`).
  Job 17064 removes all stack/spills and reaches 352.816 us versus 377.888 us
  for the original separate path. It remains slower than the new three-CTA
  LN plus vendor dNorm (about 309 us combined), so it is not selected.
- `LN_THREADS=128 LN_MINBLOCKS=3 LN_AGG=1 LN_CACHE=0`: job 17058 reduces
  output-LN time from 287.504 to 221.808 us, preserving bitwise dTri and strict
  affine gradients. The original 256-thread kernel has 212/452-byte spill
  stores/loads; this variant has 52/52 bytes. The zero-spill 200-register
  two-CTA variant is slower (242.880 us). Complete backward measurements and
  stress/sanitizer verification pass at both lengths. Jobs 17068/17073 pass
  strict gradients and improve saved-norm
  backward 2.174 → 2.110 / 8.939 → 8.659 ms and full workload
  3.205 → 3.141 / 13.159 → 12.931 ms. Validation array 17074 passes both
  lengths with zero sanitizer errors or hazards.

Detailed ring3 profile job 17078 (`ring3-L384.*`) shows 1.466 ms for the fused
kernel, 1,014 MB DRAM reads and 675 MB writes, with 35.3% active tensor cycles.
The separate warp kernels total about 1,937 MB versus the ring's 1,689 MB;
the ring therefore removes only about 13% of measured HBM traffic. Long
scoreboard and barrier stalls are prominent. This motivates a smaller ring
working set and different source/consumer CTA ratios (`check_ring_balance.py`),
not selecting the old ring on the basis of intended L2 residency.
Job 17083 tests four smaller-ring layouts at L384. All pass strict gradients,
but none beats the same-job baseline (2.065 ms backward / 3.063 ms full):
`c4n16s16` is 2.852 / 3.853 ms, `c4n24s16` is 2.356 / 3.353 ms,
`c6n12s16` is 2.741 / 3.741 ms, and `c6n12s8` is 2.673 / 3.673 ms.
Ring storage spans 12–24 MiB and 192–264 CTAs. Reducing capacity and changing
the CTA ratio alone is insufficient; these paths remain rejected and were
not promoted to full stress/sanitizer validation or engine dispatch.
NCU automatic CSV units differ by report (us/ms, Mbyte/Gbyte); `ncu_rows.py`
now preserves units and adds normalized `time_us`, `dram_read_MB`, and
`dram_write_MB` fields. Earlier profile JSONs have been refreshed from their
original CSVs without changing raw measurements.

`dn_slim.cu` / `dn_slim.py` implement the next dNorm/LN fusion: retain dNorm
in 64 KiB, retain input fragments in registers, and reuse a 32 KiB weight tile
for smaller triangle/LN row slices. A TMA producer and RS-WGMMA/LN consumer use
32/224 registers, 98,432 shared bytes and two CTAs per SM, without spills.
Job 17085 passes bitwise dTri and strict affine gradients, but takes 311.152 us
versus the current separate GEMM+LN at 309.984 us. Job 17087 passes complete
strict gradients but shows no backward gain (2.114 → 2.119 ms).

Profile 17089 (`dn-slim-L384.*`) records 308.320 us, 253.482 / 138.538 MB
DRAM read/write and 12.7% active tensor cycles. Long-scoreboard stalls remain
prominent. Job 17090 verifies four 16-/32-row and store-wait variants with
bitwise dTri and strict affine gradients; 16-row double buffering does not
improve performance. The 32-row read-only store wait is 309.856 us against a
312.064 us separate baseline, too small to select from this test. Job 17094
confirms that asynchronous LN-statistics preload lowers the fused stage to
287.904 us with zero spills; adding gamma caching reaches 282.736 us with
40-byte spill stores/loads, versus 312.048 us for the separate path. All
variants retain bitwise dTri and strict affine gradients. Complete strict
gradient checks pass in jobs 17095/17096. L384 backward improves slightly
(2.108 → 2.091 ms; full 3.141 → 3.120 ms), but L768 does not
(8.756 → 8.763 ms; full 13.108 → 13.163 ms). Gamma/spill and statistics-preload
effects at L768 are being isolated before any selection or full sanitizer run.
Job 17098 confirms that the L768 fused stage itself improves from 1,141.808
to 1,076.864 us. Gamma caching remains slightly faster than the zero-spill
variant (1,081.328 us), so spills alone do not explain the complete-workload
result. `check_dns_order.py` compares six valid stage orders with identical
buffers and identical prepare/B7 kernels to isolate ordering and cache effects.
Job 17103 confirms no meaningful full-backward gain at L768: the current
baseline is 8.679 ms and the best of six orders is 8.671 ms, while B1 itself
remains slower (2.719 versus 2.729 ms or more). The fused dNorm candidate is
not selected. Small standalone-stage gains do not justify replacing the
validated full-workload checkpoint.

Applying the same statistics preload to the separate LN kernel did not help:
job 17099 gives 222.976 → 237.280 us. Its 64 KiB triangle load and metadata
copies share the initial wait, unlike the fused kernel's overlap with GEMM.
That separate-LN variant remains disabled (`LN_STATS_TMA=0`).
`DN_SLIM=1` remains an experimental opt-in, not part of the validated selection.

## DSM cluster experiment (not selected)

`cluster_smoke.cu` first tests an eight-CTA, two-slot DSM exchange. Job 17108
passes exact data checks, changed input and four CUDA-graph replays. The
representative-thread mbarrier protocol nevertheless reports six racecheck
errors in job 17109 and is not accepted. The full-cluster synchronization
control (17110) and direct per-thread arrivals (17111) both pass memcheck and
racecheck with zero errors/hazards. The installed tool is Compute Sanitizer
2025.2.0.0. These are mechanism tests, not full-B7 validation.

`cluster_side.cu` / `cluster_b7.py` implement a separate experimental B7:
eight CTAs per cluster, two recompute/dW warpgroups per CTA, and four dX
warpgroups per cluster. Left and right launch separately, retaining the dX
prefix in FP32; a separate existing input-LN/reduction kernel finishes.
The source derivative tile is transposed in place after dW and consumed by
RS WGMMA through DSM. This prototype uses 213,120 shared bytes per CTA and
never changes production or selected dispatch. Job 17114 confirms bitwise
derivatives and dX at L384; every gradient passes the strict limits.

Job 17115 compares five direct-arrival layouts on identical buffers. All
pass strict gradients, but the fastest (14 clusters, 80 dX registers) takes
8.608 ms backward versus 2.049 ms for the current path. Moving from 64 to 80/96 dX
registers removes stack/spills, but does not make the schedule competitive.
`CLUSTER_SYNC_PIPE=1` is the next candidate: one cluster synchronization per
tile, overlapping source work for tile t with dX for tile t-1, plus a drain
iteration. This follows the already passing full-cluster synchronization
control; complete-kernel validation is still required. Job 17116 passes all
five strict comparisons, but the best backward time is 8.942 ms versus
2.049 ms. Removing the per-thread remote arrivals does not improve the
schedule, so their cost alone does not explain the slowdown. The outstanding
question is DSM operand-load latency and its interaction with RS WGMMA;
`CLUSTER_PREFETCH=1` batches four register fragments before the WGMMA fence.
Profile job 17117 (`cluster-pipe-L384.*`) measures 3.879 / 3.876 ms for
left/right and 0.256 ms for the separate finish. The three stages total about
1,239 MB DRAM traffic, versus about 1,937 MB for the earlier separate warp
profile, yet left/right tensor-pipe activity is only 7.9% / 7.2%. Barrier
stalls are 46.95 / 52.88 per issue-active cycle; this is a synchronization and
operand-supply problem rather than saturation of HBM. Maximum resident
clusters reported by the occupancy API and NCU is 15; a 16-cluster grid
therefore adds a scheduling tail.

`CLUSTER_BULK=1 CLUSTER_SYNC_PIPE=1` additionally merges the two source
warpgroups' tiles into projection64/gate64 planes and uses DSM bulk copies
into dX-local shared stages. Credit barriers prevent overwriting a dX stage
still in use, and the destination TMA barrier tracks both its local weight
copy and its remote derivative copy. Job 17120 passes bitwise derivative
and dX checks and all strict gradients at L384. This result alone does not
establish full stress, sanitizer or performance acceptance. The initial
prefetch-only job 17118 failed to compile because it read a source revision
while the wrapper was being edited; the paired rerun is job 17121, which also
compares the bulk-copy variant. Job 17121 passes all six strict comparisons
but rejects every candidate: the best bulk version (15 clusters, 96 dX
registers) is 7.270 ms backward / 8.260 ms full versus 2.068 / 3.057 ms for
the current path. Prefetched direct loads remain slower at 8.736 ms backward.
Bulk copies help within the DSM design but do not justify its use. No full
stress/sanitizer promotion or L768 performance claim is made for these B7
candidates. `CLUSTER_CLOCK=1` instruments one representative tile for source,
dX and barrier timing before any further architectural change.
Job 17122 completes this diagnostic with strict gradients still passing.
For a representative right-side tile, each source WG spends about 5,700
cycles through input wait, projection, GLU, dW and transpose, then waits
about 26,760 cycles at the cluster barrier. The dX WG spends about 28,800
cycles through operand waits and its matrix products, plus stores/barrier.
These are instrumented per-SM cycle differences, not a benchmark or a
cross-SM timestamp comparison. They identify dX operand delivery as the
limiting stage in this design.

`CLUSTER_PLANE=1` selects the separate `cluster_plane.cu` prototype. It keeps
resident source weights and two local GP slots, but exchanges each complete
64 KiB projection/gate plane before consuming it with shared/shared WGMMA.
This reduces the cross-CTA credit exchange from sixteen stages to one
projection-to-gate transition per tile. The output-gate prefix uses direct
BF16 register fragments; its FP32 accumulation is retained across the left
and right launches. Shared storage is 221,312 bytes per CTA. Job 17131 is
the first strict derivative/gradient validation of this new schedule; it
passes with bitwise derivatives and dX and all other gradients within the
strict limits. Job 17136 compares four configurations. The best (15 clusters,
80 dX registers) is 6.041 ms backward / 7.030 ms full versus 2.063 / 3.050 ms
for the current path. Both side kernels have zero stack/spills. The full-plane
design improves over small bulk transfers but remains rejected. Job 17141
adds separate timestamps for each dX weight wait, WGMMA group, and plane
transition to isolate the remaining latency.
Job 17141 passes strict gradients and identifies an approximately 8,027-cycle
projection-to-gate exchange wait. Most steady-state WGMMA groups take about
264 cycles, with approximately 106-cycle weight waits in this instrumented
sample. The first WGMMA group is longer (2,686 cycles); its dependency on the
global FP32 prefix is a plausible contributor, not an independently isolated
measurement. This does not change the rejection based on full-workload time.

## Numerical feasibility and shared-operand follow-ups

`check_dx_split_accuracy.py` evaluates the proposed producer-local dX
contributions before building their communication path. Job 17152 uses full
FP32 products with TF32 disabled; plane-major, rank-paired and balanced-tree
sums all fail strict dX (about 1.29e-4 versus the 2e-5 limit) and input-LN
affine tolerances. Job 17157 uses `split_dx_probe.cu` to produce the output-gate
prefix and each rank-local contribution with native WGMMA instead. Prefix
and partial-product differences from FP32 GEMM are only 2.94e-7 and 1.11e-7,
but sequential/tree regrouping still fails: dX is about 1.26e-4 and the BF16
dX-normalized intermediate differs in about 42,900 elements. These tested
regroupings are rejected; no numerical tolerance is relaxed. The probe
materializes large FP32 partials solely to assess numerical feasibility,
and carries no performance claim.

The next candidates preserve the original K accumulation order:

- `dx_ln_rows.cu`: one TMA producer and two full-D256 consumers process
  adjacent 64-row tiles while sharing each loaded weight tile. It uses
  32/224/224 runtime registers, 196,736 shared bytes and one CTA per SM.
  This differs from the old two-consumer implementation, which divided
  channels within one row tile. Job 17170 runs the strict/graph timing test.
- `source_pairs.cu`: two adjacent output-rank consumers share one normalized
  input tile while retaining separate weights and derivative/dW accumulators.
  It uses 32/224/224 registers and 163,968 shared bytes. GP publication and
  each dW accumulation order remain unchanged. Job 17174 tests it after 17170.
- `DX_N256=1` uses `mma256.cuh` / `dx_n256.py` to replace the two N128 WGMMA
  instructions with one N256 instruction. All accumulator stores remain
  statically indexed. Job 17176 compares this with row sharing on identical
  buffers after 17174. This flag is disabled in the validated checkpoint.

These are isolated experiments; no engine dispatch or autograd change is made.
Jobs are sequential on node01 / normal_h100. Pending job IDs remain the
authoritative handles during resource waits; they are not resubmitted.

CPU-only compilation on allocated node01 caught an N256 assembler limit in
job 17177: the old `__launch_bounds__(256,2)` static register target (128)
cannot encode N256; PTXAS requires at least 154. `dx_n256.widen` now uses
`__launch_bounds__(256,1)` for that candidate, with runtime occupancy still
queried by the launch wrapper. This changes the occupancy tradeoff and is
not a demonstrated speedup. The 384-thread row-sharing bound is unchanged.
CPU-only job 17178 compiles all four candidates at splits 8 and 16 with zero
stack and spills. PTXAS reports 224 registers for the single-consumer N256
kernel and 168 for the three 384-thread kernels; runtime register roles
remain as specified above. `compile-shared-candidates-17178.json` records
each content-addressed cubin and confirms no CUDA context was initialized.

During the earlier resource wait all eight node01 GPUs were allocated. Jobs
17170, 17174 and 17176 then requested four minutes each;
compilation success was not numerical or performance validation.
L768 performance/strict checks were queued behind those existing handles:
17180 runs `check_source_pairs.py` after 17176, and 17181 runs
`check_dx_variants.py` after 17180. Both retain node01 / normal_h100 and a
four-minute limit; all five GPU jobs run serially. The variant comparison
records a numerical failure without timing that candidate, then checks the
remaining candidates. `strict_all` reports the aggregate numerical result;
`complete` only means the comparison finished, not that every candidate passed.
`SHARED_CANDIDATE=dx_n256|dx_rows128|dx_rows256|source_pairs` explicitly
attaches a candidate in `validate_current.py` and `profile_current.py`, with
`DX_N256=0` for initial control construction. The validator records candidate
cubin, launch metadata and per-case progress, and uses a job-specific default
tag for these experiments. Full stress, graph, independent PyTorch and both
sanitizers remain mandatory before selection. When invoking
`sanitize_current.sbatch`, set a distinct `VALIDATION_TAG` as well as
`FULL_VALIDATE=1` to preserve previous accepted records and logs. No new
validation jobs have been submitted before the pending performance tests.

Rejected measurements are retained: all-cuBLAS B7, one giant dX GEMM,
row-major derivative scatter, two-GEMM CuTe dX (also failed strict accuracy),
the first ring schedules, larger resident output weights, deeper B1 preparation
buffering, and native dNorm/LN fusion. Fewer launches or less nominal HBM
traffic alone did not justify selecting them.

Current experiments use **node02 / normal_h100 QoS** (priority 100, compared
with cssb_h100 priority 1000), per the resumed user instruction. The profiling
driver accepts `PROFILE_TAG` and `LENGTH` to preserve earlier evidence.

Experiments are launched with `run.sbatch`; configuration and job ID are stored
in new `result-L*-<job>.json` records. Sanitize with
`sanitize_current.sbatch`. Run through the repository's cu128 environment on a
Slurm H100 allocation, never on the login node.

- **18217** joint input/gate dW splits32/64 pass standard strict checks but
  do not improve full F+B; D512 long split64 30.363 -> 30.569 ms. Rejected.
- **18224** bounded-unroll delta normalization passes strict checks and exact
  restored norm, but is slower than the low-register row32/minblocks3 pilot
  **18205** in the isolated norm stage. Keep the simpler full-unroll choice.
- **18230** interleaving paired contraction modes preserves GP bits and
  standard strict gradients at all four wide shapes. Pair-tiles reduces full
  F+B by 0.073 / 0.202 / 0.056 / 0.362 ms (D384 short/long, D512 short/long).
  Checkpoint14 combines this ordering with the **18205** row32 D512 norm.
  Full strict/graph/PyTorch and normal/forced-overflow sanitizers are required
  before promoting its timings into the qualified comparison table.

- **18241 / 18243** checkpoint14 full qualification and dependent direct paired
  Triton comparison. Four shapes pass five strict stress fixtures, graphs and
  PyTorch; final D512 long forced-overflow sanitizers are still running.
- **18247** overlaps output-gate gradient TMA input/output transfers, testing
  one and two input slots with unchanged DP/DG arithmetic at all six shapes.
- **18248** keeps one contraction WGMMA group outstanding, issues future TMA
  only after the previous group releases its input, and retains pair-tiles order.
- **18249** channel-owned affine sums reduce worker register pressure, with
  explicit shared-input reloads and three-CTA variants. Full LN and F+B are
  measured only after strict gradients and dTriangle checks.

- **18241 / 18243 completed**: checkpoint14 is fully qualified, including
  both lossless-normalization fallback paths. The top table now uses its
  measured direct Triton results. The six-shape 1.5x goal is still unmet.
- **18251** profiles remaining checkpoint14 contraction traffic and stalls.
- **18255** tests constant dropout indexing and scalar/128-bit vector output
  epilogues with exactly the current forward sigmoid and rounding.
- **18257** refreshes current short-D256 component timings as diagnostics.

- **18247** both gate-transfer overlap variants pass standard strict checks
  at all six shapes, with bitwise DP/DG. Isolated gains are 5-20 us; full
  long gains need repeat/qualification before selection.
- **18248** one outstanding contraction WGMMA group is strict/bitwise but
  does not improve the fused stage. Rejected.
- **18249** channel-owned affine worker first variant is strict but slower
  at D256 short (LN 231 -> 307 us). Its three-CTA variant stalled: cubin has
  REG80/STACK0 but the requested 112+56 warpgroup registers require average84.
  Only this pilot array was cancelled. Corrected 112+48 split and explicit
  cuFuncGetAttribute register-pool guard run in **18270**.
- **18267** exact shared sigmoid table covers BF16 magnitude bits
  [0x3a00,0x4200), both signs; all other values retain original sigmoid.
  16KB table includes static-shared headroom for two CTAs. Pending pilot.

- **18251** NCU current D512 long interleaved contraction: 5.413 ms,
  8.589 GB DRAM reads / 4.817 GB writes, tensor activity44.2%, barrier
  stall ratio2.93 and long-scoreboard3.80. Most remaining work is memory
  traffic; this is diagnostic, not a full timing or promotion result.
- **18255** specialized/vector output epilogues are strict everywhere but
  save at most16 us in the isolated stage; short full times are neutral or
  slower. Long full deltas exceed isolated gains and need qualification.
- **18257** short D256 diagnostics: contraction four calls339 us, source555,
  prefix dX278, output LN232, input LN129, gate preparation138, dNorm89,
  dWproj87, dWgate68. Forward front318, norm150, output epi111 us.
- **18267** shared exact sigmoid table retains two CTAs and GP bits but
  regresses the fused stage by9-11% at all shapes. Rejected.
- **18270** corrected channel affine worker passes standard strict checks
  and dTriangle bits at all shapes. No isolated-stage win. D384 row16/mb3
  still has only2 CTAs due128 bytes too much shared storage; **18289** trims
  the gamma cache to test actual3 CTAs. D512 row8/mb3 regresses heavily.
- **18278** one-warpgroup64x128 contractions, slots2/3, are strict but slower
  in all completed shapes. D384 long4.17 ->5.02 ms fused with3 slots.
- **18291** inspects channel-worker cubin register and spill resources on
  the compute node; no performance result by itself.

- **18278 completed**: one-warpgroup contraction slots2/3 are strict/bitwise
  but slower at every shape; D512 long5.57 ->6.70 ms. Rejected.
- **18289** trimmed D384 affine-worker cache achieves3 CTAs and remains
  strict/bitwise dTriangle, but LN353 ->388 us short and1.566 ->1.950 ms long.
  Rejected. **18291** confirms D512 row8 worker has REG80/STACK40, while
  D384 row16 three-CTA-bound worker has REG80/STACK0.
- **18292** new output-LN structure computes exact row sums first, then
  handles affine gradients in4/8-channel-per-lane chunks. Consumed dNorm
  slabs become reduction scratch; per-CTA FP32 partials replace persistent
  H/32 gamma/beta register arrays. A final small kernel reduces the partials.
  Full strict/dTriangle and paired whole-stage/F+B tests are pending.

- **18292 completed**: chunked affine accumulation preserves dTriangle and
  strict gradients but is slower at all shapes; D512 long2.26 ->2.97 ms LN.
  Rejected. **18298** direct tanh sigmoid fails unchanged dX tolerance at
  all shapes (3.84-3.99e-4 vs2e-5), so no timing or selection was done.
- **18302** post-dNorm dW/LN overlap with an ordinary LN launch passes
  strict and explicit CUDA-graph checks, with bitwise dTriangle. Full-grid
  overlap and half-grid overlap both regress D384/D512 long. Serial ordinary
  LN is neutral. Rejected; changing the overlap boundary did not help.
- **18304** prototype D256 warp-MMA contractions over spatial16x64 tiles
  and32 channels per CTA. Two K16 TMA slots, original K ordering; evaluates
  derivative accuracy and cost before any proposed source/dW fusion.
  This is an isolated experiment, not a replacement of the qualified path.

- **18304 completed**: exact DL/DR and strict full derivatives, but isolated
  contraction328 ->2627 us short and1424 ->19905 us long. Rejected before
  attempting source fusion; the contraction cost outweighs saved traffic.
- Next experiment enumerates cuBLASLt algorithm capabilities/configurations
  beyond the existing heuristic list. Frozen checkpoint14 remains unchanged.

- **18307** inspected installed CUDA12.9 cuBLASLt capability/configuration ABI.
  **18308** searches unsplit dX configurations across all six shapes, beyond
  the eight heuristic suggestions. D256 short **18309** checks10,255 supported
  configurations; top12 paired measurements yield no selected improvement.
  **18312** follows with wide input dW configuration search, serialized.
- **18315** tests bounded-unroll normalizer lifetimes while preserving the
  original per-warp affine accumulator order. Row8 targets three/four CTAs,
  with explicit normalizer64/affine96 register budgets at D384/D512.
  This differs from the rejected channel-owned affine accumulator pilot.
  Accuracy, resource counts and paired full measurements are pending.

- **18308/18312/18315 wide tasks and18316** initially stopped in setup:
  missing `PREFIX_COPY=tma` left `PrefixDx.copy_cubin` absent. The new entry
  scripts now explicitly set PREFIX_IMPL/PREFIX_COPY/CHECKPOINT_LN_THREADS.
  These failed wide tasks are not numerical or performance results.
- **18315 D256 /18329 wide** bounded affine LN passes strict and dTriangle
  bits, and achieves three/four CTAs with no local-memory spills. It is still
  much slower: D256 short227 ->517 us; D512 long2208 ->6987 us. Rejected.
- **18332** profiles current D512 long LN:2.129 ms,3.156 GB DRAM reads,
  1.200 GB writes, long-scoreboard ratio4.862, SM throughput41.34%.
  **18337** groups adjacent row16 tiles within each CTA to test locality
  without changing the per-row arithmetic or shared storage.
- **18333** corrected wide dX/input-dW configuration search. D384 short
  **18335** dW is bitwise and strict,608.1 ->598.8 us isolated, but full only
  4711.90 ->4707.66 us. No qualification or new checkpoint selected.
- **18338** is an isolated lossless-preactivation consumer prototype:
  mantissa/sign bytes plus exponent nibbles reduce common pre reads by25%.
  Exponent code15 reads the original BF16 location, covering arbitrary
  values without a fixed exception capacity. All65,536 BF16 patterns are
  checked. Forward encoding is not integrated, so its timing is NOT a full
  training speedup; integration/validation depend on a useful consumer gain.

- Reordered only pending own D512 tasks18333_4/5; they now run as **18339**
  after18337/18338. Active D38418333_3 and unrelated training were preserved.
- **18340** conditionally tests `wide_bf12_saved_front.py` only for shapes
  where18338 shows a strict consumer gain of at least2%. Existing BF16
  shared staging is read into registers, then repacked into4KB mantissa and
  2KB exponent slabs in the same8KB stage. Two TMA stores share one bulk
  group; AB staging and its credit protocol are unchanged. Only exception
  locations of the original pre buffer are written. The complete pre buffer
  is poisoned with NaNs before the first candidate forward to detect stale
  exception values or accidental reads of common uncompressed locations.
  This source has only passed AST inspection; GPU results remain pending.

- **18333 D384 long18336**: dX and input dW are bitwise and strict, with
  paired isolated savings30/46 us. Combined full19991.97 ->19929.55 us,
  about62 us. This is a pilot, not a new qualified checkpoint.
- **18337** adjacent LN tile groups2/4/8 all preserve dTriangle bits and
  strict gradients, but regress both lengths. Rejected.
- **18338** lossless BF12 consumer passes all65,536 encoded patterns,
  GP bits and strict full backward at all four shapes. It regresses GP by
  40-48%; D512 long5274.75 ->7796.16 us, backward19270.54 ->22219.71 us.
  Thus18340's integrated-forward condition is false at every shape.
- Resource inspection on the active compute allocation finds BF12 GP
  REG128/STACK48 versus the original REG90/STACK0. **18346** therefore
  restores one32KB BF16 plane at a time before the unchanged original GP
  arithmetic. At most32 decoded words plus the64 WGMMA accumulators are
  intended to remain live. Shared mask stays at64KB; P expansion precedes
  G expansion to protect unread compressed data. Four uniform CTA barriers
  separate reads from overlapping expansion stores. Results are pending.

- Before18346 GPU execution, CPU-only compilation of the32-word staged
  decoder found STACK1680 at both widths; it was not timed. The revised
  decoder stores each compressed64x64 tile as adjacent4KB/2KB slabs and
  expands tiles3,2,1,0, with P before G. Only8 decoded words stay live, and
  uniform barriers protect each overlapping expansion. Compute-node
  compilation now reports REG128/STACK0 for D384/D512 (84fe51d / e8adb59f).
  **18346** will test this revised version.
- **18347** combines current long-shape Lt candidates, the individually
  strict gate pipeline (slots1/2), and vector4 output epilogue. It measures
  complete backward and complete F+B separately at D384/D512 L768; no gain
  is inferred by adding the previous isolated measurements.

- **18355** staged BF12 GP profiling confirms lower DRAM reads (7.419 GB
  versus checkpoint14 8.589 GB), but 8.563 versus 5.413 ms and tensor
  utilization 27.9% versus 44.2%. Decoding costs more than the read savings.
  This direction is rejected; no compressed-forward integration selected.
- **18356** ordinary 64-row full-width dX plus input-LN passes strict
  gradients at every shape, but always regresses. D512 long two/three
  slots finish 4.64/4.69 -> 7.62/6.62 ms; full 29.29/29.52 -> 33.00/31.70.
- **18362** packed BF16 register retention across column chunks reduces
  storage and achieves three CTAs at D256/D384, two at D512. Re-reading
  GP and smaller WGMMA tiles outweigh occupancy gains. All completed
  shapes are strict, all slower; D256 short finish 401 -> 669 us,
  D384 short 746 -> 1498 us, D512 short 1214 -> 1938 us. Rejected.
- Next `wide_stream_contract_gp.cu` uses independent TMA, two ordered
  contraction warpgroups, and one GP warpgroup. Three input stages occupy
  96 KB; separate preactivation/mask and rounded contraction buffers use
  96+32 KB. Persistent CTAs may compute the next contraction while GP
  consumes the previous result. Credits protect all three buffer lifetimes.
  This is an unqualified experiment; checkpoint14 remains unchanged.

- **18368/18369/18372** persistent contraction/GP pipelines preserve GP bits
  and strict gradients, but regress D512 short GP: 1.127 -> 2.355 ms with
  one GP warpgroup, 1.127 -> 1.631 ms with two. Read-completion credits
  instead of full output-store completion do not improve the two-group case.
- **18370/18371** four GP warpgroups cannot compile: the 896-thread block
  permits 72 registers, while WGMMA requires at least 90. No GPU result.
- **18373/18374** dedicated-loader 128x64 pipeline cannot compile at two
  CTAs: 56-register allocation versus 58 required. Removing descriptor-offset
  temporaries does not change this lower bound. **18375** removes the loader
  warp, achieves two CTAs with REG64/STACK24 and strict/bitwise GP, but GP
  1.128 -> 1.346 ms and full 7.256 -> 7.572 ms. Rejected. Constant-N CPU
  compilation still has STACK24; a noinline inner contraction still cannot
  compile under the dedicated-loader 56-register bound.
- **18376** enumerates 10,248 supported unsplit configurations for each
  D256 short contraction (bc0/bc1/bc2/bc3). Selected pilots are bitwise and
  strict, saving roughly 2 us per isolated call, but combined full is neutral:
  2.75664 -> 2.75800 ms. No selection or qualification follows.
- `cluster_dn_ln.cu` is the next independent fusion prototype: each CTA
  holds 64x128 BF16 dNorm, and an eight-CTA D512 cluster covers all channels.
  DSM passes all per-lane FP32 LN sums through channel chunks in the original
  q order, then broadcasts the final row sums. Only affine partials and final
  dTriangle reach HBM. Cluster arrival/acquire barriers protect publication;
  initial/final cluster synchronization protects shared allocation lifetimes.
  Diagnostic mode writes dNorm only for correctness comparison. Qualification
  and performance remain pending; checkpoint14 is unchanged.

- **18377** cluster dNorm/LN first stopped in setup (wide Training keeps
  projection weight at b1.wp). **18378** corrected wiring is strict and
  bitwise dNorm/dTriangle, but 0.674 -> 3.418 ms combined; full 6.972 -> 9.892.
- **18379/18380** bulk DSM transfers and independent local precomputation
  reduce that regression to 2.213 ms. 18379 reused the final CTA input-sum
  area for results and has an identified potential cross-warp lifetime race;
  it is invalid regardless of observed numerical success. 18380 moves final
  results to consumed TRI storage, remains strict/bitwise, and is still slow.
- **18381** phase clock instrumentation shows rank0 final wait about25k
  cycles, rank7 chain wait about21.5k; precompute about8k and reconstruction
  about6.4k. The serial sum propagation dominates. **18382** parallel row
  partials plus TMA stats lower dTriangle discrepancy to6.3e-6, but complete
  backward amplifies it to dX1.2e-4 and affine about1e-4. Rejected before timing.
- **18383** instead adds TMA stats to the existing checkpoint14 LN structures.
  All four shapes are strict, with bitwise dTriangle and zero stack spills.
  D384 short LN311.17 ->302.90 us, full4705.25 ->4684.48;
  D384 long LN1302.62 ->1190.32 us, full19977.12 ->19863.94;
  D512 short LN459.34 ->424.51 us, full7295.52 ->7235.44;
  D512 long LN2373.84 ->1690.06 us, full30010.32 ->29897.30.
  Isolated and full gains differ materially; use directly measured full timing.
- `wide_checkpoint15.py` packages only this stats change for qualification.
  **18387** runs all strict/graph/PyTorch fixtures and memcheck/racecheck/
  synccheck, including both inherited D512 overflow branches. Pending results
  are not a promotion; checkpoint14 remains the qualified wide reference.

- **18387 / 18389 complete**: checkpoint15 is qualified at all four wide
  shapes; D384 long passes 1.5x in both scopes. D512 long is still below 1.5x.
- **18392** GP store cache priorities first/last are strict and bitwise but
  have no useful full-workload gain. **18394** saved-forward preactivation
  priorities are also strict with no useful gain. Neither is selected.
- **18393** D256 row32 CachedLN plus TMA stats is strict/bitwise dTriangle,
  zero stack, three CTAs. Full 2771.09 -> 2743.71 us short and
  11303.65 -> 11192.67 us long. **18415/18417** qualify and compare it.
  D256 short now reaches 1.2024x BWD / 1.3064x F+B; long 1.6771x / 1.6645x.
- **18407** restores existing saved input statistics into the prefix LN
  finish. Strict dX errors 5.34e-6/6.93e-6, but isolated finish saves only
  1-2 us and short full is neutral. Not selected.
- **18409** larger output-LN occupancy after TMA stats does not give a
  consistent improvement. D384 short row16/three-CTA saves about20 us full
  despite slower isolated LN; four-CTA and wide long variants regress.
- **18413** defers D256 dW completion through the next projection's wait,
  with credits released only after previous MMA and TMA shared reads finish.
  GP/weight partials are bitwise and all gradients strict, but source and
  full regress at both lengths. Rejected.
- **18418** tests LN tensor-map L2 promotion sizes with unchanged arithmetic.
- **18420** fills retired contraction K slots with gate/projection inputs
  during the final two K steps. Mask reuses the last slot after its MMA
  finishes. Shared storage, occupancy target and reduction order are unchanged;
  correctness, timing and any necessary sanitizer qualification are pending.

- **18442/18444 complete**: checkpoint16 is qualified at all four wide shapes.
  D512 long BWD 27.920 -> 18.754 ms (1.4888x), F+B 43.322 -> 28.825
  (1.5030x). The complete six-shape goal remains unmet.
- **18427** combined gate/output epilogues and Lt choices preserve strict
  gradients, but repeat full gains are small and inconsistent. Not selected.
- **18431** retaining mask values in registers introduces STACK8 and regresses
  every shape; **18438** lossless two-bit masks are strict for binary and
  nonbinary values, zero stack, but encoding/decoding costs regress full.
- **18435** D256 two/four-CTA XN multicast is bitwise and strict but slower;
  **18448** adjacent row-split consumers share one weight tile, also strict
  with zero stack, but source and full regress at both lengths.
- **18449** predicated PTX BF12 decode matches all 65,536 raw BF16 patterns
  and strict gradients. It reduces STACK48 to16 but still loses substantially
  in isolated contraction and full backward. Not integrated into forward.
- **18450** specializing N and fully unrolling twelve K iterations preserves
  GP bits and strict gradients. D384 long full 19.888 -> 19.784 ms; D512 long
  29.869 -> 29.686 ms. Short shapes have no useful repeatable gain.
  `wide_checkpoint17.py` selects only long specialization for qualification.
- **18456** M128N96 reaches three CTAs, REG74/STACK0, and bitwise GP/strict
  gradients, but extra tile traffic regresses full. Further changes need a
  distinct pipeline hypothesis before another test.
- **18473** tests four legal placements of independent output-weight gradients
  to preserve producer/consumer cache locality. No kernel arithmetic changes.

- **18479/18480 complete**: checkpoint17 long shapes pass all gates. D384 long
  BWD/full 1.5169x/1.5019x; D512 long 1.4954x/1.5120x. D512 backward needs
  about57 us more under the same paired threshold. The six-shape goal is unmet.
- **18473** all six shapes pass strict gradients with four independent-weight
  placements. No consistent short/full benefit; not selected.
- **18485** M192N96 uses REG74/STACK0 and two CTAs (six compute warpgroups per
  SM). GP is bitwise and gradients strict, but short shapes regress. Long
  isolated contraction also regresses; small D384 full fluctuation is not
  selected. Equal aggregate operand bytes do not ensure better performance.
- **18487** current contraction consumers use separate Lt workspaces and graph
  fork/join events. Both issue orders pass strict and graph checks. D512 long
  dW-first improves BWD19.553 ->19.437 ms and full29.854 ->29.705 ms.
  `wide_checkpoint18.py` selects only that shape for full qualification.
- **18493** AB projection reconstruction uses two-bit raw-BF16 corrections and
  full-value escapes (~0.2%). Normal, nonbinary/negative and zero masks preserve
  GP bits and strict gradients with poisoned unused escapes. REG95/96, STACK0,
  but inline exception reads and reconstruction regress consumer timing. This
  is not integrated into forward, and no full-workload gain is claimed.
- **18500** repeats gate-pipeline2 on checkpoint17 D512 long three times. Full
  saves6-29 us but backward is neutral/slower in two trials. Not selected.
- **18507** adds four output-gate weight ranks to D256 source and reduces their
  partials in the input finish. GP/input partials are bitwise and gate weight
  error2.64e-4 is strict short. The initial separate compute branch spills256
  bytes atREG128 and regresses source/full; consolidate accumulator live ranges.

- **18513/18514 complete**: checkpoint18 D512 long passes all strict/graph/
  PyTorch gates and memcheck/racecheck/synccheck, including both overflow paths.
  Relevant GEMMs, contraction and input-finish kernels are instrumented in
  race/sync runs; input dW and dX own separate workspaces and join explicitly
  before the input finish consumes their outputs. Direct existing/ours ratios
  are1.50218x BWD and1.51575x F+B. Long widths all pass; short widths do not.
- **18516** tests shared dW accumulator lifetimes in D256 joint-gate source.
  CPU-only compile shows REG128/STACK264, so it did not remove spills. A
  power-of-two rank-decoding diagnostic follows its timed short GPU pilot;
  compilation must not overlap the timed section.
- **18517** tests sparse AB-projection exception patching after the main GP
  consumer, with a bounded list and a dense fallback when it overflows.
  Forward integration remains absent; only consumer/BWD timing is diagnostic.

- **18516 complete**: shared-accumulator D256 gate fusion preserves GP and
  input partials and is strict, but still spills264 bytes and is slower.
  Short full2.735 ->2.926 ms; long11.164 ->11.595 ms. The CPU diagnostic
  appended after submission did not execute in that task's already loaded
  script; explicit CPU-only **18526** tested it. Power-of-two two-dimensional
  grids still spill272 bytes, so no extra GPU run is justified for that edit.
- **18517 complete**: separate sparse AB-projection patches preserve GP bits
  for normal/nonbinary/zero masks and forced list overflow. REG96/STACK0,
  but consumer plus patch is even slower: D384 short0.845 ->1.398 ms,
  D512 short1.128 ->1.812 ms. Long regressions are larger. Rejected; no forward
  integration or full-speedup claim.
- **18529** D256 compact16KB global sigmoid table matches all65,536 BF16 input
  patterns, GP and weight partials. REG128/STACK0 and original shared-memory
  occupancy are preserved; both explicit L1-cache and readonly loads are slower.
  Short source0.553 ->0.768/0.752 ms and full2.69 ->2.89 ms. Rejected.
- **18531** profiles the current qualified L384 paths at D256/384/512, then
  launches a separate process without NCU for CUDA-graph component diagnostics.
  Those isolated timings are not a new paired Triton comparison.

- **18534** revisits output-LN shared layouts on current small tiles.
  Direct channel-major reads preserve dTriangle bits but regress the full
  workload. Per-tile transpose barriers are neutral except a small D384 gain.
- **18538** applies current overlapped preactivation saves and staged fused
  contraction/GP to D256 short. All three dW split counts are strict; split8
  preserves GP and weights exactly. BWD1.871 ->1.787 ms, but full2.737 ->2.804
  ms after including forward saves. Not selected.
- **18539/18542** combine output-LN transfers into larger maps. Single 3D
  maps work, but natural row-major dNorm increases registers. **18545** uses
  permuted global strides to retain the original chunked shared layout and
  register counts. All short shapes are strict with bitwise dTriangle.
  D256/D384/D512 LN202.18/299.01/424.94 ->181.76/272.91/404.45 us;
  full2746.98/4717.10/7266.42 ->2736.05/4684.34/7235.36 us.
- **18550** stopped at a host parameter-count assertion (23 fields, not22),
  before the contraction candidate ran. Corrected **18554** groups GP
  epilogue/output, optionally contraction input, transfers into 4D maps.
  GP is bitwise and strict, REG90/STACK0, but stage improvements are marginal.
  **18556** similarly groups D256 source XN/weight loads; GP/partials are exact,
  REG128/STACK0, timing is near neutral. A submission dependency typo allowed
  18556 to overlap 18554; those timing records are not selection evidence.
  No GP/source change was selected from these jobs.
- `d256_permuted_ln_checkpoint.py` and `wide_checkpoint19.py` select only the
  measured short-shape output-LN map change. Strict fixtures, graphs, independent
  PyTorch and sanitizers run in **18558** then **18559**. Direct Triton timings
  **18560/18561** require their respective successful qualifications and remain
  serialized. The long-shape selections are unchanged. None of these pending
  jobs establishes the still-unmet all-six-shape target.
- **18563** follows the timing chain with the same one-map transfer strategy
  for input-LN/residual and dW reduction, preserving all shared indices and
  arithmetic. This remains an unqualified pilot.
- **18558/18559 completed**: all strict/graph/PyTorch and required sanitizer
  gates pass. Comparison18560/18561 stopped before timing because required
  `VALIDATION_ARRAY_JOB` was omitted. Corrected **18569/18570** explicitly
  supply that evidence ID and `COMPARE_TRITON=1`. Qualified short BWD/full
  speedups are1.2101/1.3106 (D256),1.3739/1.3777 (D384),1.3058/1.3561 (D512).
- **18563** D256/D384 input transfers are strict and only marginally faster.
  D512 short initially overwrote dWgate because the test incorrectly assumed
  that the long shape's joint dW path was active. The fixed harness inspects
  the actual reducer class; corrected D512 pilot **18574** follows.
- **18568**, `cluster_rows_dn_ln`: after all GEMMs retire their shared slots,
  bulk DSM redistributes 8-row slabs to independent full-width LN owners.
  This preserves channel sum order without the old serial sum propagation.
  D512 short dNorm/dTriangle and dX/weights are bitwise, all gradients strict,
  REG196/STACK0,30 active clusters. Combined stage620.90 ->1317.09 us, so it
  is not selected. Phase-clock job **18572** finds median GEMM8236 cycles,
  exchange2571, LN/store10514, affine aggregate/wait2808, remote affine767.
  The next **18576** doubles GEMM rows and compute warpgroups, uses16-row LN
  owners, and protects retired TRI reuse with an extra cluster barrier.
  It is an unqualified experiment, not a new performance claim.
- **18574** fixes the D512 short reduction wiring: all gradients pass, but
  input stage244.19 ->238.16 us and full7246.18 ->7233.92 us are modest pilot
  gains. **18576** doubles cluster rows and WGMMA groups with protected affine
  scratch reuse; dNorm/dTriangle and dX/weights remain bitwise, REG194/STACK0,
  but15 active clusters and stage617.39 ->1422.13 us. Not selected.
- **18578** launches independent output-gate dW alongside dNorm using its
  own workspace and graph events, joining before backward returns. Both short
  D256/D512 are strict and graph-correct; D256 regresses, D512 full is neutral.
  Not selected. The next output-LN experiment revisits nonspilling row/CTA
  choices after the single-transfer change; no occupancy gain is assumed.
- **18581** retests CachedLN occupancy with permuted whole-tensor maps.
  Nonspilling D256/D384 configurations are neutral in full workload; D512's
  three-CTA option spills72 bytes and is not timed. **18584** smaller double
  buffers are strict but regress: row8 roughly doubles isolated LN and adds
  0.26/0.41/0.41 ms to short D256/D384/D512 full workloads. Not selected.
- **18589** larger whole-map input-LN rows are strict; D256 row32 is neutral,
  row64 and D384 row32 regress. **18593** interleaves two independent LN rows
  per warp without changing per-channel affine or per-row summation order;
  only nonspilling strict candidates will receive full timing.

- **18593 completed**: interleaving independent LN rows preserves dTriangle
  and all strict gradients without spills. D256/D512 full is neutral, while
  D384 LN263.73 ->274.48 us and full4635.49 ->4637.68 us. Not selected.
- **18596** saves only gate or projection preactivation at D256 short,
  preserving original forward shared-memory size. The source recomputes the
  other half with m64n32 WGMMA. Both GP and dW partials are bitwise and all
  gradients strict, REG128/STACK0. Projection-save full2716.59 ->2830.96 us;
  gate-save2714.58 ->2833.12 us. Extra reads outweigh reduced GEMM work.
- **18599** pipelines complete spatial source chunks with full-K cuBLASLt dX
  on a separate stream, retaining native K order. All splits8/16/32 and
  two/four chunks preserve GP/dX bits and pass strict/graph checks. Both
  overlapping and serial chunks regress. The best tested overlap, splits16
  and two chunks, gives BWD1857.79 ->1885.89/full2722.29 ->2750.22 us.
  No source/DX or forward-save candidate was selected from these jobs.

- **18600** diagnostic only: disabling GP publication while preserving exact
  local dW lowers D256 short source554.91 ->524.90 us. This is not a valid
  complete backward path; it bounds the saving available to a recomputed-dX
  alternative. The additional recompute implementation is not pursued.
- **18601** retains previous local dW in flight while issuing the next
  preactivation. Previous-input credit is returned after wait_group1, before
  waiting for the new preactivation. Both full-store and read-completion
  waits preserve GP/partials/strict gradients, REG128/STACK0, but regress
  full2699.14 ->2728.74 and2705.73 ->2731.74 us. Not selected.
- **18602** removes redundant barriers after per-thread TMA acquire and/or
  before per-thread proxy fencing. All pilots are bitwise/strict, but full
  gains are marginal (both2704.37 ->2700.74 us). No selection or sanitizer
  qualification, and the current checkpoint retains its original barriers.
- **18603** keeps the original256-thread producer/consumer source and tests
  one N256 dW instruction with ordinary/constant-offset descriptors. Both
  cannot compile at the two-CTA128-register bound; PTXAS requires154.
- **18604** tests CUDA VMM inline-compressible allocations in a scoped
  PyTorch MemPool. Identical kernels and tensor layouts run on independently
  allocated old/new plans. The allocator checks actual allocation properties
  and records compression support for every allocation. This is an
  unqualified pilot; no global allocator or engine dispatch is changed.

- **18604 completed**: every scoped CUDA VMM allocation reports the
  requested compressible attribute (24/22/25 allocations at D256/384/512).
  All forward and weight/dX outputs are bitwise, affine gradients strict,
  and CUDA Graph replay passes. Hardware compression provides no gain for
  these tensors: short full2709.79 ->2714.16,4689.41 ->4709.74, and
  7239.54 ->7265.42 us. No allocator change is selected.
- **18607** keeps two-CTA aliased shared storage with three compute groups,
  preactivation BF16 register packing, and two half-column dW consumers.
  N128 opcodes cannot compile at80 registers (minimum90). **18608** changes
  each consumer to two N64 opcodes; GP and partials are bitwise, strict
  gradients pass and two CTAs are resident, but stack416/480 bytes prevents
  timing/selection of these variants.
- **18609** instead splits dW columns across separate CTAs, duplicates the
  projection, and aliases dL with GP in one73,984-byte input slot. This
  targets three CTAs with32/128 producer/consumer register budgets. Both
  float and BF16-packed preactivation retention are unqualified pilots.

- **18609 completed**: both float and BF16-packed retention versions achieve
  three resident CTAs, REG80/STACK0, bitwise GP/partials and strict gradients.
  Single-slot latency and duplicated projection dominate: source534 ->1542
  us, full2678 ->3778 us (float); packed behaves similarly. Rejected.
- **18610** screens a new lossless byte-plane format at the GP consumer.
  Sign/mantissa remain8 bits; the exact8-bit exponent is stored separately
  in hardware-compressible memory, optionally XOR127. Unlike BF12, decoding
  never branches to an escape buffer. Every65536 BF16 bit pattern must
  round-trip. Original retired-slot prefetch is retained, and an explicit
  barrier prevents output expansion from overwriting another byte reader.
  Packing is currently separate and outside backward timing; no full-workload
  improvement is claimed without eventual forward integration.

- **18610** byte-plane decoding passes all65536 patterns, bitwise GP/dX/weights
  and strict gradients at both widths, REG127/STACK0. Timing then hits a
  host allocator cache-release segmentation fault after the temporary decoder
  MemPool owner leaves scope. There is no timing result from18610. **18612**
  retains decoder and candidate pool owners through every benchmark cache
  release, then reruns the same consumer screen.

- **18612 completed** with retained pool owners: byte-plane consumers remain
  bitwise and strict, but regress. XOR0 D384 GP845.66 ->884.86 us and
  BWD2980.45 ->3082.85; D512 GP1127.22 ->1168.94 and BWD4628.18 ->4762.16.
  XOR127 is slower still. No forward integration.
- **18614** initially used a128B-swizzled map with a16-byte fastest box
  dimension; the GPU reports illegal access. **18617** uses non-swizzled
  input/output slabs, is bitwise/strict and lowers registers to95/96 without
  spills, but regresses GP by about28% at both widths. Rejected.
- **18616** profiles the original and byte-plane GP consumers. Reported
  DRAM reads are2.11684/2.11321/2.11464 GB (original/XOR0/XOR127), writes
  about1.19 GB throughout. Opt-in compression did not reduce reported
  traffic meaningfully; profile durations are1.139/1.259/1.295 ms. These
  profile durations are diagnostic, not paired selection evidence.
- **18619** tests contiguous48-byte lossless BF12 blocks for32 values.
  A4D TMA box zero-fills the fourth16-byte slab into64 shared bytes, matching
  a warp's output ownership and allowing local expansion without retaining
  every output in registers. Full BF16 escape fallback remains available.
  This is a consumer-only pilot, pending correctness and timing.

- **18619 completed**: padded48-byte BF12 blocks preserve all65536 decoder
  cases, bitwise GP/dX/weights and strict affines, REG128/STACK0. Nevertheless
  D384 GP845.15 ->1189.73 us/BWD2936.45 ->3471.49; D512 GP1126.77 ->1562.03
  us/BWD4599.10 ->5348.50. Rejected before forward integration.
- **18621** uses three exact planes: sign/mantissa, low exponent nibble,
  and high exponent nibble after subtracting116 modulo256. The high nibble
  is normally zero and resides in compressible memory. Reconstruction has
  no escape branch or indirect original-preactivation read. This remains
  a consumer-only pilot; all65536 BF16 patterns and full gradients are gated.

- **18621 completed**: branch-free exponent-nibble planes preserve all65536
  BF16 patterns and strict/bitwise GP, dX and weights. REG128/STACK0, but
  D384 GP845.41 ->962.72 us/BWD2956.67 ->3180.96; D512 GP1127.76 ->1274.24
  us/BWD4615.31 ->4913.36. Rejected before forward integration.
- **18623 completed**: dedicated fifth-warp output-LN loader is strict and
  bitwise in dTriangle. D256 REG127/STACK0 regresses; D384 row16 is REG157/
  STACK0, isolated LN285.23 ->264.62 us but full4693.52 ->4693.89 is neutral.
  D384 row8 and D512 row8 spill80/88 bytes and are not timed. A follow-up
  gives the consumer warpgroup an explicit register budget and a complete
  producer warpgroup; only nonspilling variants will be timed.

- **18626** dynamic LN register pilot stalled because the D256 consumer's
 136-register request exceeded the per-CTA pool implied by REG80. That
 experiment array alone was cancelled. **18628** uses128 consumer registers
 at D256 and224 at wider dimensions; all are strict/bitwise and STACK0.
 Full is neutral/slower everywhere; D512 row8 is substantially slower. Rejected.
- **18631** adaptive exponent bases per16 values reduce escape-block fraction
 to0.058%. All65536 BF16 decoder patterns and GP/strict gradients pass, but
 STACK40 prevents timing. A follow-up pads compressed32-value slabs for local
 expansion, keeps bases outside those slabs, and permutes encoded slab pairs
 so original128B-swizzled GP stores remain safe. No forward integration yet.

- **18633** padded adaptive exponent pilot passes the standalone decoder but
 initially fails GP because its encoder uses the flattened storage width
 instead of logical N for row-dependent slab permutation. **18635** passes
 all bits and strict gradients with logical N, REG125/STACK0. It still
 regresses: D384 GP845.49 ->1028.56/BWD2969.04 ->3257.79 us; D512 GP1127.60
 ->1340.64/BWD4647.04 ->5019.47. Rejected before forward integration.
- **18637** D256 joint-row source lets two row owners share GP while each
 owns half the dW input columns, retaining original K order. Both224/160
 consumer register budgets give STACK0, bitwise GP/partials and strict
 gradients. One CTA/SM; full2721.30 ->2778.77 us at224 registers, similar
 at160. No source candidate is selected.

- **18638** output-only first WGMMA accumulator initialization is bitwise,
 strict and STACK0, but source/full times are neutral. Not selected.
- **18639** joint-row source splits4/16 preserve GP bits and strict gradients
 with matching reducers. Full2721 ->2766/2803 us; not selected.
- **18640** single-buffer joint rows alias dead dL with GP and obtain two
 CTAs, REG128/STACK0. GP/partials are exact but full2712 ->2927 us; early
 packed preactivation is similarly slower. Rejected.
- A full-width dNorm/LN pilot uses four compute groups over H512/H768,
 TMA statistics and whole-tensor triangle/output transfers. Unlike the prior
 streamed design it covers all columns in one pass. Accuracy, resources,
 and complete timing remain unverified.

- **18641** full-width dNorm/LN is bitwise in dNorm/dTriangle and strict.
 D256 has no spills but combined stage274.82 ->556.66 us/full2679 ->2957;
 D384 normal kernel has STACK16 and is not timed. A D256 follow-up separates
 TMA issue from three compute groups (N192/N192/N128) and retains exact K/LN
 order. This tests operand latency rather than assuming fusion is faster.

- **18643** independent producer plus N192/N192/N128 D256 consumers is strict
 and bitwise in dNorm/dTriangle. The timed kernel is STACK0 (the diagnostic
 emitting dNorm has STACK48), but combined stage274.16 ->601.41 us and
 full2675.46 ->2996.88. A row64 LN variant next reuses the entire retired
 input area, avoiding four separate triangle transfers/transposes/stores.

- **18644/18646** row64 output LN is strict/bitwise. D256 synchronous stage
 274.05 ->521.58 us/full2677.26 ->2920.26, async stage275.20 ->526.70/full
 2681.66 ->2928.62. D384 still spills8 bytes and is not timed. Not selected.
- **18647** per-CTA cycle diagnostics preserve dTriangle and STACK0. Sync
 median GEMM14668, triangle load/transpose4370.5, separate affine7885,
 LN/transpose11683, store/read-wait2206 cycles. Async GEMM12090, load5032,
 affine12131, LN12496, store2193. These are diagnostic cycles, not wall time.
 The next candidate integrates affine accumulation into the LN row pass,
 with optional one-map whole-weight TMA to reduce GEMM issue overhead.

- **18648** integrating affine sums in the row LN pass preserves all strict
 gradients/dNorm/dTriangle, STACK0. Async full-width stage improves to436 us
 but remains slower than276 us separate, full2689 ->2844. Whole-weight TMA
 alone is neutral. No selection.
- **18652** two compute groups/half-width streamed weights achieve two CTAs,
 REG128/STACK0 and bitwise dNorm/dTriangle. Full2693 ->2810 us; not selected.
 The next resident-register dP pilot applies whole-map transfers and shared
 gamma to the older slim architecture, and tests one RS-N128 instruction in
 place of two RS-N64 instructions with packed BF16 publication.

- **18654** modern slim RS-N128 passes bitwise dNorm/dTriangle and all strict
 gradients, REG128/STACK0, two CTAs. Stage277.97 ->303.46 us/full2696.72
 ->2723.84 remains slower. The otherwise identical two-RS-N64 variant spills
 56 bytes and is not timed. A follow-up stores affine accumulators only
 between persistent row tiles, shortening their register lifetime during
 GEMM and permitting a two-RS-N128/N256-width experiment.

- **18656** scoped affine accumulators make N128 and two-N128/N256-width
 slim kernels STACK0 with two CTAs and bitwise dNorm/dTriangle. N256 stage
 278.45 ->291.44 us/full2698.32 ->2709.82 remains slower. N128 is slower
 still at294.58 us. A new transpose assigns each complete64x32 shared tile
 to one warp, removing its cross-warp exchange, while retaining the final
 consumer barrier. Test both standalone qualified LN and the slim pilot.

- **18658** one-warp whole64x32 output-LN transposes are strict/bitwise,
  REG128/STACK0, but standalone181.55 ->180.58 us and full2726.16 ->2719.65
  are only a small pilot gain. **18659** applying this to slim DN/LN is slower.
- **18661** per-warp weight-credit arrivals, optionally removing acquired-data
  and disjoint dNorm-store barriers, pass all six strict pilots with no spills.
  No full improvement; no synchronization change selected or sanitized.
- **18667** modern gate/DN/LN fusion preserves DP, DG, dNorm, dTriangle and
  strict gradients but spills. **18669** bounded64-channel gate fragments
  remove timed spills; both N128/N256 remain slower: full2686 ->2748 us.
- **18671** a separate preactivation producer still cannot compile N256 dW
  under two-CTA bounds: PTXAS needs154 registers versus128. No GPU result.
- Next pilot delays independent output dW until contraction, source, or dX
  starts, with separate workspaces and explicit graph event joins.

- **18672** delays both output dW GEMMs to contraction/source/dX boundaries.
  All six candidates are strict and graph-correct with separate workspaces.
  Source boundary with weights queued first gives the best pilot: backward
  1859.84 ->1839.97 us, full2728.26 ->2711.36 us. Not yet qualified.
- **18678** uses two disjoint CUDA green contexts. An SM-ID sampling kernel
  confirms disjoint resources both eager and after CUDA-graph replay. All
  arithmetic and graph checks pass, but every8/16/24/32-SM partition regresses.
- **18682** fits source CTA counts to the remaining SM resources: splits7
  at16/20 side SMs and splits6 at32 side SMs. All strict/graph checks pass;
  input-weight errors are about2.5e-4, but full times still regress. Rejected.
- **18685** standalone128x256 prefix dX stopped during compilation because
  its NSLOT macro collided with upstream headers. **18688** fixes that name
  and tests two/three/four input slots, whole3D TMA maps, and one outstanding
  ordered WGMMA group. This is a new pilot, not a changed checkpoint.

- **18688** full-width prefix dX is strict/bitwise and nonspilling. Four TMA
  slots reduce the new stage from673 us (two slots) to308 us, still slower
  than selected Lt278 us. Three slots give376 us. No selection.
- **18691** K64/depth0, K64/depth2 and K48/six-slot depth1/2 are also strict
  but slower. K32/nine-slot tasks initially failed because18 barriers exceed
  the128-byte barrier allocation. **18700** allocates rounded barrier storage;
  both K32 candidates are strict/nonspilling but still slower,356-370 us.
- **18697** prefix dX/input LN reuse shared dXn, with per-CTA affine partials
  and final affine/weight reductions. All dXn/dX/weights are bitwise and affine
  errors are strict, REG168/STACK0. Combined stage399 ->460/475/539 us for
  wait depths0/1/2; full regresses. No qualification or dispatch change.
- **18704** uses the existing input-statistics saves in that fused epilogue.
  dX error5.34e-6, remaining gradients strict, no spills; depth0 stage399 ->443
  us, full2694 ->2745 us. **18707** reuses retired GEMM slots for32/64-row
  LN transfers, with exact dX and no spills, but also regresses full.
- **18713** measures diagnostic per-CTA GEMM/epilogue/store cycles and separates
  final reduction launch costs. **18714** repeats selective output-dW overlap
  at the source boundary for dWproj, dWgate, and both. Pending evidence only.

- **18713** diagnostic phase clocks show input-LN/affine epilogues add about
  22-26K cycles to a52K-cycle GEMM. Separate final reductions cost about42 us.
- **18718/18719** parallel affine reduction and packed input-weight finishing
  reduce that overhead, retaining strict gradients and zero spills. Best
  combined dX/LN398.50 ->418.08 us/full2695.06 ->2711.78, still slower.
- **18714/18715/18717** three repeated pairs confirm only delaying both output
  dW GEMMs helps: backward improves about18-20 us, full about17-18 us. Delaying
  either alone regresses. **18724** passes all five fixtures, graph/PyTorch
  and all three sanitizers at both lengths. **18726/18727** directly compare
  to Triton: short1.2187x BWD/1.3148x full, long1.6678x/1.6502x. Goal unmet.
- **18728/18730/18731** N128 two-row-consumer prefix dX with a single loader
  warp has90-95 registers, no spills and two resident CTAs. All gradients
  and dXn are bitwise/strict, but best dX279.28 ->322.48 us and full2736.34
  ->2776.51. Rejected. Next pilot swaps GEMM axes to let each warpgroup own
  64 input channels by256 spatial rows, retaining exact ordered K arithmetic.

- **18735/18736/18737** transpose native dX output axes to128 input channels
  by256 spatial rows. All dXn/gradients are bitwise/strict with no spills,
  but dX335-340 us versus278-282 us; full slower. Rejected.
- **18738** wide late-dW setup fails before timing because its output dWproj
  dispatch is torch.mm, not the unused Lt descriptor. **18751** preserves
  that actual operation. All12 strict/graph pilots pass, but only D384 dX
  boundary/weights-first has a small full pilot gain (~8 us). Other D384
  and every D512 ordering regress. No qualification or selection.
- **18763** packed BF16 retention through output LN preserves dTriangle bits
  and strict gradients. D512 row16/two-CTA stage404.67 ->389.66 us, full
  7269.54 ->7229.10; D384 row16/prefetch full4693.95 ->4672.08 despite a
  slower isolated stage. Three-CTA D512 spills136 bytes and is not timed.
  Repeat full-workload evidence is required before selecting either gain.
- **18769** tests pure fixed-length WGMMA contractions, omitting the old
  fused-GP epilogue. One/two compute warpgroups and three CTA orders retain
  the original K arithmetic, writing ordinary dL/dR for the current source.

- **18769** pure contractions are strict with bitwise dL/dR and zero spills.
  Pairing left/right channel tiles improves locality: one-group stage
  330.08 ->315.46 us/full2738.27 ->2721.68; two-group stage329.90 ->305.22
  us/full2736.48 ->2718.42. Grouped-by-mode grids are slower. These pilots
  include the qualified late-output-dW schedule in both comparison arms.
- **18774/18776** repeat packed LN three times. D384 full gains19-26 us,
  D51235-41 us, all strict/bitwise dTriangle. **18777** qualifies checkpoint20
  at both widths and lengths, including D512 overflow branches; **18779**
  is its dependent direct Triton comparison. Results remain pending.
- **18781** uses whole3D TMA maps in the native contraction and tests two
  versus three input slots with one/two compute groups. **18783** bounds
  LN scalar intermediates within explicit exact FP32 PTX steps, testing
  whether the packed D512 path can reach three CTAs without spills.

- **18777 complete**: checkpoint20 passes five strict fixtures, graph replay,
  independent PyTorch and all three sanitizers at D384/D512, both lengths.
  D512 normal and both forced overflow paths have zero errors/hazards/warnings.
  **18779** directly compares all four shapes: short D3841.3853x/1.3816x and
  D5121.3149x/1.3660x BWD/full; long D3841.5315x/1.5115x and D5121.5082x/1.5096x.
  Short shapes remain below1.5x. No production dispatch change.
- **18781** whole-map pure contractions are all bitwise/strict and nonspilling.
  Best remains two groups/three slots: stage329.09 ->300.98 us, full2748.82
  ->2728.82. Two slots regress, including the four-CTA single-group variant.
  **18784** repeats original and whole-map winners; **18785** tests N192.

- **18783** explicit PTX scalar LN steps preserve bitwise dTriangle and strict
  gradients but do not lower registers or spills: two-CTA REG216/STACK0,
  three-CTA REG168/STACK136. They give no improvement over the already
  qualified C++ packed operand path; retain checkpoint20.
- **18784/18795** three repeated pairs confirm the native D256 contraction
  gain. Whole-map128x128 with three slots saves18-20 us backward and15-19 us
  F+B, with bitwise dL/dR and strict gradients. Original2D maps save14-16 us
  full. **18785/18796/18797/18798** N192 tiles are also bitwise/strict and
  nonspilling, but no configuration beats the whole-map128x128 winner.
  `d256_contract_checkpoint.py` packages that winner for full qualification,
  poisoning dL/dR, GP and dXn before every fixture and graph replay.

- **18799** D256 native contraction checkpoint passes both lengths: all five
  strict fixtures, independent PyTorch, graph replay with poisoned dL/dR/GP/
  dXn, and memcheck/racecheck/synccheck. **18801/18803** direct Triton ratios
  are short1.23545x BWD/1.32953x F+B and long1.67089x/1.66113x. Goal unmet.
- **18802/18805** reuse the same native primitive for the two forward products.
  Triangle and all derivatives are bitwise/strict, zero spills, but full
  regresses9-18 us. Retain existing forward.
- **18804** applies warp-local transpose synchronization to packed row16 LN
  while retaining its final CTA barrier. Three repeat comparisons per width
  measure the interaction with checkpoint20; no selection before results.

- **18804/18807** warp-local synchronization in the packed LN transpose is
  strict/bitwise with unchanged resources and no spills. D384 short F+B
  improves10.59/8.93/9.50 us in three pairs; D512 improvements are small.
  `wide_checkpoint21.py` selects D384 L384 only, retaining a final CTA join.
  It still requires all synchronization sanitizers before qualification.

- **18808/18811** checkpoint21 D384 passes all fixtures, graph/PyTorch and
  all three sanitizers at both lengths. Direct short1.38951x BWD/1.38523x
  full; long1.52231x/1.50339x. The six-shape goal remains unmet.
- **18814** pure output projection dW uses two row warpgroups, whole TMA,
  FP32 split partials and a packed BF16 finish. Initial8/16/32-split pilots
  preserve strict gradients and have no spills, but do not improve full.
  **18825** tests full N256 WGMMA and grid-aligned split counts; D384 uses
  14/28 uneven intervals, with exact coverage and unchanged per-interval K order.

- **18825** N256 output dW passes strict gradients, REG154/STACK0 and one
  resident CTA. Uneven splits cover every K tile and D384 weight errors are
  below2.1e-4. No full gain at any shape; no selection. A three-compute-group
  M192N256 follow-up uses the remaining register capacity in that CTA. D512
  pads only partial output rows to576; TMA zero-fill handles the extra64 rows,
  and final reduction reads only the valid512 output rows.

- **18832/18833-18837** three-group M192N256 output dW is strict and
  nonspilling (REG154), but all splits regress full. No selection. The next
  D256 fixed-contraction pilot uses pairs of M64 row CTAs and whole-map B
  multicast. Per-slot remote arrivals ensure both destination transaction
  barriers are armed; the final cluster join preserves shared lifetimes.

- **18838/18839** M64N128 paired-CTA contraction multicast is bitwise/strict
  and nonspilling, but stage301 ->506/516 us and full regresses. Rejected.
  Next pilot initializes output-affine gradients in the preceding gate kernel
  and launches output LN without its initial grid barrier. One/two/four waves
  retain row arithmetic; complete workloads include the changed gate kernel.

- **18840/18841-18848** early output-affine initialization is strict with
  bitwise dTriangle and no spills. D256 two/four waves saves7-8 us full;
  D512 four waves saves8.6 us; D384 is near neutral. Follow-up also initializes
  input affines early, removes the input-LN grid barrier and tests packed
  exact-order weight reduction. These remain unqualified pilots.

- **18861** ordered two-warp LN setup stopped at an overbroad host string
  assertion. **18871/18875-18877** fixes the kernel-scoped match. dTriangle
  and all main gradients are bitwise, affines strict, no spills. D384 is
  neutral/slower. D512 reaches three CTAs (REG155), and two pilots save47-51
  us full despite slower isolated LN. Repeat combined early initialization
  with ordinary and pair-local barriers before any selection.

- **18865/18866-18874** repeats early affine initialization. D256 full saves
  18.86/14.06/20.63 us; D3843-5 us; D51215.55/6.25/27.07 us. D256 candidate
  uses output-grid2x, input-grid1x, packed exact-order weight finishing.
  `d256_early_affine_checkpoint.py` freezes it for strict/graph/PyTorch and
  all three sanitizer gates, poisoning all four affine buffers each replay.

- **18878/18879-18883** combined ordered D512 LN plus early affine init
  passes strict gradients, bitwise dTriangle, zero spills. CTA-barrier full
  gains83.87/70.01/78.85 us; pair-local gains74.16/71.86/85.82 us. Retain CTA
  barriers because the full gain is not larger with narrow barriers.
  `wide_checkpoint22.py` packages D512 short ordered LN/early init and D384
  short early init. Long shapes are unchanged. Full qualification is pending.

- **18884** D256 early-affine checkpoint passes both lengths: all five
  poisoned-affine fixtures, graph replay, independent PyTorch, memcheck,
  racecheck and synccheck. **18886/18888** direct Triton comparison gives
  short1.24344x BWD/1.33302x full; long1.67619x/1.67300x. Goal unmet.
  **18887 ->18889** validates/compares wide checkpoint22; **18892** then
  tests packed two-warp row8/16 LN, followed by contextual profile18894.

- **18887 short D512 fails racecheck**: one warning reports a shared WAR
  between prefix-sum reads and lane0 final-sum overwrite in ordered LN.
  Numerical passes do not qualify it. Pending comparison18889 and descendants
  18892/18894/18895 were confirmed pending and cancelled; active long18887_3
  was preserved. Checkpoint22 is rejected. `wide_checkpoint23.py` writes final
  sums into a separate128-byte area, retaining the CTA barriers and arithmetic.
  Repeats and the entire four-shape validation chain will be rerun. The packed
  row8/16 and D256 pilots also receive disjoint storage before resubmission.

- After identifying checkpoint22 as superseded, its remaining own long task
  **18887_3** was cancelled to avoid repeating inherited overflow validation
  before the replacement. No unrelated job was cancelled. Safe repeat18896
  precedes full checkpoint23 validation18898. All four shapes and both D512
  overflow branches will be checked again on the fixed source.

- **18896/18897/18899** disjoint-sum D512 repeats retain all bits/strict
  gradients and zero spills. Full gains80.61/63.36/81.32 us. Qualification
  **18898** and direct paired comparison**18900** remain pending. Follow-ups
  are serialized: packed row8/16**18902**, contextual NCU**18903**, D256
  ordered pair**18905**, single-warp nonintegral-wave grid control**18907**.

- **18898/18900/18910-18912** checkpoint23 is fully qualified at all four
  shapes, including both D512 overflow paths. Current ratios are in the table.
  L384 remains below1.5x everywhere; no dispatch change.
- **18902/18913-18915** packed ordered LN is bitwise/strict and nonspilling;
  row8 reaches four CTAs but adds about1.07 ms full, row16 adds27-40 us. Reject.
- **18905/18916-18918** ordered two-warp D256 LN is bitwise/strict with no
  spills, but every row16/32 and barrier variant regresses full. Reject.
- **18907/18919/18920** one-warp D512 grids396/660/792 are bitwise/strict
  but slower full than qualified ordered LN. Reject.
- **18908/18921-18926** wide native dX with N192/N256, four/five slots and
  wait depths0/1/2 is bitwise/strict and nonspilling. All full times regress.
- **18903** NCU recorded no kernels because its filter used sanitizer-style
  regex= syntax. Corrected to NCU regex: before resubmission; no profile
  conclusion is drawn from the missing report.

- **18928** corrected NCU context profile observes output LN445.25 ->403.17
  us, with three rather than two resident CTAs; DRAM reads628.6 ->667.8 MB.
  This is diagnostic, not a replacement paired timing.
- **18931/18933** three LN chunks with row-local even contractions preserve
  dTriangle/dL/dR bits and strict eager/graph gradients, with no spills.
  Splitting alone adds163 us full; overlap still adds148 us. Reject.
  Next full-N384 contraction tile tests larger input reuse without chunking.

- **18938/18939/18941-18944** full-N384 D256 contraction preserves bits and
  strict gradients. M128, three input slots, N256+N128 yields a full gain.
  Three repeats **18948/18949/18951** save17.54/19.81/18.03 us full and
  19.06/17.17/18.13 us BWD. REG227/STACK0, one resident CTA.
  `d256_spatial_checkpoint.py` freezes this candidate for complete gates.
- **18950/18952/18954-18958** column warpgroup splits are slower; the
  M128N192-per-group version spills8 bytes. Six-group and two-resident
  three-group versions cannot compile within their register budgets. Rejected.

- **18963/18965/18968** D256 spatial checkpoint passes all strict/graph/
  independent PyTorch and three sanitizer gates at both lengths. Short direct
  BWD2.22310 ->1.75560 ms (1.26629x), full3.54155 ->2.62048 (1.35149x).
  Long1.69795x/1.68128x. The all-six target remains unmet.
- **18969** wide full-spatial pilot stopped before candidate setup because
  PREFIX_COPY=tma was missing. Corrected **18973/18974-18976** is bitwise GP,
  strict and nonspilling, but roughly doubles contraction time. Reject.
  Follow-up stages the mask through TMA and tries one-warpgroup N192 at
  three/two resident CTAs; the N384 mask-only control isolates that change.

- **18979/18980-18984** TMA mask restores much of full-N384 performance,
  but all N192/N384 one-row-group variants regress. GP is bitwise/strict,
  no spills. **18985/18986-18993** N64/N128 variants achieve3-6 resident
  CTAs without spills, but also regress full. No wide candidate selected.
- **18994/18995/18996** D256 outstanding ordered WGMMA groups are bitwise
  and strict without spills. Later prefetch costs more than overlap saves;
  full regresses23-42 us. Rejected. Next M192N192 tile balances A/B reuse
  across three compute groups and tests two/three/four input slots.

- **19001/19002/19006** M192N192 contractions are bitwise/strict and
  nonspilling; four slots nearly match the selected M128N384 stage, but
  every complete workload regresses. No selection.
- **19025/19026-19030** cache-sized GP chunks (8/16/24 spatial regions)
  retain GP/dX bits and strict eager/graph gradients with32-192 weight
  splits. Full regresses0.36-0.83 ms after all partial reduction costs.
- **19037** reused-buffer pilot stopped on an overbroad source rewrite
  assertion before the new source launched. **19054/19055-19060** narrows
  that rewrite, reuses compact GP storage, and selects a bitwise compact
  dX algorithm. Strict and graph checks pass, but all full workloads still
  regress. L2 evict-last hints give only a small diagnostic reduction.
  Next transposed dX uses one N256 compute group and one loader warp to
  permit two resident CTAs without dynamic register redistribution.

- **19068/19069-19071/19073** single-compute-group transposed D256 dX is
  bitwise/strict with zero spills and reaches two CTAs for the small stages.
  Every complete workload regresses. **19080/19081-19085** confirms the
  same result for wide dimensions. No dX replacement selected.
- **19101/19102-19108** staggering paired source consumers is bitwise in
  GP/partials and strict, with no spills, but all full times regress.
- **19112/19113-19115/19117-19120** whole-map paired source with three
  input slots is at best near neutral (about1 us full difference). All
  variants retain GP/partial bits and strict gradients, REG168/STACK0.
- **19123/19124/19126-19131** retains dW while issuing next projection.
  It is also bitwise/strict with no spills; three-slot variants are neutral
  or slower and two-slot variants regress about60 us. No source selection.
  **19138** tests GP fragments loaded once into registers for both dW halves;
  fragments remain live through the asynchronous MMA completion.

- **19138/19139-19141** register-A GP reuse is bitwise and strict, with no
  spills, but every complete workload is slower. No source change selected.
  The next source pilot packs exact tanh-to-sigmoid ULP corrections into
  a 1KB shared table; all 65536 BF16 inputs must match before timing.

- **19153/19154** compressed sigmoid correction matches all65536 BF16
  inputs, GP/partials and strict gradients, REG128/STACK0. The additional
  1KB shared table reduces residency to one CTA; source/full regress heavily.
  Rejected. **19168** inspects N256 dW compiled at the relaxed launch bound,
  including SASS register lifetimes and ELF allocation metadata. No candidate
  execution or resource-metadata change is implied by this inspection.

- **19168** SASS confirms registers above127 are used only after the consumer
  acquires224; the producer uses at mostR5 after releasing to32. **19177/19178**
  initial register-pool metadata224 ->128 preserves machine code and all GP/
  partial bits, strict gradients and zero spills. Residency becomes two CTAs;
  full gain is only about4 us in one pair, not a qualified selection.
- **19184/19185/19187/19190** one full-width dX compute group plus one loader
  warp achieves two CTAs with REG154/STACK0 and bitwise dXn/dX, but all fused
  dX/LN complete workloads regress. No change selected.
  Next pilot prefetches input-LN data into retired GEMM slots and double-buffers
  the row epilogue, retaining ordered math and the fast affine/weight finish.

- **19195/19196/19197** prefetched input-LN keeps bitwise dXn and strict dX
  (saved-statistics error5.34e-6), with no spills. Best combined stage remains
  about11 us slower, and every full workload regresses. No selection.
  **19208** diagnoses whether reassociated dX differences concentrate near
  BF16 boundaries. Reference-value substitution is an oracle diagnostic only,
  not an implementation or timing claim.

- **19208** rounding diagnostic: split2/3/9 dX differs at39-45K of37.75M
  BF16 elements. Oracle repair within128 FP32 ULPs of a midpoint reaches
  strict dX (1.55-1.72e-5), but selects about149K elements across94K rows.
  A real repair implementation is not assumed cheap; no performance claim.
- **19210/19214-19216** N256 pending-dW overlap is bitwise/strict but slower.
  Read-completion wait alone is modestly favorable, about5.5 us full.
- **19221-19224** N256 consumer budgets192/200/208/216 and matching initial
  pools112/116/120/124 preserve GP/partials and strict gradients, no spills,
  two CTAs. Full pilots save4.00/3.63/7.76/5.07 us. These are not repeated or
  fully qualified selections; the spatial checkpoint remains unchanged.

- **19227/19228/19229/19237** read-wait plus budget168 is stopped by the
  prelaunch SASS guard (entry R154 exceeds the proposed100-register pool).
  Budget184 spills128 bytes and is not timed;192/208 are strict/nonspilling
  with small gains, but read-wait adds no clear gain over full completion.
- **19238/19239/19241** three repeats of budget208 with full TMA completion
  save7.13/6.11/8.12 us F+B and8.53/7.73/8.78 us BWD. The candidate is now
  frozen as `d256_pool_checkpoint.py` for full correctness/sanitizer gates.
  This is a small improvement and leaves the all-six1.5x goal unmet.

- **19247/19248** pool checkpoint passes all five strict fixtures, poisoned
  intermediates and graph replay, independent PyTorch and all three sanitizers
  at both lengths. **19249/19254** directly measure short1.26826x BWD/1.35310x
  F+B and long1.69919x/1.66209x. Report authority now includes pool checkpoint.
  **19259** follows with N192 contraction using K32 and64B-swizzled inputs,
  testing additional resident compute groups while preserving ordered K.

- **19259-19262** K32/N192 contractions preserve dL/dR bits and strict
  gradients, with no spills. Two/four resident CTAs are achieved, but full
  regresses25-153 us. Follow-up retains outstanding MMA groups and tests a
  full N384 one-group tile using64B swizzles at two resident CTAs.

- **19270-19275** K32 outstanding-MMA and full-N384 one-group variants are
  bitwise/strict and nonspilling, but all full workloads regress. Reject.
- **19276** current D256 short NCU: dX285.95 us/tensor85.78%; contraction
  311.42 us/tensor49.75%/barrier stall9.10; source594.98 us/tensor67.89%.
  Contraction reads453.1 MB and writes286.0 MB. These are profiler diagnostics,
  not additive or replacement full timings. Next experiment separates a TMA
  producer from the two full-N384 consumer groups, with checked32/232 roles.

- **19278/19279/19287** dedicated D256 contraction producer initially stops
  before launch: the conservative SASS guard rejects an indirect jump table.
  Explicit PTX predicates retain direct dispatch branches. **19290/19291**
  then pass bitwise dL/dR and strict gradients at32/224 and32/232 register
  roles, zero spills. Contraction saves4-5 us, but F+B does not improve.
  No selection; the selected initial-pool helper is unchanged.
- **19296-19299** one loader warp plus two wide contraction consumers retains
  two CTAs, REG90/STACK0, bitwise GP and strict gradients. Full changes are
  small: D384 saves about1 us; D512 runtime-loop saves11 us while unrolling
  is neutral. **19303/19304/19307/19308** whole-map transfers retain the same
  resource usage and bits, with only small full gains. Not yet qualified.
- **19306** current D512 short NCU: input dW881.1 us/tensor94.98%, dX961.9 us/
  tensor98.04%; contraction/GP1138.4 us with2.117 GB DRAM reads and1.192 GB
  writes. This is diagnostic profiling, not an additive F+B measurement.
- **19311-19314** M128N192 whole-map/direct-mask contractions retain two
  CTAs, REG122/STACK0, bitwise GP and strict gradients, but every full workload
  regresses340-465 us. Rejected. The next bounded pilot measures only direct
  GP spatial compaction for zero masks, including map/padding costs. It has
  no complete-workload speedup claim or consumer integration yet.

- **19318/19319** compact-mask setup stops before launch on a ctypes integer
  argument unsupported by the launch wrapper; plain integer arguments fix it.
  **19321/19322** compact GP and zero padding are bitwise, REG90/STACK0/two
  CTAs, but scalar scatter is about4x slower. **19323/19324** shared compaction
  with aligned vector stores is about3x slower. **19325/19326** bulk TMA row
  interiors improve that to2.7x slower. These producer-only candidates are
  rejected without attempting consumer or conditional-graph integration.
  All retain the dense-capacity overflow design, but its full qualification
  is not claimed. Selected kernels and speedup authority remain unchanged.
- **19330** next D256 source pilot separates a loader warp, projection/GLU
  group and N256 dW group.32/80/160 register roles with112 initial registers
  target two CTAs. Both dW completion and GP-store completion protect slot
  reuse. A separate conservative resource helper guards this experiment;
  the qualified source and its helper are unchanged.

- **19330/19331** loader source with112 initial registers reaches only one
  CTA; no candidate launch. **19332/19340** N256 dW needs at least154 compiler
  registers, so144/152 targets do not compile. **19334/19336/19339** lowering
  the projection role too far causes PTXAS to discard dynamic allocations;
  the SASS guard rejects those cubins before metadata adjustment or launch.
- **19338** partial-loader320-thread candidate reports two CTAs but stalls
  before a correctness result. Only this job was cancelled. The experimental
  helper now conservatively checks every SMSP allocation as well as the CTA
  sum; this rejects the partial-group budget. **19348** full384-thread24/64/152
  roles with two N128 dW accumulators pass bitwise GP/partials and strict
  gradients, but spill800 bytes and serialize WGMMA; not timed or selected.
- **19341/19342-19344/19346/19347** repeat wide whole-loader GP three times.
  D384 full gains0.32/4.32/6.72 us are too small to justify selection.
  D512 full gains9.12/10.83/14.46 us, BWD14.61/18.48/19.92 us. Only D512
  short is frozen in `wide_checkpoint24.py`; **19351** qualifies both D512
  lengths, including the inherited overflow paths. Authority remains23 until
  all gates and the dependent direct Triton comparison actually complete.

- **19355** queues a D256 two-group source with N256 dW and whole XN/weight
  maps at80/176 and96/160 register roles. This avoids the failed extra loader
  group's register budget; no result or selection is assumed yet.
- **19370** pipelines three128-row spatial GP bands with ordered Lt dX,
  separate consumer workspace and explicit graph events/joins. **19378**
  additionally uses persistent GP producers and an82KB native dX tile, so the
  two workloads can fit shared memory together. **19384** uses two dX compute
  groups with a128x128 tile to reduce duplicate GP operand loads. All are
  pending experiments; full timings and strict graph output checks govern.

- **19351/19352 complete**: checkpoint24 passes all strict fixtures, graph,
  PyTorch and three sanitizers, including both D512 overflow branches. Paired
  **19353/19391** gives short1.31866x BWD/1.34872x full and long1.50383x/
  1.51345x. Only D512 moves to24; the six-shape goal remains unmet.
- **19355/19399** whole-map D256 two-role source is bitwise/strict, zero spills,
  two CTAs, but full regresses26-27 us. Rejected.
- **19370/19400-19404** three-band GP/Lt-dX and dW overlap are bitwise/strict
  with correct poisoned graph replay, but full regresses252-409 us. Rejected.
- **19378/19408-19410** persistent GP and one-group native band dX are exact,
  strict and graph-correct with no spills, but full regresses1.16-2.76 ms.
  **19384/19411/19414/19415** two-group dX reduces that loss, still0.72-2.04 ms
  slower. Rejected. Reduced GP residency and inefficient native dX overwhelm
  the overlap. Next pilot compresses only the mask TMA operand losslessly and
  uses a smaller dX tile, aiming to preserve two GP CTAs plus a dX CTA.

- **19432/19433** lossless byte mask round-trips all65536 BF16 bit patterns;
  GP, gradients and poisoned graph replay pass. REG90/STACK0, two80KB CTAs,
  but GP regresses157-207 us and full231-326 us. **19442/19443** mask/gate
  prefetch keeps correctness but does not improve those losses. Rejected.
- **19438-19441** byte-mask GP with36KB native dX fits two GP plus one dX
  CTA's register/shared budgets. Bitwise GP/dXn and strict graph checks pass,
  but full regresses0.76-1.17 ms. Rejected; no further overlap tuning planned.
- Next D256 source pilot retains the selected N25632/208 two-CTA source.
  Whole3D input/weight maps are a control. Multicast2/4 variants arm each
  input transaction at consumer retirement before sending remote reuse
  credits, removing the old producer-to-producer readiness handshake.

- **19453/19454/19455** selected-source whole maps and prearmed multicast
  are bitwise for GP/partials, strict, graph-correct and spill-free. Whole-map
  control is neutral (full2655.25 ->2654.19 us); cluster2 regresses144 us and
  cluster4 regresses683 us despite two CTA occupancy. No selection.
- **19456** instruments one middle source tile per CTA with device clocks;
  **19457** profiles tensor/SFU/FMA and issue stalls on the selected source.
  These are diagnostics, not candidate performance or speedup authority.

- **19456 complete**: instrumented source is strict and nonspilling. Typical
  per-tile medians are input wait164, projection751-772, GLU602-615, GLU sync
  101-103, dW issue262-271, dW wait799-926, store wait114-136 cycles.
  **19457** selected-source NCU is593.89 us/tensor68.03%, FMA10.96%; diagnostics
  identify arithmetic sequencing rather than input wait as the main limit.
- **19459-19462** adaptive input-credit polling on the aliased two-group
  source is exact/strict, graph-correct, two CTAs and spill-free. Immediate
  dW completion still costs24-26 us full; one pending dW costs286-290 us.
  Rejected. Next candidate divides exact GLU over the existing loader and
  compute warpgroups, reusing retired derivative shared storage for BF16
  preactivations and synchronizing all256 participants before overwrite.

- **19466/19467** exact GLU shared between loader and consumer is bitwise
  for GP/partials and strict.64/192 roles retain two CTAs and no spills but
  full regresses36 us;80/176 spills160 bytes and is not timed. Rejected.
- **19468** tests redundant source joins around proxy fences, acquired TMA
  data and completed WGMMA/stores. Completed variants are bitwise/strict,
  graph-correct, no spills, but full gains are0-2 us; not selected.
- Next contraction uses one M64N384 compute group plus a full loader group,
  two K64 input slots and32/224 or24/232 dynamic register roles. Its114816B
  shared allocation and128 initial registers target two CTAs per SM. This
  differs from the prior K32 one-group and K64 two-compute-group candidates.

- **19476/19477** M64N384/K64 two-slot contraction with loader group achieves
  two CTAs,128 initial registers, zero spills, bitwise dL/dR and strict
  gradients. Full gains0.91/3.28 us are too small to select. **19478/19479**
  one outstanding MMA group preserves bits but regresses full33 us. Reject.
  Follow-up pairs adjacent row CTAs and shares the48KB B input with multicast;
  consumer retirement prearms remote input slots before returning credits.

- **19483/19484** full-N384 B multicast retains132 clusters, two CTAs,
  zero spills, bitwise dL/dR and strict gradients. Full regresses76-78 us.
  Rejected. Next source pilot packs exact BF16 preactivations into registers
  before GLU and groups8/16 independent ex2/reciprocal instructions. It keeps
  the original sigmoid math, multiplication order and selected32/208 pool.

- **19485-19488** early BF16 packing and batched sigmoid instructions retain
  GP/partial bits, strict graphs, two CTAs and no spills, but full regresses
  3-11 us. Rejected. **19492-19496** per-SM alternating CTA startup offsets
  (0-2000 cycles, including counter reset) are also strict/graph-correct but
  give only0-2.4 us full changes; not selected.
- Next D256 gate/dNorm fusion is a new two-slot pipeline: one gate producer
  and two N256 dNorm compute groups, exact tanh gate math and ordered K.
  DP overwrites consumed projection data; DG has separate shared storage.
  Both gradient consumers release each slot, and the producer also completes
  both GP stores before reuse. It retains early input/output-affine zeroing.

- **19513/19514** pipelined gate/dNorm is bitwise in DP/DG/dNorm and strict
  in all gradients and poisoned graph replay. Both register splits are
  spill-free, one CTA, but combined stage228 ->265 us and full +41-45 us.
  Next K32/64B DP slots fit two/three stages; dNorm retains ordered K and a
  final producer-done barrier protects GP stores before shared output reuse.

- **19530-19533** K32 compact gate/dNorm has exact DP/DG/dNorm, strict
  graphs and no spills, but full regresses110-143 us. Rejected.
- **19544/19545** D512 full shared DP with two streamed H512 passes is
  bitwise/strict.64/192 has16 local bytes;80/176 is nonspilling but
  stage507 ->717 us and full +216 us. Rejected.
- Next D256 pilot revisits four groups with aliased two-slot storage,
  whole TMA maps and one N128 dW per consumer. Audited24/56/88/88
  roles target64 initial registers and two resident512-thread CTAs;
  larger-budget controls distinguish allocation limits from scheduling.

- **19549-19552** four-group N128 consumer needs at least90 compiler
  registers.64/96 roles are strict/nonspilling but one CTA and full +190 us.
  **19553-19555** constant slot addresses do not reduce that minimum.
- **19556-19559** N32 projection preserves exact GP/partials;40 producer
  causes PTXAS to discard register redistribution and is rejected by SASS
  audit.48/96 and64/96 are nonspilling but one CTA and slower full.
- **19560-19563** two N64 dW operations allow two CTAs at64/80 roles,
  but416-480B local storage remains. Higher budgets are slower.
- **19564-19567** combined-loader three-group source uses audited initial
  metadata.64/88 still spills416B, whereas N32 projection48/96 has only16B
  local storage, exact GP/partials and two CTAs. A bounded follow-up times
  these strictly correct pilots including their spills, without promotion.

- **19569/19570** explicitly time spill-bearing three-group pilots:416B
  local storage costs +3.1 ms full;16B/two-CTA N32 costs +284 us. Reject.
  **19571-19574** constant producer slots do not improve that result.
- Next source uses the weights as register-A in a transposed projection.
  Cached global loads supply K64/K128 fragments, then an8KB shared
  transpose restores the original GLU layout. Shared storage drops to90KB;
  the original GP and N256 dW arithmetic and all completion barriers remain.

- **19579-19582** weight-register projection is bitwise/strict, but static
  chunks spill280-464B. **19583-19586** runtime K chunks remove spills at
 96/160 roles with two CTAs; full still +0.75-0.77 ms. Rejected.
- **19589** current-source dense projection/native GLU/split dW is bitwise
  for all eight Lt algorithms in each GEMM. Best projection280 us, dW248 us;
  full source1049 vs543 us and full +540 us. Rejected.
- **19596-19599** whole-map A multicast across adjacent N tiles is bitwise,
  strict, graph-correct and nonspilling. Control tile order is neutral;
  cluster2/4/8 regress full318/401/469 us. Rejected.
- Next source defers dW output coordinates until after accumulation using
  an accumulator-dependent zero (shared allocation below1MB), and makes
  consumer iteration counts independent of the output rank/split.

- **19600-19603** output-index deferral retains the N128 minimum90; N64
  still spills416B. **19605-19607** unifying dW consumer code paths does not
  change the limits. No selection. **19604/19608** isolate the failing PTX
  WGMMA instruction; even a minimal N128 kernel fails at80/88 and compiles
  at90/96. Do not retry N128 below90 via source-address rearrangement.
- Next two-group aliased source moves GP-store completion from immediately
  after GP publication to the following projection, before its retired input
  slot is overwritten. Final stores are fully completed before CTA exit.

- **19609-19612** delayed GP-store completion is bitwise/strict, graph-safe
  in the pilot and nonspilling, but full still +25-30 us. Rejected.
- **19613-19616** source splits4/6/12/16 with explicit FP32 folding into
  the existing eight partials all pass strict gradients and graph/eager
  checks. Full regresses63-450 us including folding. Rejected.
- Next source keeps selected N256 dW and two input slots, but lets the
  single compute warpgroup issue the next input after its completed dW/GP
  stores. This removes the idle loader warpgroup and its credit handshake.

- **19617-19620** one compute group with its own input prefetch is exact,
  strict, graph-correct and nonspilling, but full +11-35 us. Whole maps
  reduce the loss; neither variant is selected.
- **19621/19622** selected-source stores overlapped across the next
  projection retain exact GP/partials and strict graphs without spills,
  but full is neutral/slightly slower (0.85-2.48 us). No selection.
- Next D256 source keeps projection and GLU, replacing only dW with
  warp-local m16n8k16 BF16 MMA. It retains FP32 accumulation and K16 order;
  GP/partial/gradient correctness gates precede any performance conclusion.

- **19623/19624** warp-MMA pilot stops at compile because the PTX MMA
  accumulator type suffix was missing. **19625/19626** corrected kernels
  are rejected before launch: thread ID uses a register beyond the proposed
  initial pool. Volatile role-local thread-ID reads fix that lifetime.
- **19627/19628** warp-local dW then passes SASS audit, exact GP/partials,
  strict gradients/graphs and zero-spill two-CTA checks, but full is slower.
  A follow-up uses two N128 warp-local dW groups so each needs only64
  accumulator registers, aiming to retain two CTAs with projection overlap.

- **19629-19631** two N128 warp-local dW groups retain exact GP/partials
  and strict gradients but spill192-512B. **19633/19634** compiler levels
  do not eliminate spills. **19635** non-unrolled slot loop reduces local
  storage to32B at88 consumer registers and two CTAs. **19632**96-register
  non-unrolled control is spill-free but one CTA and full +499us.
- Next bounded pilot inserts warp barriers after2/4/8 N8 columns to limit
  speculative shared operand loads; includes honest full timing of the
  existing32B spill-bearing candidate and any strict candidate at<=32B.

- **19637-19641** warp operand barriers increase spills, all GP/partials
  remain exact. The32B spill control is strict/graph-correct but source
 546->985us/full2648->3054us. Reject warp-local dW. node02 is occupied by
  another user's eight-GPU job; these serialized normal_h100 pilots move
  to available node01, preserving all other jobs.
- Next pilot keeps WGMMA dW and instead uses warp-MMA projection. A48-reg
  producer plus two96-reg dW groups fits two CTAs if spills are avoided.

- **19642-19645** warp projection preserves GP/partial bits and all strict
  gradients; runtime K reduces48-register producer spills400->24B. Static
 64-register producer still spills176B. Next move TMA issuance before live
  accumulators, and test two producer warps plus two aligned WGMMA groups
  with uniform96 registers/320 threads. No dynamic allocation in that variant.

- **19646-19649** moving input issue and two-warp projection are bitwise/
  strict/graph-correct but regress full685-1119us. Warp-MMA projection is
  rejected. Next keep selected WGMMA source and reuse incoming-gradient
  storage as GP after reads. Gradient TMA overlaps projection; XN retains
  two prefetch slots. Freed8KB holds an exact shared sigmoid table, keeping
  two CTAs. Positive mantissa plus signed9-bit negative ULP delta covers
  BF16 magnitudes[2^-14,4); all65536 patterns are checked before launch.

- **19650-19653** shared8KB LUT passes all65536 BF16 patterns, exact GP/
  partials, strict graphs, two CTAs and no spills. Nevertheless direct LUT
  full+124us and compressed LUT+312-319us; no-LUT buffer control+100us. Reject.
- **19654** cluster smoke stops on a missing CUDA device index. **19655**
  confirms cluster sizes2/3/4 execute correctly. Next resident3-CTA contraction
  retains A128x384 and B384x128 in each CTA, exchanges B via two DSM slots,
  and computes all three N128 panels. This halves requested global input bytes
  versus three independent full-N384 row CTAs; actual DRAM traffic and full
  timing must validate whether that is useful.

- **19656** resident3-CTA contraction has39 active clusters,94 registers,
  no spills, bitwise dL/dR/GP/partials and strict eager/graph gradients. It
  nevertheless regresses contraction286->611us and full2642->3017us. Reject;
  no sanitizer qualification or checkpoint selection is claimed.
- **2026-09-27 user status request:** GPU experimentation is stopped because
  the user says no GPU is currently available. Live `squeue -u psk6950` is
  empty; there are no running or pending experiment jobs to cancel. All
  experiments through19656 are complete (19654 was a corrected host setup
  failure). Qualified authority remains D256 pool checkpoint, D38423,
  D51224. No new gain or engine dispatch change in this continuation.
  All L768 targets pass1.5x BWD and full; all three wide-D L384 targets
  still fail. See qualified_speedups.md for direct Triton comparisons.
