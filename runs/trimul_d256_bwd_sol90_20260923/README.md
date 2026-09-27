# MiniWorld D256 backward — SOL90 target not achieved

2026-09-23, H100 node01, B=1, BF16, bidirectional H=2D, dropout25%, mask,
residual and all eleven gradients. The selected native B7 reduces backward
time by 12.0–12.2%. **SOL90 remains unachieved.** No automatic engine dispatch
or autograd registration was changed.

## Selected implementation

The D128 reference was `trimul_ln_gradient_20260922/fixed/joint.cu`:
generate gate/projection derivatives, accumulate dW locally, then expose the
derivatives to the dX consumer. Its multiplication order is retained exactly:
`dg = bf16(((da * projection) * sigmoid) * (1 - sigmoid))`.

`source.cu` ports producer-local dW to D256 with 32-channel gate/projection
pairs, resident weights, double-buffered TMA inputs and shared-operand WGMMA.
Each CTA immediately accumulates both dW halves in FP32. Thus the separate
weight-gradient GEMMs no longer reread the complete derivative planes or x_n.
`finish.cu` computes dX_n, input LN backward, and reduces partial dW. Both are
native CUDA; existing four cuBLAS contraction-gradient calls are unchanged.

The full derivative planes remain for dX. Their D256 size is 576 MiB at L384
and 2304 MiB at L768. This implementation does **not** claim bounded workspace.
L384 uses 8 dW splits; L768 uses 16. Eight splits at L768 pass the loose PyTorch
comparison but fail the stricter 5e-4 previous-schedule dW bound (max 5.884e-4),
so that configuration is not selected.

## Same-process comparisons

150 alternating CUDA graph samples. Both sides use our new wide forward and
the previously selected B1 preparation optimization. The only selected change
in these comparisons is B7. These are not Anthropic comparisons. Allocation,
compilation, CPU dispatch, RNG generation, and optimizer are excluded.

| L | B7 old → new ms | Backward old → new ms | Fwd+bwd old → new ms |
|---:|---:|---:|---:|
| 384 | 2.041 → 1.549 | 4.173 → 3.671 | 5.109 → 4.610 |
| 768 | 8.183 → 6.111 | 16.989 → 14.911 | 20.643 → 18.589 |

B7 improves 24.1–25.3%; backward 12.0–12.2%; fwd+bwd 9.8–9.9%.
Raw data: `result-L384-s8.json` (job16710), `result-L768-s16.json` (job16734).
`summary.json` records derived values and explicitly sets `sol90_verified=false`.

## Accuracy and memory checks

All eleven gradients pass relative-L2 < 0.01 against the existing compiled
PyTorch BF16 fixture. The selected schedule also passes the D128-style stricter
comparison to the previous B7: dX < 2e-5, LN gradients < 5e-6, weight gradients
< 5e-4. In the standard case dX is bit-identical. This second comparison holds
forward/B1 numerics fixed; it is not a claim of 5e-6 full PyTorch agreement.

`stress.py` checks normal input, changed x/weights/dy/mask/dropout, partially
zero LN gamma, all-zero mask, all-zero dropout, plus replay of a graph captured
before mutations. Both lengths pass (10 cases total). Maximum dW difference
is 3.639e-4. Graph/eager LN reductions have small atomic-order variation within
5e-6; graph bit-identity is not claimed.

Job16742: source and finish memcheck at L384/L768 report zero errors; racecheck
at both lengths reports zero hazards/errors/warnings. See `stress-L*.json`,
`memcheck-L*.log`, `racecheck-L*.log`. Job16750 verifies the selected wrapper's
live shared weight pack and changed-weight/dy graph replay at both lengths.

## SOL definition and measured bottlenecks

Use the repository's nominal dense BF16 peak of 989.5 TFLOP/s and bandwidth
3.35 TB/s, not a sparse peak. For M=L², the recomputing backward's GEMMs require
`66*M*D² + 8*D*L³` FLOPs: B1 16*M*D², B7 50*M*D², and four contractions.
An optimistic unique-input/output payload is
`20*M*D + 44*D² + 48*D + 2*M + 2*L*D` bytes. It excludes internal activation
round trips, instruction/shared-memory costs, barriers, and dependency delays.
It assumes recomputation is retained; it is not a proven attainable runtime or
a bound for every possible saving/recomputation policy.

`lower_bound = max(FLOPs / 989.5e12, payload / 3.35e12)`.
The FLOP bound dominates both lengths. Additional measured DRAM rereads are
not counted as necessary payload to inflate the SOL percentage.

| L | Optimistic lower bound ms | Selected backward ms | Model efficiency | Runtime needed for 90% ms |
|---:|---:|---:|---:|---:|
| 384 | 0.762 | 3.671 | 20.7% | 0.846 |
| 768 | 3.516 | 14.911 | 23.6% | 3.906 |

Fresh NCU job16744 (`profile-L384.ncu-rep`, CSV, `profile-summary.json`) is
separate from the event benchmarks:

| L384 stage | NCU time ms | DRAM read+write GB | Tensor active % of sustained-active peak |
|---|---:|---:|---:|
| B1 prepare | 0.787 | 0.654 | 7.54 |
| B1 finish | 1.024 | 1.461 | 9.47 |
| B7 source/local dW | 0.821 | 1.010 | 40.66 |
| B7 dX/LN/reduce | 0.716 | 1.077 | 24.75 |

Four contraction GEMMs total approximately 0.311 ms. Tensor active percentage
is not the same quantity as whole-backward roofline efficiency. B1 remains the
largest stage; B7 still writes and rereads its global derivative planes.

## Rejected or diagnostic candidates

- B1 local dWproj+dWgate with no global normalized triangle: correct under the
  original loose bound, but B1 1.806 → 2.612 ms, 356 B spill stores / 436 B spill
  loads, plus repeated LN reads. Not selected.
- N128 finish tiles: backward 4.170 → 3.789 ms including new B7, slower than the
  selected 3.671 ms schedule. Not selected.
- D128-style bounded producer/consumer ring: 12 MiB ring and strict-correct dX/
  gradients, but B7 2.038 → 2.629 ms. 960 B spill stores / 1072 B spill loads;
  eliminating the full workspace alone did not give a speedup. Not selected.
- Restricting loop unrolling to reduce register pressure: slower for both the
  two-stage and ring variants (B7 1.672 / 2.788 ms). Not selected.
- cuBLAS diagnostic for remaining B1 GEMMs: B1 1.819 → 1.727 ms; valid standard
  gradients but only a small improvement. This is an oracle experiment, not the
  selected native path or a claim that our B1 reached hardware limits.
- Four B7 dW splits lose nearly all speedup. Sixteen is retained for L768
  precision; eight is faster for L384. Raw candidate JSON/logs are preserved.

The large remaining work is a B1 schedule with bounded live accumulators and
less normalized-triangle/derivative traffic, and a B7 producer/consumer balance
that avoids spilling. Porting D128's dimensions mechanically did not achieve it.

## Explicit entry and reproduction

`selected.Training(leaves, mask, dropscale, dy)` combines new forward, current
B1, and selected B7. Call it under `torch.no_grad()` to receive `(y, gradients)`.
The leaves/gradient order is x, WL, WLg, WR, WRg, Wgate, Wproj, gamma_in,
beta_in, gamma_out, beta_out. Packed weights are shared with forward and updated
there. Fixed dy storage may be updated in place; replacing bindings requires
a new plan. Mask conversion is owned at construction. Outputs and saves reuse
buffers; use separate plans for outstanding forwards. This is a development
entry, not automatic PyTorch autograd. Its allocator still uses the existing
width plan's workspaces.

Run from the environment used by `runs/anthropic_adoption_20260919/env.sh`,
with this directory on `sys.path`. Submit `check.sbatch` for L384; use
`--export=ALL,LENGTH=768,SPLITS=16` for L768. `verify.sbatch` repeats stress and
sanitizers; `profile.sbatch` collects NCU. Leave all experimental flags unset.
`source-manifest.json` records source hashes.

Failed tooling attempts are preserved: job16714 had an NCU Python-path
conflict; 16719/16720 were cancelled after discovering a local module name
collision. The collector is now named `collect_ncu.py`; accepted profiling
uses completed job16744, not those attempts.
