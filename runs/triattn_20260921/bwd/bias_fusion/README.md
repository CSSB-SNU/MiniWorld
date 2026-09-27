# Installed native CUDA backward bias fusion

> **Latest installed backward:** [../below80/README.md](../below80/README.md) records the qualified three-kernel update16580, additional 7.2–10.1% backward reduction, and projection/LN HBM SOL82.84–84.94%. Earlier results below are tied to their named artifacts.


> **Newer installed result:** [../SOL90_PROGRESS.md](../SOL90_PROGRESS.md) records the qualified grouped core, native dQ, and projection/LN/residual updates, including L384; the latest gate/context status is recorded there. Historical qualification and traffic data below remain tied to their named artifacts. Complete backward SOL90 is not reached.

2026-09-23. Selected artifact: **q32_four**, FP32 bias partials, R4, L768/1024.
The prior CUDA projection backward remains enabled in both comparison arms.
Forward SOL90 remains closed. This is an additional backward improvement.

## Whole-module result

Installed verification **job16112**, H100 80GB, B1 square BF16 inputs/weights, C128/H4/D32, no QK norm,
nonzero matrix weights, every seventh key masked, dropout0 for timing. Includes
all input and parameter gradients, residuals and required layout conversions.
Twelve balanced AB/BA rounds, fifteen CUDA-graph replays per measurement.
Times are separate medians in ms; reductions use the median **paired ratio**.
These are whole-module results, not whole-training-step results.

| L | Direction | Bwd before | Bwd after | Reduction | Fwd+bwd before | Fwd+bwd after | Reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 768 | starting | 7.850 | 7.834 | 0.32% | 10.420 | 10.382 | 0.34% |
| 768 | ending | 8.126 | 8.107 | 0.23% | 11.057 | 11.032 | 0.25% |
| 1024 | starting | 17.125 | 16.583 | 2.99% | 22.584 | 21.971 | 2.48% |
| 1024 | ending | 17.546 | 17.057 | 2.74% | 23.640 | 23.111 | 2.21% |

L384 regresses about3–4% and **retains the previous implementation**.
L768 has only a small installed speed benefit,0.23–0.32%; its bias-scratch
reduction is still50%. L1024 improves2.74–2.99% in backward and2.21–2.48% in
forward+backward. These are additional gains over the existing projection fusion.
Raw installed timings: `installed-paired-16112-L768.json` and
`installed-paired-16112-L1024.json`.

Earlier isolated qualification job16098 measured0.82–1.11% at L768 and2.15–2.43%
at L1024; do not substitute those for the installed numbers above. Its L384
regression is recorded in `paired-16098-L384.json`. Both stages use12 balanced
rounds, but they are separate GPU runs/wrappers; no exact cause is established
for the modest timing differences.

## Fusion, CTA ownership and memory

One CTA owns four outer rows, one head and one64-key tile. A dedicated producer
warpgroup issues TMA; four consumer warpgroups each own one row and retain their
final dK/dV accumulators. The640-thread CTA loops over32-query tiles. The producer
stages Q/dO and FP32 LSE/delta with two-stage TMA rings; resident K/V and the shared
bias tile are reused. Transaction barriers announce arrivals; empty-stage and
consumer barriers prevent overwrites before WGMMA **and** bias reduction finish.

Each query tile's four BF16 dS tiles are reduced directly from shared memory.
Only their FP32 group sum reaches the bias scratch. The final CUDA reduction
combines groups. No global elementwise atomic or dK/dV split partial is used.
Compiler: zero spills, no WGMMA serialization warning; shared memory109568 bytes.

The old per-row BF16 bias scratch is 2*H*L^3 bytes; the new FP32 group scratch is
4*H*L^3/4 bytes. This is a **50% allocation reduction**, not75%: partials are FP32.

| L | Previous bias scratch, decimal GB | New bias scratch, decimal GB |
|---:|---:|---:|
| 768 | 3.624 | 1.812 |
| 1024 | 8.590 | 4.295 |

Each dS contribution still rounds to BF16 before the FP32 sum; the group partial
has no additional BF16 rounding. Reduction order changes, so this is numerically
qualified, not bitwise-identical backward. Forward is unchanged. dQ and delta
preprocessing still use their existing Triton kernels; the new dK/dV and bias
reduction are native CUDA C++/TMA/WGMMA. This does not make every backward kernel CUDA.

## Qualification and dispatch

Job16098 qualifies the exact copied source/binary (`q32_four/build-ready.json`):

- Eight independent FP64 core cases, L64/128, no/mixed/one-key/all-masked inputs;
  deterministic results. Exact-zero one-key gradients use absolute comparison
  against the baseline's rounding residual, not division by zero.
- Sixteen full-module cases: both directions, four mask cases, dropout0/.25.
- Ten opaque-API checks: CUPTI confirms native execution; all input/parameter
  gradients, all-masked behavior, SGD changes to every parameter, frozen weights,
  and fullgraph Inductor forward/backward in both directions pass.
- Memcheck0 errors, racecheck0 hazards/errors/warnings, synccheck0 errors. Sanitizer
  cases exercise L64/128 and the pipeline phases; they are not full-L768/1024 runs.
- Full L768/1024 gradients are compared in paired graph measurements.

Installed code lives in the engine's `kernels/triangle_attention/cuda/`:
`bias_backward.py`, `bias_fusion.cu`, `triattn_bias_fusion.so`, `build_bias.py`,
`bias_manifest.json`. Module dispatch uses H100, B1/H4/D32, square L768/1024,
BF16 Q/K/V/bias in projection layout, grad enabled and no QK norm. Unsupported
cases retain the previous path. `model._fuse_bias_backward=False` disables only
this new fusion, preserving the earlier projection improvement.

Installed verification job16112 additionally passes13 dispatch/numerical records:
L768 starting/ending, mixed mask with dropout.25, all masked, all gradients,
SGD updates affecting every parameter, frozen weights, fullgraph Inductor, L384
fallback and unsupported metadata guards. CUPTI confirms native kernels run by
default at768 and disappear with opt-out/at384. Maximum gradient relative L2 vs
the prior path is0.115% before optimizer update and0.176% after update; compiled
vs eager maximum is about1.95e-6. Forward and all-masked outputs are unchanged.
Two AMP records confirm BF16-autocast compiled execution uses the native kernel
and FP16 metadata declines it (the pre-existing FP16 module limitation remains).
Evidence: `installed-correctness-16112.json`, `amp-correctness-16112.json`,
`installed-16112.log`. Full L1024 gradients pass both-direction graph comparisons.

The prior projection manifest (five entries) and forward manifest (twelve entries)
remain unchanged and pass hash verification. The new manifest checks provenance
of its five source/build/header/binary files; `installation.json` records the
selected artifact, qualification job and module hash.

## Experiment history

Warmed CUDA-graph pilot medians for the complete core, including delta, dQ and
final bias reduction. These are diagnostics, not adopted module improvements.
Completed valid pilots compare every core gradient against stock.

| Variant / job | L768 baseline core ms | R4 core ms | R8 core ms |
|---|---:|---:|---:|
| Q64, corrected reciprocal /16031 | 5.993 | 21.438 | 33.254 |
| Q64, coalesced partial /16034 | 5.991 | 21.140 | 33.681 |
| Q32 /16036 | 5.998 | 12.121 | 14.640 |
| Q32, explicit shared /16056 | 5.989 | 10.114 | 9.849 |
| Q64, explicit shared /16065 | 5.953 | 9.731 | 16.777 |
| Q64, row-first L2 partial /16068 | 5.952 | 12.886 | 14.411 |
| Q64, TMA statistics /16081 | 5.991 | 7.631 | 10.192 |
| Q32, TMA statistics /16086 | 5.989 | 7.779 | 7.890 |
| Q32, separate score/store phases /16091 | 5.992 | 6.376 | 6.711 |
| Q64, separate score/store phases /16092 | 5.951 | 5.973 | 10.888 |
| **Q32, four consumers /16097** | **5.951** | **5.896** | unsupported |

The selected four-consumer pilot at L1024 is13.822 ->13.222ms; job16098 supplies
the earlier isolated whole-module evidence. Installed job16112 is the final timing reference. Q64/four consumers has compiler
serialization and448-byte stack and was rejected before GPU launch.

A Q64/BF16-group-partial pilot (job16099) gives L7686.006 ->5.852ms and
L102413.970 ->13.169ms, but introduces about0.257% relative L2 bias-gradient
error versus stock. It has no full qualification and no compelling demonstrated
module advantage over the FP32 candidate; **not adopted**.

Earlier NCU job16064 measured the rejected Q32/shared variant at L768:
whole-core DRAM9.422GB baseline,6.327GB R4 and4.828GB R8. Its lower HBM traffic
still lost badly because the fused kernel stalled. Those counters do **not**
describe the final four-consumer kernel. Final NCU job16104 measures the selected four-consumer kernel at L768:
**9.4255 ->6.3435GB**, a **32.70% reduction** in whole-core DRAM reads+writes.
The new fused dK/dV kernel reads1.2701GB and writes2.1245GB; final bias reduction
reads1.8120GB. Delta and dQ are included in the totals. These are warmed replay
counters with uncontrolled clocks/caches and multi-pass profiler replay, not a
streaming roofline or an end-to-end speed measurement. `profile-final-summary.json`
records the units and kernels; raw CSV/report pairs are `profile-16104-r0/r4`.
NCU emitted an embedded-Python encoding diagnostic during CSV import, but both
reports contain all expected kernels and nonzero byte counters.

Installed verification job16103 passed both L768 directions, including compile,
then stopped at an overly strict bitwise assertion on unchanged L384 LN parameter
gradients. Diagnostic job16110 found only LN weight/bias variations, relative L2
about0.6–1.0e-6, also present between two stock calls. Output, input gradient and
all other parameters are bitwise identical. The corrected fallback check requires
bitwise equality outside LN and bounds LN error against repeated stock execution.
This was a test assertion correction; installed sources/binary were unchanged.

Implementation lessons:

- TMA descriptors must be grid-constant.
- Use reciprocal multiplication for bias scaling: precise division of the BF16
  mask sentinel emits a slow exceptional-value path. Preserve negative infinity.
- GMMA shared-layout pointer swizzles are byte-addressed. Manual BF16 indexing
  requires `as_position_independent_swizzle_layout`; `check_layout.cu` exhaustively
  matches all used coordinates against actual CuTe addressing.
- TMA-stage LSE/delta with Q/dO, and include their bytes in the transaction count.
- Give each consumer a row; reading all four shared dS tiles removes the separate
  FP32 shared bias accumulator and its update loop.
- Job16013 only completed old L64 code before a build failure; job16030 failed
  from a Python file shadowing `profile` (renamed `roof.py`). Jobs16040/16046 had
  invalid manual swizzling; job16041 profiling was canceled. None qualify a win.

## Reproduction

From the workspace root, use prebuilt artifacts for GPU qualification:

```bash
sbatch --export=ALL,FUSION_ARTIFACT=q32_four,FUSION_ROW_OPTIONS=4 \
  runs/triattn_20260921/bwd/bias_fusion/qualify.sbatch
sbatch runs/triattn_20260921/bwd/bias_fusion/installed.sbatch
sbatch --export=ALL,FUSION_ARTIFACT=q32_four,FUSION_PROFILE_ROWS='0 4' \
  runs/triattn_20260921/bwd/bias_fusion/profile.sbatch
```

Experimental imports never JIT-build; `native.py` verifies source/binary hashes.
The installed package can be rebuilt in the CUDA12.9/Torch2.10cu128 environment:

```bash
MAX_JOBS=2 bash runs/anthropic_adoption_20260919/env.sh python \
  runs/trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda/build_bias.py
```

CUTLASS4.2 comes from the existing sibling run or `CUTLASS_PATH`. The builder
refreshes `bias_manifest.json`; a rebuilt artifact needs its own validation.
