# Installed training attention forward

2026-09-25. **Native CUDA/TMA `cooperative_q1_s2` is installed and verified**,
promotion 17345. This is an additional improvement over the already installed
CUDA `cooperative_q2` from 17310. H100, B1/H4/D32 BF16 projection layout,
L384/768/1024, both directions. All experiments used node02 / normal_h100.
**SOL90 remains unmet.**

## Actual installed full workload

Frozen CUDA 17310 versus the newly installed path, same process and inputs,
32 alternating AB/BA rounds × 40 CUDA graph replays per arm. Nonzero parameters,
all input/parameter gradients. Times are arm medians; reductions use paired
median ratios. Every recorded full-workload comparison has zero output/gradient
error against CUDA 17310. This table is not a comparison with the old Triton path.

| L | Direction | Full FWD ms | Reduction | F+B ms | Reduction |
|---|---|---:|---:|---:|---:|
| 384 | starting | 0.4243 → 0.4093 | 3.55% | 1.2736 → 1.2577 | 1.32% |
| 384 | ending | 0.5199 → 0.5050 | 2.87% | 1.4224 → 1.4062 | 1.11% |
| 768 | starting | 2.0979 → 2.0110 | 3.87% | 6.9486 → 6.8708 | 1.16% |
| 768 | ending | 2.4768 → 2.3730 | 3.86% | 7.4942 → 7.4188 | 0.83% |
| 1024 | starting | 4.3708 → 4.1751 | 4.28% | 14.8218 → 14.6385 | 1.20% |
| 1024 | ending | 4.9959 → 4.8087 | 3.59% | 15.7912 → 15.6220 | 1.04% |

Full backward is unchanged. Raw rounds and paired bootstrap median intervals
are in `installed-results.json`; intervals describe sampling uncertainty within
these runs, not variation across nodes or workloads. The original 17310-versus-
Triton results remain in [INITIAL_17310.md](INITIAL_17310.md).

## Selected change

One 128-thread CTA computes a 64×64 attention tile stream. A single FP32 score
fragment replaces the two-fragment pipeline. K/V/bias TMA buffering falls from
three stages to two. QK completion retires the previous PV, then all four warps
synchronize before thread zero reuses the prior stage for the next TMA loads.
The cooperating warpgroup consumes those tiles. Resident Q is reused for the
final TMA output store. No preparation kernel or score/probability HBM buffer.

Registers fall from 116 to 80 per thread, kernel shared storage from 54,272 to
37,888 bytes, with zero spills. Six CTAs fit per SM instead of four. Stable
online softmax, unrounded FP32 probability sums, BF16 PV operands/output and
FP32 base-2 LSE retain the original arithmetic. Production O/LSE match CUDA 17310
bitwise in all six core fixtures at each production length.

The selected build still has PTXAS C7515 serialization warnings. Removing a
warning alone did not improve speed: the three-stage single-score control
removed C7514 but was slightly slower. This selection is based on measured
complete-workload gains, not a claim of fully overlapping WGMMA operations.

## Core and profile

Core-only job 17344, randomized inputs and mixed mask,
12 AB/BA rounds × 20 replays, separate from the complete-module timings:

| L | CUDA 17310 → selected ms | Paired reduction |
|---|---:|---:|
| 384 | 0.1529 → 0.1363 | 10.88% |
| 768 | 1.1741 → 1.0420 | 11.22% |
| 1024 | 2.7058 → 2.4591 | 8.45% |

NCU source profiles 17321 → 17337, L768:
1.043360 → 0.977600 ms. The selected kernel reaches
SM 73.77%,
L2 77.58%,
HBM 32.95%,
tensor pipe 24.97%,
and active-warp occupancy 36.85%.
These single-kernel profiles are not paired end-to-end benchmarks or SOL90.

## Validation and rejected experiments

- Qualification 17336: eight independent FP64 core gradient fixtures and 18
  complete-module checks covering both directions, masks, dropout, optimizer
  updates, fullgraph and frozen parameters. All pass.
- Memcheck/racecheck/synccheck pass at L64/256, all six fixtures; full-shape
  L768 mixed and fully masked cases pass in 17338. No errors or hazards.
- Installed 17345: fresh-process cold fullgraph F+B, schema/fake/AOT opcheck,
  actual six module dispatches, layout/dtype/opt-out guards, supported BF16 AMP,
  and the installed paired benchmarks above pass.
- The six backward manifests and every dependency remain unchanged from 17211.
  Dispatcher and Python guard logic also remain unchanged from 17310.

| Candidate | Result |
|---|---|
| `cooperative_earlyqk`, 17325 | Earlier QK launch retained C7514; all production shapes slower. |
| `cooperative_q1`, 17328 | No serialization warning; 96 registers, four CTAs; slightly slower. |
| `cooperative_unroll`, 17329 | Removes warning but spills; L768 core 1.108 → 1.402 ms. |
| `cooperative_q1_s2`, 17333/17344 | Selected; six CTAs, all complete workload cases improve. |
| `cooperative_q1_m128`, 17334 | Two query warpgroups share K/V; no consistent complete-workload gain. |
| `cooperative_bounds`, 17340 | Range-certified softmax passes O/LSE fixtures; including preparation, slower. Not fully qualified. |
| `cooperative_s2_fence`, 17341 | Extra accumulator fence compiles to identical 1,952 SASS instruction lines; no distinct improvement. |
| `cooperative_m128_s2`, 17343 | K/V sharing plus two stages improves over 17310, less than selected in measured overall results. Not promoted. |

## Reproduction and rollback evidence

- Source: [cooperative_q1_s2/fused.cu](cooperative_q1_s2/fused.cu).
- Promotion: [promotion-17345.json](promotion-17345.json).
- Complete frozen native package: [checkpoint17345/snapshot.json](checkpoint17345/snapshot.json).
- Original CUDA baseline remains in `cooperative_q2` and `checkpoint17310`.
- `module_check.py --baseline-artifact cooperative_q2 --installed` compares
  a new installed path with the frozen CUDA baseline; plain `--installed`
  still compares with the old Triton opt-out path. No implicit baseline mixing.
- `MINIWORLD_TRIATTN_TRAINING_FWD=0` retains the established opt-out behavior.
