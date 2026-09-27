# Training forward producer and CTA follow-up

> Historical producer/cluster comparisons against17345; the later head-order change is in [README.md](README.md).

2026-09-25. No candidate is promoted. The native CUDA/TMA installation
`cooperative_q1_s2` (promotion17345) is retained; all 36 frozen package
files and the dispatcher match checkpoint17345. SOL90 remains unmet.

Experiments run on node02 / normal_h100 against the frozen current CUDA
kernel, not the earlier Triton path or CUDA17310. Core measurements use
12 alternating AB/BA rounds × 20 graph replays; complete module measurements
use 16 rounds × 20 replays, both directions, nonzero output weights, and
all input/parameter gradients. Percentages below are time increases from
`100 * (1 / median(paired baseline/candidate ratios) - 1)`.

| Candidate | L384 core | L768 core | L1024 core | Full FWD increase, six cells |
|---|---:|---:|---:|---:|
| `producer_warp_s2` | +238.17% | +238.70% | +207.90% | +65.48–+125.62% |
| `producer_wg_s2` | +8.39% | +5.48% | +4.07% | +1.78–+2.71% |
| `cooperative_s2_late` | +14.10% | +11.39% | +7.79% | +3.26–+6.08% |
| `cooperative_s2_retire` | +3.14% | +3.06% | +2.27% | +0.54–+1.29% |
| `producer_warp_relaxed` | +12.16% | +10.92% | +8.92% | +2.96–+4.64% |
| `cooperative_n32` | +30.22% | +32.98% | +31.32% | +7.95–+18.72% |
| `producer_warp_n32` | +28.96% | +32.15% | +31.47% | +7.63–+18.79% |
| `cooperative_pshared` | +12.01% | +13.22% | +11.64% | +3.24–+7.40% |
| `producer_warp_pshared` | +89.50% | +55.82% | +51.92% | +24.82–+31.35% |
| `cooperative_cluster2_v2` | +12.55% | +6.01% | +3.52% | +1.48–+3.55% |
| `cooperative_cluster4_v2` | +19.63% | +13.65% | +9.70% | +3.92–+6.32% |
| `cooperative_cluster2_early` | +22.12% | +18.01% | +13.22% | +5.25–+8.91% |

## Resource and lifetime findings

- The spill-free full producer warpgroup has initial64 registers/thread,
  producer32/consumer96 redistribution, and four CTAs/SM in profile17495.
  It removes WGMMA serialization warnings but loses to the six-CTA baseline.
- A compact producer warp with a tight register cap spills. Relaxing the
  cap eliminates spills at90 registers but still regresses measured speed.
- Eager PV retirement removes serialization warnings, but95 registers
  reduce CTA residency and the complete workload is slower.
- Smaller key tiles reduce fragment size but increase iterations and
  synchronization. They pass the sampled FP64 forward cases; changed online
  softmax grouping produces small BF16 differences, not bitwise equality.
- Shared P reuses dead bias storage and adds no HBM intermediate. Extra
  shared-memory operations and synchronization outweigh the register saving.
- Multicast recipients arm their local full barrier before leader TMA.
  The existing all-reader barrier precedes remote empty arrivals, so the
  leader cannot overwrite a bias slot before every CTA has finished it.
  Cluster entry/exit synchronization protects remote shared storage lifetime.
- Warp-level early bias release moves the handoff before softmax. It stays
  spill-free but grows to96 registers/thread and is slower than late release.

## Multicast traffic experiment

NCU17547 directly counts L768 traffic in the same allocation:
two-CTA multicast reduces L2 read sectors from228,449,858 to185,836,385
(18.65%) and total L2 tag sectors from236,021,731 to216,094,411 (8.44%).
HBM reads stay essentially unchanged at910.85 versus910.74MB. The bias
was already cached in L2; multicast does not remove a material HBM pass.
Profile17541 retains six CTAs/SM and80 registers but increases the barrier
stall-per-issue-active ratio from0.488 to0.820 versus profile17337.
The traffic reduction is real, but complete-workload measurements are slower.
The early-release variant has185–188 million L2 read sectors,96 registers,
and still slower measured execution. NCU export emits a Python site/encoding
diagnostic; all profile jobs exit0 and the report/CSV required metrics exist.

## Evidence and limits

All completed performance candidates pass six sampled independent FP64
forward fixtures at each production length and changed-input graph replay.
Complete-module benchmarks check gradients and actual native dispatch.
Performance rejection does not imply full sanitizer or strict independent
FP64 gradient qualification; these experimental candidates are not approved
for dispatch/autograd. Failed builds are not performance measurements.

The lower-register producer and K-stash controls fail PTXAS C7602 at the
first QK WGMMA (jobs17493/17494; retained PTX probe17506). Dependencies17502/
17503 were cancelled. The first cluster builds17525/17526 had a C++ overload
error, fixed in separate v2 artifacts before benchmarking.

See [pipeline-experiments.json](pipeline-experiments.json) for artifact hashes,
all paired core/full-workload rounds, compiler diagnostics and build failures.
The unchanged installed result remains in [README.md](README.md).

Harness changes: `check.py` accepts an explicit frozen baseline; native builds
can retain PTX with `KEEP_PTX=1`; `bench_17345.sbatch` fixes the baseline;
`bench_cluster.sbatch` checks two-stage wraparound before production shapes.
`promote_incremental.py` accepts explicit baseline/launch metadata but was
not used for a promotion in this follow-up.
