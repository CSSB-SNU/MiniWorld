# QKV projection and attention fusion

2026-09-26. Seven native CUDA candidates were implemented and measured on
node02 / normal_h100. **Q-only projection fusion is the selected design.**
The full-QKV resident designs lose on L768/1024 despite removing many reads.
Q-only fusion preserves the 64-query streaming attention ownership and gives
a repeatable full-forward improvement. SOL90 remains unmet.

**Installed and verified: promotion17774**, `q_only_head4`. The full17628
baseline remains available through the Q-fusion opt-out. All pre-install and
actual installed selection gates passed.

## Actual installed full-module result

64 alternating AB/BA rounds x40 graph replays per arm, same inputs/process.
Times are arm medians; reductions use paired ratios. All outputs/gradients
are bitwise equal. Full intervals and raw timings are in `qkv-fusion-results.json`.

| L | Direction | Full FWD ms | Reduction | F+B ms | Reduction |
|---|---|---:|---:|---:|---:|
| 384 | starting | 0.4068 -> 0.3863 | 4.94% | 1.2615 -> 1.2484 | 1.29% |
| 384 | ending | 0.5005 -> 0.4821 | 3.67% | 1.4075 -> 1.3910 | 1.13% |
| 768 | starting | 1.9930 -> 1.9586 | 2.39% | 6.8273 -> 6.7983 | 0.43% |
| 768 | ending | 2.3626 -> 2.3179 | 2.02% | 7.3826 -> 7.3471 | 0.58% |
| 1024 | starting | 4.1443 -> 4.0717 | 1.74% | 14.5933 -> 14.5231 | 0.49% |
| 1024 | ending | 4.7796 -> 4.7107 | 1.40% | 15.5280 -> 15.4720 | 0.35% |


## Implemented ownership

`q_only_head4` computes Q with native TMA/WGMMA in the attention CTA. It writes
the BF16 Q training save, then reuses that same shared-memory tile for attention.
K/V retain their two existing GEMMs and are streamed with two TMA stages.
Projection Z/weight scratch aliases the later K/V/bias buffers after the Q
projection finishes. Four consecutive head CTAs share normalized input cache
lines. It uses 128 threads, 90 registers, zero spills, 37,888 B shared storage
and five CTAs/SM. The attempted six-CTA register cap spilled and was slower.

The producer phase and attention phase are sequential inside a cooperative
warpgroup. This is not a claim of dedicated producer/consumer overlap.
The same conservative WGMMA completion, shared fences and 128-thread barriers
are retained. No global score/probability tensor is introduced.

The new front+attention autograd boundary eliminates the original Q GEMM;
it does not compute Q twice. It preserves Q/K/V saves and reuses the existing
gate/delta, dQ, dK/dV+bias, projection/LN/residual and joint weight-gradient
kernels. LN, gate and global bias preparation stay outside this CUDA fusion.
Output-weight gradient execution and temporary lifetimes match the old boundary.

## All candidate comparisons

These are paired speedup ratios for **all three projections plus attention**,
including QKV training saves, against frozen17628's three projections and core.
Each cell has its own same-process baseline. Values below1 mean slower.
Core tests use 12 alternating AB/BA rounds x20 CUDA-graph replays per arm.

| CUDA candidate | L384 | L768 | L1024 | Decision |
|---|---:|---:|---:|---|
| Full QKV resident, one projection WG | 0.816x | 0.618x | 0.695x | Reject |
| Full QKV resident, shared Z across Q/K/V | 0.870x | 0.663x | 0.751x | Reject |
| KV resident, four parallel projection WGs | 1.022x | 0.878x | 0.932x | Reject |
| KV resident, six parallel projection WGs | 1.027x | 0.935x | 0.963x | Reject; spills |
| Q-only, two adjacent heads | 1.044x | 1.044x | 1.023x | Qualified control |
| Q-only, six-CTA register cap | 1.025x | 1.020x | 0.987x | Reject; spills |
| Q-only, four adjacent heads | 1.056x | 1.047x | 1.025x | Selected for full-module gates |

Every candidate matched all three BF16 projections, attention output and LSE
bitwise at L64/256/384/768/1024 on six fixtures: mixed, dense, one key,
all masked, late live keys and large logits. This correctness result does not
qualify the rejected candidates for production: they were not all run through
the full module and sanitizer suite.

## Why reducing reads was insufficient

NCU17703 profiles the complete three-projection-plus-attention region at L768.
The full-resident reuse candidate reduces HBM reads from910.93 to245.16MB, but
its profiled total is1.911ms versus1.270ms for the baseline kernels combined.
Its SM throughput is40.58% versus74.18% for the baseline attention kernel.
One CTA/SM holds 214,016B, and three consumer WGs wait for a single projection WG.

The diagnostic-only clock64 build17702 puts the median per-CTA projection share
at47.9% /41.8% /34.4% for L384/768/1024. These are phase shares of the instrumented
CTA, not an exact decomposition of the uninstrumented GPU time. Parallelizing
the projection and retaining only K/V reduces that cost substantially, but still
does not beat the baseline at the two large lengths.

For the selected Q-only head4 design, NCU17767 measures total HBM reads across
projection+attention at910.94 ->759.91MB (-16.58%). Writes are555.06 ->570.54MB;
total HBM traffic falls about9.25%. L2 read sectors rise259.00 ->274.58million.
Therefore it is inaccurate to claim that every memory level improves. The fused
attention kernel has SM70.51%, L288.05%; this is not SOL90. NCU uses uncontrolled
clocks/caches, and its replay timings are explanatory rather than selection data.
The ncu CSV exporter prints a Python site-encoding warning; all expected kernel
rows and reports were written, and the profiling jobs exited0.

## Correctness and evidence

- `qkv_check.py` checks independent FP64 projections and sampled attention,
  changed-input CUDA-graph replay, output strides and all six numerical fixtures.
- Staged qualification17762: eight independent FP64 projection/attention gradient
  fixtures and18 complete module cases covering both directions, masks, dropout,
  optimizer updates, fullgraph and frozen parameters.
- Staged extra verification17765: cold fullgraph, custom-op schema/fake/AOT,
  BF16 AMP, no mask, partial weights, weights-only gradients and dispatch guards.
- Sanitizer17766: memcheck/racecheck/synccheck, L128 andL768, mixed/all-masked.
  Zero errors and zero hazards in every run.
- Initial experimental fullgraph17724 failed because a CustomOpDef was captured
  in a closure. Moving the op registration to module scope fixed it (17732,
  final adapter17741). The failure is retained in the logs.
- Build17712 was cancelled after discovering a barrier-array reference error,
  before publication. The corrected parallel candidates use separate artifacts.
- Full-module confirmation17756 uses64 alternating AB/BA rounds x40 replays,
  all inputs/parameters nontrivial, both directions and all three production sizes.
- `qkv-fusion-results.json` collects source/binary hashes, candidate comparisons,
  raw full-module rounds, paired bootstrap intervals, phase diagnostics and NCU
  counters. Intervals are within-run uncertainty, not cross-node guarantees.

Build/bench pairs:17677/17686,17687/17701,17708/17714,17721/17728,
17722/17729,17727/17734,17743/17745. Full-module controls:17693,17704,
17723,17737. The design and traffic equations are retained in
[QKV_ATTENTION_FUSION.md](QKV_ATTENTION_FUSION.md).

## Next bounded experiment

The selected CTA already loads the complete128-channel Z tile for Q. A useful
next candidate is **Q + gate projection + attention**, keeping K/V streaming.
Generate the gate head tile from that same Z, save it for the existing gate
backward, then reuse its shared output slot for Q. The Z tile is16KiB and one
head weight tile is8KiB, which can reuse the existing32KiB projection/attention
scratch; gate and Q weights can be loaded sequentially. This could remove the
standalone gate GEMM and its Z read without retaining an entire QKV row.
It still requires the gate training save, an additional projection/TMA-store
phase, a new output/autograd interface and complete F+B validation. This idea
has **not** been implemented or benchmarked; no extra gain is claimed.
