# Output fusion and single-head pipeline experiments

Historical checkpoint18962: existing per-length cores plus native `outproj_c1`.
The later [query-reuse follow-up](CORE_FOLLOWUP_RESULTS.md) replaces only the
L1024 core and retains this output boundary. Production dispatch and training
remain unchanged. The four-head attention CTA direction stays closed.

## Complete inference FWD

Node02, normal_h100; actual `candidate.load()` entry, job18962;64 AB/BA rounds x40 CUDA Graph replays; B1/C128/H4/D32 BF16, both directions. Includes LN, bias, all projections, attention, gate and input-preserving residual. Compare with the faster of two original Anthropic residual forms. Times are individual medians; reductions and confidence intervals use paired ratios.

| L | Direction | Anthropic ms | Ours ms | Reduction vs Anthropic | Reduction vs previous ours |
|---:|---|---:|---:|---:|---:|
| 384 | starting | 0.4609 | 0.3125 | 32.12% | 8.85% |
| 384 | ending | 0.4674 | 0.3160 | 32.45% | 7.09% |
| 768 | starting | 1.7535 | 1.5991 | 8.63% | 5.82% |
| 768 | ending | 1.7157 | 1.6042 | 6.47% | 5.83% |
| 1024 | starting | 3.5232 | 3.3439 | 5.03% | 4.83% |
| 1024 | ending | 3.4635 | 3.3590 | 2.98% | 4.32% |

## Selected output boundary

One128-thread CTA computes64x128 output values with WGMMA. TMA loads attention output, output weight and residual; the residual load can overlap GEMM. The epilogue preserves `BF16(X + BF16(A @ Wo.T))`, reuses completed input scratch for output, and TMA stores directly in the required starting/ending orientation. X stays unchanged. Register count90, no spills, dynamic shared memory66560B. No dedicated producer warpgroup is justified for this short boundary; one elected thread issues the asynchronous transfers.

The projected-output temporary is eliminated. The idealized traffic model is5S to3S, where S=2*L*L*128. Actual warmed NCU totals include caching and GEMM tiling:

| L | Direction | Previous HBM MB | Fused HBM MB | Reduction |
|---:|---|---:|---:|---:|
| 384 | starting | 167.4 | 114.4 | 31.7% |
| 384 | ending | 168.1 | 111.9 | 33.4% |
| 768 | starting | 734.3 | 454.7 | 38.1% |
| 768 | ending | 734.4 | 454.7 | 38.1% |
| 1024 | starting | 1321.5 | 806.7 | 39.0% |
| 1024 | ending | 1321.5 | 808.5 | 38.8% |

NCU18953 used five warmups, cache-control none and clock-control none; profiling is separate from paired timings. The known NCU Python-site export warning occurred; all six reports, CSVs, numerical checks and the Slurm job completed.

The2/4-warpgroup output variants pass the same numerical pilots but do not beat the1-group boundary, so the simpler1-group candidate is selected. Only this selected output gets full qualification.

## Core controls

All controls retain single-head attention CTAs and resident KV. `serial32_c4` isolates the smaller key tile. `pipe32` uses two alternating QK accumulators. `pipecopy` keeps QK accumulators separate from scalar softmax work. `pipeclean` additionally drains before reading QK results, then issues the next QK before current softmax. These are source-level schedules; compiler serialization diagnostics are recorded in the JSON and prohibit claiming effective overlap.

| Candidate | L384 ms start/end | L768 ms start/end | L1024 ms start/end | Compiler warnings |
|---|---:|---:|---:|---:|
| pipe32_c4 | 0.443 / 0.439 | 1.991 / 1.994 | 4.144 / 4.153 | 5 |
| serial32_c4 | 0.431 / 0.424 | 1.958 / 1.964 | 4.125 / 4.140 | 0 |
| pipe32_c2 | 0.371 / 0.363 | 2.895 / 2.899 | 6.158 / 6.200 | 5 |
| pipecopy64_c2 | 0.501 / 0.494 | 2.612 / 2.611 | 5.487 / 5.502 | 5 |
| pipecopy32_c4 | 0.466 / 0.461 | 2.183 / 2.184 | 4.581 / 4.597 | 5 |
| pipeclean64_c2 | 0.492 / 0.484 | 2.585 / 2.585 | 5.397 / 5.409 | 5 |
| pipeclean32_c4 | 0.464 / 0.459 | 2.163 / 2.174 | 4.555 / 4.579 | 5 |

All seven core controls are slower than the previous selected core paths. They are rejected; the three existing per-length cores remain selected. No core speedup or successful hardware overlap is claimed. The smaller-tile serial control also loses, so serialization alone does not explain the regression.

Core control timings use the historical cuBLAS tail, so compare them with their same-run previous-selection rows in JSON, not the new fused-output times. Native FP64 fixtures and retry/changed graphs pass; these pilots do not constitute full sanitizer qualification.

## Qualification and scope

Selected output: all60 output fixtures at L64/128/384/768/1024 in both directions are bitwise equal to cuBLAS+post, including zeros, large values and cancellation. Independent FP64 samples, out-of-place/input-preservation checks and changed-input/weight graph replay pass. All three sanitizers pass at128/384/768/1024. Full module qualification and actual-default qualification cover mask/affine fixtures, FP64 samples, changed-input/weight/mask graphs, no_grad and fullgraph at384/768/1024. All48 training18246 files still match their snapshot.

At checkpoint18962, `load()` selected this experiment. The current `load()` follows
the newer per-length selection; `load(out_artifact=None)` disables output fusion
while retaining those current cores. Reproduce18962 with explicit
`load("h4kv_local1"/"hot6t"/"hot4t", out_artifact="outproj_c1")` for384/768/1024.
Explicit core names retain their historical unfused tail unless
`out_artifact="outproj_c1"` is given.

All-masked semantics remain ours zero update versus Anthropic mean-V; this is not an unrestricted drop-in replacement. Inference only, no backward saves or gradient qualification. SOL90 is not achieved.

Initial output builds18932/18934 failed on CuTe namespace collisions and host binding ambiguity; fixed before18937. No failed build is performance evidence. Evidence: [JSON](output-pipeline-results.json), [design](PIPELINE_OUTPUT_DESIGN.md).
