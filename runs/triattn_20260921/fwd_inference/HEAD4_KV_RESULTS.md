# K/V placement and four-head reuse

Qualified experimental default: `candidate.load()` selects L384 `h4kv_local1`, L768 `hot6t`, and L1024 `hot4t`. Actual selected entry verified by job18745. Production dispatch and all48 training18246 files remain unchanged. No SOL90 claim.

## Complete inference FWD versus original Anthropic

Node02 / normal_h100, B1/C128/H4/D32 BF16, both orientations, 64 AB/BA rounds x40 graph replays. Includes LN, bias, every projection, attention, gating and input-preserving residual. Use the faster original residual form per cell. Percentages are paired ratios, times are separate medians.

| L | Direction | Anthropic ms | Selected ms | Time reduction | Speedup 95% CI |
|---:|---|---:|---:|---:|---|
| 384 | starting | 0.4616 | 0.3446 | +25.97% | 1.3498–1.3517 |
| 384 | ending | 0.4617 | 0.3405 | +26.88% | 1.3663–1.3695 |
| 768 | starting | 1.7417 | 1.6991 | +2.51% | 1.0228–1.0281 |
| 768 | ending | 1.7196 | 1.6970 | +1.23% | 1.0083–1.0173 |
| 1024 | starting | 3.4880 | 3.5014 | -0.56% | 0.9923–1.0000 |
| 1024 | ending | 3.4410 | 3.5214 | -2.35% | 0.9759–0.9798 |

## Increment over previously qualified hot6t

| L | Direction | hot6t ms | Selected ms | Time reduction |
|---:|---|---:|---:|---:|
| 384 | starting | 0.3725 | 0.3424 | +8.10% |
| 384 | ending | 0.3694 | 0.3410 | +8.06% |
| 1024 | starting | 3.5511 | 3.4917 | +1.63% |
| 1024 | ending | 3.5579 | 3.5100 | +1.35% |

L768 retains hot6t; no improvement is claimed there.

## What four heads buy

Four D32 heads form one dense N128 projection. The wide kernel loads Z once for both K and V and all four heads. The strong narrow control also puts its four head CTAs next to each other, so this comparison excludes the avoidable slow head-major grid. Both controls have the same row-local attention kernel, bitwise-equal projections/full outputs, and changed-input projection graph checks. Paired 48x40 measurements:

| L | Adjacent single-head KV us | Joint head4 KV us | Projection time reduction |
|---:|---:|---:|---:|
| 384 | 45.84 | 44.15 | 3.71% |
| 768 | 177.30 | 164.65 | 7.12% |
| 1024 | 313.29 | 291.04 | 7.29% |

The first head-major control was2.15–2.34x slower; much of that difference was input-cache locality, not the projection width alone. Head-specific K/V values and softmax remain independent.

## K/V lifetime and CTA scheduling

L384 materializes KV once and streams two small K/V+bias tiles per warpgroup. Query groups and the four heads for one outer row are adjacent in grid X. Q/gate remain fused with attention. L768/1024 retain fully on-chip QKV; L1024 uses four cooperative warpgroups and two bias stages, 126 registers and no spills, removing the old6+6+4 query tail. Its lower warp count loses at768, so768 retains six warpgroups. There is no new dedicated producer warpgroup in these selected kernels.

At768 NCU reports streaming attention HBM traffic3.932→0.605GB when only the CTA grid order changes; core time1.414→1.109ms. KV projection adds0.160ms and0.431GB, so combined projection/core still loses to hot6t1.239ms/0.406GB. These are warm-cache NCU diagnostics (17 passes/kernel), not the complete-FWD selection timings. The export emitted the known Python-site warning, but reports, metric rows and exit status completed successfully.

## Controls

Pilot complete-FWD times, starting / ending ms. Unselected candidates are not promoted or represented as fully qualified.

| Candidate | L384 | L768 | L1024 |
|---|---:|---:|---:|
| h1adj_local1 | 0.347 / 0.345 | 1.763 / 1.782 | 3.719 / 3.743 |
| h1kv_stream2 | 0.415 / 0.413 | 2.077 / 2.090 | 4.367 / 4.372 |
| h4kv_front1 | 0.375 / 0.373 | 1.813 / 1.841 | 3.825 / 3.828 |
| h4kv_front1v | 0.358 / 0.358 | 1.797 / 1.796 | 3.774 / 3.758 |
| h4kv_local1 | 0.343 / 0.340 | 1.737 / 1.748 | 3.679 / 3.704 |
| h4kv_local2 | 0.353 / 0.349 | 1.799 / 1.808 | 3.765 / 3.757 |
| h4kv_local4 | 0.407 / 0.407 | 1.878 / 1.884 | 3.905 / 3.919 |
| h4kv_resident6 | 0.404 / 0.403 | 1.863 / 1.869 | 3.844 / 3.838 |
| h4kv_single1 | 0.413 / 0.411 | 2.316 / 2.323 | 4.989 / 4.932 |
| h4kv_stream1 | 0.384 / 0.383 | 2.069 / 2.071 | 4.454 / 4.451 |
| h4kv_stream2 | 0.362 / 0.360 | 1.874 / 1.903 | 4.051 / 4.126 |
| h4kv_stream4 | 0.417 / 0.416 | 1.908 / 1.911 | 4.002 / 3.993 |
| hot4t | 0.368 / 0.366 | 1.723 / 1.726 | 3.520 / 3.534 |
| hot4t_s1 | 0.369 / 0.366 | 1.753 / 1.754 | 3.582 / 3.578 |

The LN/bias/KV front fusion removes a Z reread but loses to separate front8 plus KV projection. 128-bit shared-memory accesses improve that control without making it the winner. Scalar front fusion passes front-specific bitwise Z/bias, sampled FP64 KV, graph and three-sanitizer checks at128/384/768/1024 (18733); the vectorized control has numerical pilots only.

## Validation and scope

Selected native sources pass30 FP64/edge fixtures each. L384 candidate has retry/changed-path graphs at64/128/384 and all three sanitizers at128/384 (18722). L1024 candidate has retry/graph and all three sanitizers at1024 (18721). Existing hot6t768 qualification is18586/18598. The selected default entry passes complete-module FP64, mask/affine cases, changed-input/weight/mask graphs, fullgraph and no_grad/inference_mode at all three lengths in18745, with actual kernel attribution. All-masked semantics remain ours zero-update versus upstream mean-V; LN affine fixtures match upstream BF16 rounding. No unrestricted numerical parity or backward qualification is claimed.

Job18702 failed before compilation because sources had not been generated with the available python3 executable;18703 reran successfully. Every subsequent pilot/control job completed.

Evidence: [results JSON](head4-results.json), [pilots](head4-pilots.json), [NCU](head4-ncu.json), [design](HEAD4_KV_DESIGN.md), selected logs and per-shape JSONs.
