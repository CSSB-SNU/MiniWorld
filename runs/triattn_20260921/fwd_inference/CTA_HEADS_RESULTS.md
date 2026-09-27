# Head-last bias and four-head attention CTA results

2026-09-26: implemented contiguous BF16 `[1,L,L,4]` bias and one CTA covering all four heads. Six CUDA/TMA variants were measured. None beats the current selected complete inference FWD; the selected entry remains L384 `h4kv_local1`, L768 `hot6t`, L1024 `hot4t`, all with `front8`. Production dispatch and all 48 training18246 files remain unchanged. SOL90 is not achieved.

## Best head-last candidate by length

Node02 / normal_h100, BF16 B1/C128/H4/D32, complete input-preserving inference FWD, 24 balanced AB/BA rounds x30 graph replays. Every projection, LN, bias, attention, gate and residual is included. No bias-layout conversion is performed in these measured paths. Times are separate medians; percentage changes use median paired ratios. These are rejected pilots, not qualified promotions.

| L | Direction | Candidate | Current ms | Head-last ms | Slower than current | Anthropic ms | Slower than Anthropic |
|---:|---|---|---:|---:|---:|---:|---:|
| 384 | starting | cta_heads4_s2 | 0.3427 | 0.4690 | +37.10% | 0.4565 | +1.44% |
| 384 | ending | cta_heads4_s2 | 0.3405 | 0.4670 | +37.34% | 0.4606 | +0.11% |
| 768 | starting | cta_heads4_s2 | 1.6592 | 2.4561 | +48.01% | 1.7005 | +47.04% |
| 768 | ending | cta_heads4_s2 | 1.6666 | 2.4607 | +47.79% | 1.6773 | +48.47% |
| 1024 | starting | cta_heads4_s2 | 3.4954 | 5.1819 | +48.00% | 3.4852 | +48.64% |
| 1024 | ending | cta_heads4_s2 | 3.4930 | 5.1805 | +48.09% | 3.4580 | +50.37% |

## All controls

Starting / ending complete-FWD milliseconds:

| Candidate | L384 | L768 | L1024 |
|---|---:|---:|---:|
| cta_heads4_s2 | 0.469 / 0.467 | 2.456 / 2.461 | 5.182 / 5.181 |
| cta_heads4_s1 | 0.541 / 0.542 | 3.034 / 3.030 | 6.313 / 6.313 |
| cta_heads4_planar | 0.408 / 0.406 | 2.021 / 2.025 | 4.153 / 4.137 |
| cta_heads4_repack | 0.491 / 0.489 | 2.621 / 2.635 | 5.580 / 5.579 |
| cta_heads4_scalar | 0.469 / 0.470 | 2.468 / 2.479 | 5.221 / 5.219 |
| cta_heads4_indep | 0.393 / 0.392 | 1.932 / 1.935 | 3.992 / 3.996 |

`s2`: two packed KV/bias stages, one shared all-head completion barrier, direct vector bias reads. `s1`: one stage. `repack`: in-place shared-memory conversion to head planes, then ldmatrix reads; no extra HBM tensor. `scalar`: direct reads of just the required head value. `planar`: same shared four-head CTA using head-major global bias and ldmatrix. `indep`: planar control with independently loaded and retired per-head KV/bias rings. All candidates share normalized Z during Q/gate projection.

## Producer-only ablation

The old front already writes `[1,4,L,L]` directly, so there was no separate transpose kernel or `[1,L,L,4]` intermediate to remove. The new front writes one eight-byte vector per query/key pair. Z and logical bias are bitwise equal, including changed-input/affine/weight/mask graph replay. Layout conversion appears only in untimed validation. Matched 48x40 producer graphs:

| L | Direction | Planar us | Head-last us | Time reduction |
|---:|---|---:|---:|---:|
| 384 | starting | 52.80 | 53.25 | -0.85% |
| 384 | ending | 52.71 | 52.92 | -0.40% |
| 768 | starting | 195.80 | 194.08 | +0.88% |
| 768 | ending | 196.35 | 193.30 | +1.55% |
| 1024 | starting | 344.37 | 340.53 | +1.12% |
| 1024 | ending | 345.29 | 338.65 | +1.92% |

## Profiling and interpretation

Warm-cache L768 NCU measured `s2` attention at 1.914 ms and 604.7 MB DRAM traffic (18809), versus the existing row-local streaming attention at 1.104 ms and 604.9 MB (fresh18831). The separate joint KV projector adds approximately 0.160 ms and 430.8 MB in both paths. The score matrix is never written to HBM. Sharing Z reduces L2 read sectors, but DRAM traffic is essentially unchanged because the previous CTA order already reused it through cache. Short-scoreboard stalls rise from 0.363 to 3.686 per issued instruction; barrier stalls rise from 1.054 to 1.531. The planar common-barrier control takes1.430ms and the independent head-ring control1.306ms, both at approximately604.8MB; their short-scoreboard stalls remain near0.36 while barrier stalls are2.104 and1.320 respectively. Packed bias read dependencies and grouping/synchronization costs outweigh the producer gain. NCU is diagnostic; the paired full-FWD runs determine the result. This rejects these implementations, not every possible head-last algorithm.

## Validation and limits

Every candidate passes 30 native fixtures: six cases at L64/128/384/768/1024, including sampled FP64, masks, large logits and changed-input/weight/bias graphs. `s2` and `repack` additionally pass retry-path/changed-path graphs at64/128/768/1024, all three sanitizers at128, and complete module FP64/masks/affine/changed-input-weight-mask/fullgraph/no_grad tests at384 (18818). The front passes independent FP64 at128/384/768/1024 and all three sanitizers at128. No full-length sanitizer qualification or production promotion is claimed. Original Anthropic all-masked behavior still differs: our zero update versus upstream mean-V; no unrestricted parity claim. Original source hashes and actual profiled kernel names are recorded per pilot.

Jobs18806,18812,18824 are native/full-FWD pilots;18809/18831 are NCU;18818 is validation and producer ablation. [Design](CTA_HEADS_DESIGN.md), [complete evidence](cta-heads-results.json), [raw paired pilot summary](cta-heads-pilots.json), [profiling](cta-heads-ncu.json). All six owned jobs completed with exit0. NCU exports emitted the pre-existing Python-site encoding warning, but all CSV metrics, reports and process exits were verified.
