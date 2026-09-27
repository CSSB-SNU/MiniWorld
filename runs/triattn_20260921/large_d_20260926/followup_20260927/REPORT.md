# Wide TriangleAttention follow-up — 2026-09-27

Qualified explicit experiment; installed engine dispatch is unchanged. The previous qualified wide candidate and the original Triton engine are distinct baselines. Width D=256/512 is total pair/QKV width with4 heads (head64/128), B1, BF16 on node02 H10080GB. FWD remains the previous native attention implementation; this follow-up targets backward HBM intermediates.

| Width | L | BWD vs original Triton | F+B vs original Triton | BWD vs previous wide | F+B vs previous wide |
|---:|---:|---:|---:|---:|---:|
| 256 | 384 | 1.272–1.293x | 1.207–1.233x | 1.018–1.024x | 1.012–1.014x |
| 256 | 768 | 1.334–1.357x | 1.251–1.281x | 1.029–1.038x | 1.017–1.026x |
| 512 | 384 | 1.251–1.270x | 1.175–1.195x | 1.126–1.137x | 1.081–1.095x |
| 512 | 768 | 1.262–1.267x | 1.186–1.218x | 1.103–1.107x | 1.074–1.091x |

Ranges cover starting and ending. All three arms run in the same job/process on the same GPU, using all six replay-order permutations,18 rounds of10 CUDA Graph replays. Outputs and all9 input/parameter gradients pass against original Triton before timing. Profiler confirms3 cudaGraphLaunch calls and actual native/core/compact-bias kernels. The new width512 projection path shows cuBLAS beta-add GEMMs plus the native vector epilogue. Each BWD/F+B paired bootstrap95% interval for time saved over the previous candidate is positive; raw rounds and interval calculation are in summary.json. These intervals cover within-job variation, not all future machines/runs.

| Width | L | Direction | Original Triton F+B ms | Previous wide F+B ms | New wide F+B ms |
|---:|---:|:---|---:|---:|---:|---:|
| 256 | 384 | starting | 3.301 | 2.715 | 2.677 |
| 256 | 384 | ending | 3.630 | 3.045 | 3.008 |
| 256 | 768 | starting | 17.176 | 13.762 | 13.412 |
| 256 | 768 | ending | 18.458 | 15.012 | 14.758 |
| 512 | 384 | starting | 6.173 | 5.657 | 5.165 |
| 512 | 384 | ending | 6.827 | 6.280 | 5.808 |
| 512 | 768 | starting | 31.130 | 27.883 | 25.562 |
| 512 | 768 | ending | 33.530 | 30.356 | 28.268 |

**Selected changes.** Both widths retain FP32 local bias-gradient accumulation but store the grouped partial buffer as BF16, cutting that buffer and its logical write/read byte count in half. The final reduction is FP32. At L768, partial-buffer size drops from1.812→0.906GB for width256 and3.624→1.812GB for width512 (decimal GB). dK/dV remains bitwise equal in the direct comparison; dBias gains one BF16 rounding step. The buffer is still cubic in L and has not been eliminated.

Width256 retains the previous native TMA/WGMMA projection-dgrad fusion. Width512 uses cuBLAS GEMMs that accumulate into the same BF16 dX output with beta=1, then a native CUDA bias-projection epilogue. This removes separately materialized projection dX tensors and their separate addition kernels. It still reads/writes dX between GEMMs; it is not an all-on-chip projection fusion. The epilogue specializes256/512 channels, replaces integer division with compile-time indexing, and accesses BF16x2.

**Measured effects at width512/L768 starting.** The initial scalar bias epilogue took1.193ms; the bitwise-equivalent specialized vector epilogue takes0.519ms. The prior separate add kernels took about2.91ms and the selected path retains about0.58ms of residual additions. Bias reduction drops from about1.17ms to0.69ms. Grouped dK/dV remains around5ms and dQ around2.8ms; they are still major bottlenecks. Width256 bias reduction drops from about0.59ms to0.35ms.

**Rejected controls.** Four new native projection organizations were measured at width512 against the previous custom fused projection: a3-stage pipeline was approximately tied; two consumer warpgroups sharing weights, N256 tiles, and K128 tiles were slower. Initial aggressive occupancy requests for three configurations failed PTXAS C7602; the relaxed configurations compiled and passed bitwise/FP64 checks before being timed. These controls were not selected or promoted. The beta-accumulate path with the scalar epilogue was also slower than the final vector epilogue. At width256 beta accumulation lost to the previous native fusion, so that width preserves it.

**Accuracy and safety.** Compact-bias job19508 passes16 independent FP64 output/adjoint cases (head64/128, L64/128, four masks), plus changed-input graph replay. Whole-module PyTorch job19539 covers both directions, dropout, frozen branches and two SGD steps with changed inputs/weights/dy/masks; maximum relative error is under0.85%. Job19536 confirms full-module changed-state graphs and actual dispatch. Native epilogue changes are bitwise equal to the scalar version at both widths and small/nonmultiple/large row counts.

Compact bias storage increases dBias FP64 relative-L2 error in the stress fixture from about0.242% to0.296%; all pre-existing full-core and full-module tolerance limits remain unchanged. dO magnitude65536 passes with finite dBias values above4million, preserving BF16 range. The cuBLAS accumulation path has FP64 projection error about0.356%, matching the original split BF16 reference (the all-FP32 fused custom path was about0.166%). It passes the full-module comparison budget. These are measured rounding differences, not bitwise claims for the entire new backward.

All three sanitizer tools pass for compact bias at L256 with repeated TMA-stage reuse (19522_0–5), and for the final vector epilogue/accumulation graph at both widths and boundary row counts (19536). No sanitizer errors, race hazards or warnings. Initial scalar accumulation also passed19529.

**Use and scope.** Import selected_next.attach from this directory and apply it to a compatible original-Triton TriangleAttention instance. It chooses prior projection fusion for width256 and beta accumulation plus the vector epilogue for width512, with compact bias partials for both. Baseline runtime sources/binaries are frozen and verified through BASELINE.json; all48 installed checkpoint18246 file hashes remain unchanged. New native binaries and runtime sources are pinned in CANDIDATE.json. This is an explicit experimental entry, not a global engine installation.

No new FWD algorithm improvement is claimed. No NCU wide SOL measurement was taken and SOL90 is not established. This does not qualify B>1, other head counts, head dimensions256/512, wide L1024, QK normalization or torch.compile integration. All owned Slurm jobs are terminal; the failed builds and dependency-cancelled screen are retained as rejected evidence.

**Evidence.** Selected paired result files:
- [compare-C256-L384-prior-b1-19517.json](compare-C256-L384-prior-b1-19517.json)
- [compare-C256-L768-prior-b1-19517.json](compare-C256-L768-prior-b1-19517.json)
- [compare-C512-L384-accumulate_fast-b1-19537.json](compare-C512-L384-accumulate_fast-b1-19537.json)
- [compare-C512-L768-accumulate_fast-b1-19537.json](compare-C512-L768-accumulate_fast-b1-19537.json)
