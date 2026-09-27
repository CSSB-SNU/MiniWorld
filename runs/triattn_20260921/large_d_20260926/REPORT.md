# Wide TriangleAttention results — 2026-09-27

Qualified experimental native CUDA implementation; installed engine dispatch is unchanged. C128 work is closed at training18246/inference19157. The original engine baseline is the original Triton module/core, not our installed C128 CUDA checkpoint.

Here D means pair width and total QKV width:256/512, four heads, hence head64/128. B1, H10080GB on node02, BF16 activation/projection weights, FP32 LN affine. Timings include LayerNorm, all projections, attention, gate, residual and all nine input/parameter gradients where applicable. FWD retains training saves. They are CUDA Graph timings, not eager host-launch latency.

Selection: width256 uses native attention plus fused projection input gradients; width512 uses native attention with the original projection path. Both retain one head per CTA. dK/dV groups4 or2 outer rows per CTA, respectively.

| Width | L | FWD speedup | BWD speedup | F+B speedup |
|---:|---:|---:|---:|---:|
| 256 | 384 | 1.070–1.091x | 1.245–1.265x | 1.193–1.217x |
| 256 | 768 | 1.117–1.150x | 1.291–1.315x | 1.228–1.245x |
| 512 | 384 | 1.037–1.043x | 1.111–1.115x | 1.079–1.096x |
| 512 | 768 | 1.073–1.096x | 1.148–1.150x | 1.114–1.118x |

Ranges cover starting and ending directions. Every speedup uses original/candidate measurements from the same job and alternating replay order; widths256 and512 use jobs19431 and19416, respectively. Each cell has16 paired rounds of10 replays; profiler uses3 further replays and verifies actual native kernel names and cudaGraphLaunch.

| Width | L | Direction | Original Triton F+B ms | Selected CUDA F+B ms | Time saved |
|---:|---:|:---|---:|---:|---:|
| 256 | 384 | starting | 3.290 | 2.704 | 17.8% |
| 256 | 384 | ending | 3.633 | 3.045 | 16.2% |
| 256 | 768 | starting | 17.091 | 13.730 | 19.7% |
| 256 | 768 | ending | 18.424 | 14.998 | 18.6% |
| 512 | 384 | starting | 6.173 | 5.634 | 8.7% |
| 512 | 384 | ending | 6.856 | 6.355 | 7.3% |
| 512 | 768 | starting | 31.290 | 27.987 | 10.6% |
| 512 | 768 | ending | 33.788 | 30.317 | 10.3% |

**What changed.** Wide native FWD uses TMA Q/K/V/bias and WGMMA online softmax. dQ keeps Q/dO in registers, reuses their retired shared storage for bias and doubles K/V stages. dK/dV uses one producer warpgroup and two consumers, fuses bias-gradient partial generation, then reduces those partials. Shared layouts are transposed from the actual load layout; the first naive wide layout failed FP64 and is retained under rejected_v1_transpose.

At width256, fused projection dgrad accumulates four square products and the narrow bias product on chip before one BF16 store, eliminating four full input-gradient intermediates and their separate additions. Parameter gradients retain cuBLAS. Width512 projection fusion passed correctness but was slower than attention-only CUDA and is rejected for selection. The64-row projection tile beats128 rows at both lengths and widths; projection-tiles-19436.json records the bitwise/timing comparison.

**Resource facts.** Head64/128 FWD uses58,368/99,328 bytes SMEM and96/129 registers. dQ uses50,176/99,328 bytes and154/218 registers. Grouped dK/dV uses175,104/157,696 bytes. All builds have zero spills. Projection M64 uses90 registers; its experimental setmaxnreg requests were ignored by PTXAS (C7508), so it uses the reported static allocation. Do not claim active register redistribution in that projection kernel.

Bias-gradient FP32 scratch is half the original BF16 per-outer-row scratch at R4/head64. At R2/head128 the byte count is equal; the head128 attention gain comes from kernel scheduling, not a claimed scratch-byte reduction.

**Validation.** Job19405 passes16 independent FP64 core forward/adjoint cases across head64/128, L64/128, no mask/mixed/one-key/all-masked. Initial and changed-input graph replays are bitwise with native eager execution. One-key reference-zero gradients use an absolute bound; all-masked outputs/gradients are exactly zero. Whole-module PyTorch checks19417/19430 cover both directions, dropout0/.1, frozen parameters, changed inputs/masks/dy/weights and two SGD steps. Job19445 checks the selected entry in full-module graphs with three input/weight/dy/mask versions and actual native dispatch.

Core memcheck/racecheck/synccheck19418_0–5 pass at L256, exercising repeated TMA buffer reuse. Projection FP64, changed-gradient/weight graphs and all three sanitizers19429 pass at both widths. No sanitizer errors, race hazards or warnings. Full target-shape paired benchmarks check outputs and all nine gradients against original Triton before reporting timings.

**Remaining bottlenecks.** At L768 starting, selected C256 backward spends about2.87ms in grouped dK/dV,1.52ms in dQ and1.41ms in fused projection dgrad. C512 spends about5.02ms in grouped dK/dV,2.82ms in dQ and2.91ms in separate additions; projection/weight GEMMs also remain substantial. The grouped bias reduction still costs0.59/1.16ms. Native delta preprocessing is not faster than the original Triton preprocessing. These are the next measured opportunities.

**Scope.** No wide NCU SOL measurement was taken; SOL90 is not established. Head dimensions256/512, B>1, other head counts, QK normalization, wide L1024 and torch.compile integration are not qualified by this campaign. Installed training checkpoint18246 retains all48 original file hashes. All owned jobs are terminal.

**Use the explicit candidate.** Run through the project env.sh, put this directory on sys.path, and call selected.attach(model) on a compatible original-Triton TriangleAttention instance. selected.py chooses the qualified width-specific path; it does not replace global engine dispatch. Load the desired trained state normally. Source/binary hashes and evidence paths are pinned in CANDIDATE.json.

**Evidence.** BASELINE.md and baseline-C*-19395.json hold the untouched wide original-Triton profile. Selected paired timings:
- [compare-front-C256-L384-full-19431.json](compare-front-C256-L384-full-19431.json)
- [compare-front-C256-L768-full-19431.json](compare-front-C256-L768-full-19431.json)
- [compare-C512-L384-full-19416.json](compare-C512-L384-full-19416.json)
- [compare-C512-L768-full-19416.json](compare-C512-L768-full-19416.json)
