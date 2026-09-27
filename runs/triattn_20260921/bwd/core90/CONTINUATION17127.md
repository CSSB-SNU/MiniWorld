# Continuation from installed job17069

> **Status:** Installed and verified by job17211; see [current installed report](README.md). The earlier queue snapshot below is historical.

The selected `rs8_async_bias64` overlaps deterministic FP32 bias partial reduction with the next query tile. Two otherwise idle producer warps perform reduction; separate ready/empty barriers protect both dS stages. No global tensor or approximate reduction was added. dQ remains `rs_softmax_overlap`.

## Direct comparison with the preceding installation

Job17163: 24 balanced AB/BA rounds, 20 graph replays. All input and parameter gradients are bitwise equal for every case. Reductions are paired medians; displayed times are arm medians.

| L | Direction | Before BWD ms | Candidate BWD ms | BWD reduction | F+B reduction |
|---:|---|---:|---:|---:|---:|
| 384 | starting | 0.9135 | 0.9021 | 1.21% | 0.73% |
| 384 | ending | 0.9678 | 0.9556 | 1.26% | 0.78% |
| 768 | starting | 5.0982 | 5.0086 | 1.72% | 0.97% |
| 768 | ending | 5.3014 | 5.2060 | 1.91% | 0.88% |
| 1024 | starting | 11.1390 | 10.8526 | 2.41% | 1.04% |
| 1024 | ending | 11.4330 | 11.2244 | 1.75% | 1.01% |

## Qualification and SOL

- Qualification17162: ten full-module cases including dropout, SGD updates, frozen weights and fullgraph; independent FP64/masks24 records; exact-zero bias cancellation; memcheck/racecheck/synccheck at64 and256 all pass.
- Build17151 and qualification/full comparison/NCU17164 refer to the same source and binary digests.
- L768 candidate dK/dV main:2.4288ms, compute46.06%, memory49.61%, maximum SOL49.61%. The unchanged dQ retains its hash-matched NCU17063 result:78.09%. **SOL90 is not achieved.**
- The adoption script dry-run passes all checks against the intact checkpoint17069. This is not an installed-path verification result.

## Queue and publication

At the latest queue snapshot, node01 has no free GPU. Slurm estimates promotion start at2026-09-27 02:22:13 (cluster local time); this is not guaranteed and can change. QoS remains normal_h100. Job17175 rechecks the frozen checkpoint and candidate gates only after allocation, then installs, runs final integrity/dispatch/cold compile/AMP/full timings plus attribution, and refreshes reports. Failure restores the prior changed source, manifest and report files. No package file was changed while queued.
`promotion-17175.json` is written when the job runs; `PROMOTION_COMPLETE 17175` in `final-17175.log` and state `complete` mark actual completion. `promote_async.py` and `promote_async.sbatch` define the complete authorized workflow.
Further Q/dO shared prefetch-depth trials17167/17168 depend on successful17175. They add no HBM buffer. They have no measurement yet and will compare with the newly installed async candidate.

## Other measured controls

`continuation17127-results.json` and `EXPERIMENTS.md` record all candidates. Q16, shared FP32 probability staging and all adjacent-query dQ K/V-sharing controls were slower. Three dK/dV consumers improved isolated timing and passed qualification17154, but full comparison17155 gave no L1024 improvement, so that variant is not selected. Its producer56/consumer152 control17160 spills and serializes WGMMA and is slower.
