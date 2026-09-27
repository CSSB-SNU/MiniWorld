# Q and gate projection fused with attention

Full-QKV follow-up: [twelve CUDA/TMA controls and validation](QKV_STREAM_RESULTS.md).
HBM reads fall, but target-length full FWD still regresses; installed18006 below remains selected.

2026-09-26. Installed native CUDA/TMA `qg_scoped`, promotion18006, against frozen
Q-only checkpoint17774. H100 B1/H4/D32/C128 BF16, L384/768/1024, both directions.
Experiments ran on node02 / normal_h100. **SOL90 remains unmet.**

## Installed full workload

64 alternating AB/BA rounds x40 CUDA Graph replays per arm, including all input
and parameter gradients. Times are arm medians; reductions use paired median
ratios. All outputs and gradients are bitwise equal. Every FWD bootstrap95%
lower bound exceeds1 and every F+B median improves. Intervals describe within-run
uncertainty, not cross-node guarantees. Raw rounds are in
[qg-fusion-results.json](qg-fusion-results.json).

| L | Direction | Full FWD ms | Reduction | F+B ms | Reduction |
|---|---|---:|---:|---:|---:|
| 384 | starting | 0.3849 -> 0.3679 | 4.43% | 1.2426 -> 1.2313 | 0.99% |
| 384 | ending | 0.4812 -> 0.4627 | 3.87% | 1.3891 -> 1.3699 | 1.30% |
| 768 | starting | 1.9570 -> 1.8890 | 3.16% | 6.7294 -> 6.6773 | 0.79% |
| 768 | ending | 2.3124 -> 2.2566 | 2.58% | 7.2918 -> 7.2400 | 0.72% |
| 1024 | starting | 4.0354 -> 3.9401 | 2.40% | 14.4501 -> 14.3590 | 0.61% |
| 1024 | ending | 4.6792 -> 4.5827 | 2.10% | 15.3955 -> 15.3053 | 0.54% |

## Implementation and HBM traffic

Each64-query CTA loads normalized Z once, loads Q/gate weights together, projects
and saves gate, then projects and saves Q. Q stays in shared memory for attention.
After all projection readers retire, the32KiB Z/weight scratch becomes K/V/bias
pipeline storage. The gate save finishes reading the shared output tile before
Q overwrites it. Removing a redundant post-Q barrier and shortening projection
live ranges improved the scoped control. K/V still use two GEMMs.

128 threads,37,888B shared memory,96 registers,five CTA resource capacity per SM.
Compiler reports8B stack,4B spill stores and8B spill loads; these are not zero.
This is cooperative phase scheduling, with two-stage TMA attention loading;
there is no dedicated projection producer running concurrently with consumers.

NCU17990, complete **Q/K/V/gate projections + attention** atL768:
HBM reads911.06 ->760.15MB, writes703.89 ->723.44MB, summed kernel duration
1.33136 ->1.26912ms. Do not compare these four-projection totals directly with
the earlier three-projection report. The fused kernel alone grows1.02218 ->1.06320ms
because it now computes gate; the separate gate GEMM disappears. Its SM throughput
is68.34% and L2 throughput86.76%. Neither is whole-module SOL90 evidence.

## Controls and full-QKV follow-up

All four projections plus attention, compared with17774. Positive means reduced
time. Core controls use12 paired rounds x20 replays; full-module selection above
uses64x40. Every control passes six mask/logit fixtures at all three lengths.

| Artifact | L384 reduction | L768 reduction | L1024 reduction |
|---|---:|---:|---:|
| qg_serial3 | 5.67% | 2.36% | 1.62% |
| qg_preload3 | 6.87% | 3.26% | 2.25% |
| qg_scoped | 8.31% | 5.31% | 3.62% |
| qkv_five | -2.47% | -20.98% | -10.27% |

The full-QKV control keeps K/V resident and uses five projection/attention
warpgroups instead of six. It removes the six-warpgroup register spill, but
still needs up to218,112B shared memory and one CTA per SM. It computes gate
with a separate GEMM to preserve the same four-projection comparison boundary.
AtL768,12 query tiles require three rounds across five warpgroups; only two
warpgroups have work in the last round. Six warpgroups had two full rounds.
It is an isolated performance control, not a promoted implementation; it has
not received full-module gradient/sanitizer qualification.

Fresh NCU18004, four-projection boundary against17774: HBM reads910.97 ->406.49MB,
writes703.30 ->728.62MB, duration1.33053 ->1.58957ms. Total traffic falls29.68%,
yet execution grows19.47%. Its fused kernel SM throughput is52.31% and L2
throughput30.66%. Eliminating spills alone did not solve this schedule.

Full QKV fusion does remove HBM reads. Earlier resident-QKV measurements reduced
three-projection+attention reads911 ->245MB but increased time1.27 ->1.91ms.
Training still requires saved Q/K/V for backward. Lowering shared-memory occupancy
cost and projection waits remains necessary before selecting full residency.

## Validation and installed dispatch

- Staged17994:18 full-module fixtures plus8 independent FP64 gradient fixtures.
- Staged17995:cold fullgraph, schema/fake/AOT, AMP, partial/frozen weights and guards.
- Sanitizer17996:memcheck/racecheck/synccheck atL128/L768, mixed/all-masked;zero errors/hazards.
- Promotion18006:fresh installed dispatch/opt-out checks and repeated full FWD/BWD/F+B.
- Existing CUDA sources and binaries are byte-identical to17774. Only the front
  dispatcher and two existing manifests change; the new implementation has its
  own `qg_fwd_manifest.json`. The checkpoint captures all installed files.

`MINIWORLD_TRIATTN_QG_FWD=0` restores Q-only17774. `MINIWORLD_TRIATTN_Q_FWD=0`
restores separate projections. The broader training-forward opt-out still works.
All backward saves, numerical operations and native backward binaries are preserved.

[Promotion](qg-promotion-18006.json), [snapshot](checkpoint18006/snapshot.json),
[CUDA source](qg_scoped/fused.cu), [previous Q fusion](IMPROVEMENT_17774.md),
[earlier full-QKV experiments](QKV_FUSION_RESULTS.md).
