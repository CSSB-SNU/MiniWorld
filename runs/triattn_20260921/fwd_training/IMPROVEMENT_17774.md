# Installed training attention forward

2026-09-26. **Native CUDA/TMA Q-projection plus attention fusion is installed and
verified**, promotion17774, artifact `q_only_head4`. Scope: H100,
B1/H4/D32/C128 BF16, L384/768/1024, both directions. Experiments used
node02 / normal_h100. **SOL90 remains unmet.**

## Actual installed full-workload comparison

Compare the preserved17628 separate-projection path with the installed Q-fused
path in the same process, using64 alternating AB/BA rounds x40 CUDA-graph
replays per arm. Times below are arm medians; reductions use paired median
ratios. Every output and input/parameter gradient is bitwise equal. Every FWD
bootstrap95% lower bound is positive, and every F+B median improves. Full raw
rounds and intervals are in [installed-results.json](installed-results.json).
These intervals describe within-run uncertainty, not cross-node guarantees.

| L | Direction | Full FWD ms | Reduction | F+B ms | Reduction |
|---|---|---:|---:|---:|---:|
| 384 | starting | 0.4068 -> 0.3863 | 4.94% | 1.2615 -> 1.2484 | 1.29% |
| 384 | ending | 0.5005 -> 0.4821 | 3.67% | 1.4075 -> 1.3910 | 1.13% |
| 768 | starting | 1.9930 -> 1.9586 | 2.39% | 6.8273 -> 6.7983 | 0.43% |
| 768 | ending | 2.3626 -> 2.3179 | 2.02% | 7.3826 -> 7.3471 | 0.58% |
| 1024 | starting | 4.1443 -> 4.0717 | 1.74% | 14.5933 -> 14.5231 | 0.49% |
| 1024 | ending | 4.7796 -> 4.7107 | 1.40% | 15.5280 -> 15.4720 | 0.35% |

## Selected implementation

The attention CTA projects its own64 query rows with TMA/WGMMA, saves BF16 Q
for backward, then consumes the same Q shared-memory tile. K/V retain their
two GEMMs and use the existing two-stage attention pipeline. Projection scratch
aliases K/V/bias storage after projection completion. Four consecutive head
CTAs reuse normalized input Z. Resource use:128 threads,90 registers,zero spills,
37,888B shared storage,five CTAs/SM. The attempt to force six CTAs spilled and
was slower. This is a cooperative phase schedule, not dedicated producer/
consumer overlap.

A combined front+attention autograd boundary prevents duplicate Q projection
and preserves all backward saves. Existing backward CUDA binaries and numerical
operations remain unchanged; only the front dispatch/autograd wiring changes.
The original separate-projection path remains as `ln_backward.unfused_forward`.

Set `MINIWORLD_TRIATTN_Q_FWD=0` to use that baseline. Unsupported inputs and
disabled required fusions also retain the baseline. The broader
`MINIWORLD_TRIATTN_TRAINING_FWD=0` opt-out still disables the new Q path.
The existing `cooperative_head2` attention core is preserved for these paths.

## Validation and measured limits

- FP64 projections/attention and8 independent FP64 gradient fixtures;
  production shapes on6 mask/logit fixtures; changed-input CUDA-graph replay.
- Staged17762:18 full-module cases including dropout, optimizer updates,
  fullgraph and frozen parameters, plus the8 FP64 gradient fixtures.
- Sanitizer17766:memcheck/racecheck/synccheck atL128/L768;zero errors/hazards.
- Fresh installation17774:cold fullgraph, schema/fake/AOT, dtype/layout/shape/
  opt-out guards, actual native dispatch, BF16 AMP and partial-parameter gradients.
- Actual installed paired FWD/BWD/F+B repeats the complete production workload,
  including all input and parameter gradients. Promotion rolls back on failure.
- NCU17767, complete projections+attention:L768 HBM reads910.94 ->759.91MB,
  writes555.06 ->570.54MB. Total traffic falls about9.25%, while L2 read sectors
  rise259.00 ->274.58million. Fused kernel SM70.51%,L288.05%;SOL90 is not achieved.
- Seven CUDA fusion candidates are documented in
  [QKV_FUSION_RESULTS.md](QKV_FUSION_RESULTS.md). Full-QKV residency reduced reads
  but lost on large shapes because of projection serialization and occupancy.

## Evidence

[Promotion17774](q-promotion-17774.json),
[frozen package](checkpoint17774/snapshot.json),
[selected CUDA source](q_only_head4/fused.cu),
[all fusion experiments](qkv-fusion-results.json),
[fusion design](QKV_ATTENTION_FUSION.md).

The only existing package files changed are `ln_backward.py` and its manifest;
the new Q forward implementation has its own `q_fwd_manifest.json`. All existing
CUDA binaries/sources, the attention dispatcher and the original forward guard
match checkpoint17628. The new snapshot contains40 files.

Prior results remain in [IMPROVEMENT_17628.md](IMPROVEMENT_17628.md),
[IMPROVEMENT_17345.md](IMPROVEMENT_17345.md),
[INITIAL_17310.md](INITIAL_17310.md), and
[PIPELINE_FOLLOWUP.md](PIPELINE_FOLLOWUP.md).
