# Attention core follow-up

> Subsequent work changed the actual attention calculation kernel. See [CORE_KERNEL_REPORT.md](CORE_KERNEL_REPORT.md) for the installed broadcast specialization, final timings and correctness evidence. Statements below about an unchanged core describe this earlier mask-preparation pass.

2026-09-21, H100 80GB, BF16 C128/H4/D32. Work is isolated under
`core_pipeline/`; the installed attention binary remains the original M1 core.

## Current bottleneck

Fresh NCU capture (job 13942, L768, shared irregular key mask) confirms the actual
`triattn_m1_kernel<Traits<0>>` CUDA route, with no flash_triattn fallback:

| Metric | Value |
|---|---:|
| Hot attention kernel | 880.352 us |
| Empty SAFE follow-up kernel | 4.000 us |
| SM throughput | 55.58% of theoretical peak |
| Tensor pipe active | 31.51% |
| XU instruction throughput | 55.58% |
| Issue active | 39.10% |
| Active warp occupancy | 21.38% |
| Shared memory per CTA, including driver | 191,488 bytes |
| Resident CTA limit, shared memory / registers | 1 / 1 |
| DRAM read + write | 607.755 MB |
| L2 sectors | 153,938,636 (4.926 GB) |

This is not the streaming-bandwidth roof used for the surround SOL90 claim.
The kernel already has one producer warpgroup and three consumer warpgroups,
TMA K/V and bias rings, WGMMA, and dynamic register allocation (producer 32,
consumer 160). Merely adding producer/consumer roles is not a missing feature.
The low tensor utilization and exposed instruction/softmax waits are the useful
optimization targets. Shared memory and registers both prevent a second CTA.

## Candidates

1. **Distribute exp2 away from the SFU.** A degree-five polynomial approximates
   the fractional part of exp2, with integer exponent reconstruction. Fractions
   of 1/4, 1/2, or all logits use the polynomial; the rest retain native exp2.
   The SAFE recomputation remains unchanged. This preserves the hot pass's
   roughly 2^-64 operating scale, unlike the rejected full-range fp16 exp2
   experiment. It is a numerical change, so candidates are compared against
   an independent FP64 row reference as well as the original core. The allowed
   RMS error increase in the initial benchmark is 2%; exactness is reported
   separately. No candidate is installed based on timing alone.

2. **Broadcast-mask preparation.** The block's mask has stride zero along pair
   rows, but the original mask preparer repeats word construction for every row,
   atomically ORs identical words, then classifies each row in a second kernel.
   The new kernel owns 32 rows per CTA, computes shared mask words, and writes
   all seven staging outputs in one launch without the preliminary zero fill.
   Each batch extent and OR-mask word has one writer. Empty masks retain the
   uniform-mean-of-V convention. Nonbroadcast masks retain the original path.

## Reproduction

```bash
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/core_pipeline/build.py
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/core_pipeline/build_mask.py
sbatch runs/triattn_20260921/core_pipeline/profile.sbatch
sbatch runs/triattn_20260921/core_pipeline/bench.sbatch
sbatch runs/triattn_20260921/core_pipeline/mask.sbatch
sbatch runs/triattn_20260921/core_pipeline/sanitize_mask.sbatch
```

The candidate Python module and binary use distinct names from the installed
package, so the package's prebuilt injection cannot overwrite the candidate
or make the rebuilt baseline silently refer to the installed binary. Benchmarks
include the installed core and a freshly rebuilt flags=0 control, check kernel
names, and use three alternating CUDA graph rounds.

## Broadcast-mask prototype results

L768 first comparison (job 13951): preparation 18.536 -> 4.071 us, with all
seven staging outputs, counts, core output and whole-block output bitwise equal
for dense, empty, irregular and prefix masks. The main M1 kernel time remained
approximately 872 us. Thus the saving comes from core preparation, not a faster
attention matrix/softmax loop. Whole-block graph medians in this run were
1424.16 -> 1404.02 us, but clock drift makes the 14.5 us component saving the
more reliable attribution.

Racecheck and synccheck passed (job 13953) on B=1/2, pair-row tails, noncontiguous
key-axis storage, lengths 1/385/768/1024 and empty/live batches. Each stage output
was also compared against independent CPU bit packing and range derivation.

Experimental core build pitfall: C++ template-local statics can collide between
two loaded libraries even if the Python extension names differ. An initial build
shared the `configured` dynamic-shared-memory flag with the installed M1 kernel
and failed its launch. All experimental C++ symbols now live in the distinct
`triattn_core_candidate` namespace as well. Failed jobs are not timing evidence.

## Installed mask path

The mask optimization is now the default for broadcast masks at L768/1024 on
the qualified H100 stack. Set `FPF_TRIATT_MASK_STAGE=off` to reproduce the old
preparation path. Nonbroadcast masks and other lengths use the original preparer.
The optional hook is in `triattn_pkg/cuda_b/triattn_m1.py`; the kernel is compiled
into the existing `triattn_surround_tma` extension. Standalone package deployments
without that integration retain their original preparation path.

Job 13957 validated the installed path on both lengths and directions, with
bitwise equality of staging outputs, counters, core and whole-block outputs for
four mask patterns, and actual-kernel checks that reject fallback. Job 13958
passed installed-artifact racecheck/synccheck; job 13959 checked default selection,
opt-out, nonbroadcast masks, unsupported lengths and varied batch/key strides.

| Direction | L | Old preparation us | New preparation us | Old block us | New block us |
|---|---:|---:|---:|---:|---:|
| starting | 768 | 18.58 | 4.00 | 1456.70 | 1439.43 |
| starting | 1024 | 29.01 | 4.18 | 3001.87 | 3003.04 |
| ending | 768 | 18.80 | 4.12 | 1450.18 | 1438.94 |
| ending | 1024 | 28.95 | 4.09 | 3045.46 | 3011.69 |

The stable attribution is ~14.6 us saved at L768 and ~24.8 us at L1024. Whole-block
variation is larger: the starting L1024 block did not improve in this particular
run, despite its faster preparation. Do not attribute the entire block-median
difference to this small kernel change.

The main core `.so` SHA256 remains
`3c766e0296351d643329c7203f94f25ad3c584c5d400b1ca92b74d35fdb225ba`.
The wrapper's recorded Python digest was refreshed; no numeric test vector was
regenerated. `codex_core_mask.patch` is incremental over the CTA follow-up.
After applying it, run `cuda_tma/build_native.py`, `cuda_tma/install_native.py`,
and `core_pipeline/refresh_core_record.py` in the qualified Python environment.
Final installed tests are `core_pipeline/serving.sbatch` and
`core_pipeline/sanitize_serving.sbatch`.

## Exp2-distribution result: rejected

Job 13962 completed with the corrected, independently named CUDA build. The
rebuilt flags=0 control was bitwise equal to the installed M1 core. All three
polynomial candidates passed the initial FP64 RMS criterion, but all regressed
substantially, so no wider numerical qualification or installation was warranted.
The mask optimization was explicitly disabled for every variant in this comparison.

| Variant | Core entry us, including preparation | Change vs rebuilt | FP64 RMS |
|---|---:|---:|---:|
| serving | 955.43 | -0.5% | 0.00028883485 |
| rebuilt | 960.56 | +0.0% | 0.00028883485 |
| poly_quarter | 1099.23 | +14.4% | 0.00028882609 |
| poly_half | 1261.58 | +31.3% | 0.00028882228 |
| poly_all | 2020.50 | +110.3% | 0.00028881036 |

The added range-reduction and polynomial instructions outweighed any benefit from
reducing SFU work in this implementation. This supports rejecting this particular
substitution; it does not establish that every alternative exp2 implementation
must be slower. The main attention binary and its numerical algorithm remain
unchanged.

The next structural question is whether a smaller query/bias tile can permit two
resident CTAs without losing too much bias reuse, while keeping softmax and MMA
work overlapped. This has not been implemented or measured in this pass. Simply
adding more CTAs to the grid cannot overcome the current 191 KB shared-memory
and full-register-file residency limits.
