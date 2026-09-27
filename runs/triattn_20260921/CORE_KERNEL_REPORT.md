# Attention calculation kernel optimization

2026-09-21, H100 80GB, BF16 C128/H4/D32. Experiments and reproduction are in
`core_tiles/`. This follow-up changes the main CUDA attention calculation loop.

## Selected implementation

The `ta_core_broadcast` CUDA kernel specializes the M1 hot pass for one key mask
shared by all pair rows (or no mask). Row kinds and ranges are common, so the hot
pass derives one key range and removes the per-row irregular/ragged-mask branches,
leading/trailing follower loops, and generic row-range bookkeeping. It still skips
empty batches and masked leading/trailing key tiles. The exact SAFE recomputation
uses the original generic implementation. No probability approximation, rounding
rule, reduction order, or test vector changes are introduced.

The tile remains M128/N128, three pair rows per CTA, 512 threads: one producer
warpgroup and three consumer warpgroups. It retains TMA Q/K/V loads, the two-slot
K/V ring, shared fragment-order bias ring, QK issued two chunks ahead, delayed PV,
and the original per-warp barrier ownership. The producer has 32 registers and
each consumer has 160 through dynamic register allocation. This is still one
resident CTA per SM; the improvement comes from less control work in the hot
calculation path.

Dispatch is scoped to square B1/H4/D32, L768/1024, BF16 on the qualified
Torch 2.10.0+cu128 / CPython 3.10 / sm90 stack, with a bool row-broadcast mask or
no mask. Other shapes and masks keep the existing core. The optional hook in the
native wrapper reuses the same original mask/bias preparers and output/census
buffers. `FPF_TRIATT_CORE_BROADCAST=off` selects the original calculation kernel;
`auto` is the default. Existing mask preparation optimization remains enabled.
L384 is outside this core change.

The separately named library and C++ namespace avoid template-local static
collisions with the original M1 library. Source/binary hashes are checked when
loading the new artifact. The original M1 .so stays available for other masks
and opt-out. The native wrapper's recorded hash and package SHA256SUMS are
refreshed without regenerating expected outputs.

## Prototype measurements

Job 14014, three alternating CUDA graph rounds per variant, both directions.
The earlier mask-preparation optimization was disabled for every prototype
variant, so it cannot explain these differences. Actual kernel names were
checked and any flash_triattn fallback was rejected.

| Direction | L | Original hot kernel us | Specialized hot kernel us | Original block us | Specialized block us |
|---|---:|---:|---:|---:|---:|
| starting | 768 | 877.62 | 842.03 | 1440.24 | 1408.33 |
| ending | 768 | 871.74 | 832.86 | 1437.79 | 1440.66 |
| starting | 1024 | 1919.86 | 1863.84 | 2990.11 | 2948.73 |
| ending | 1024 | 1896.23 | 1845.79 | 3032.12 | 2979.10 |

Hot kernel values above are separate CUPTI measurements of five whole-block
calls. Block values are graph medians. Do not equate their absolute difference:
clock/power state and timing method differ. In particular, the ending L768 block
was slightly slower in this prototype run despite the faster calculation kernel.
The final installed measurements use repeated CUDA-graph replay profiling.

## Rejected structural candidates

All measured candidates passed their initial FP64 accuracy gate before timing;
only the broadcast specialization is selected. A higher resident CTA count by
itself was insufficient.

| Candidate | Job | L768 hot kernel us | Outcome |
|---|---:|---:|---|
| R1/M128, two CTAs per SM | 13985 | 1154 | bitwise equal, slower; less bias reuse |
| M64/N64, two consumer warpgroups | 13982 | 1303 | slower; exposed pipeline waits |
| M64/N64, three consumer warpgroups | 13993 | 1677 | register spills and WGMMA serialization |
| Three consumers, fragment bias and delayed PV wait | 13999 | 1653 | still spills, slower |
| Two consumers, fragment bias and delayed PV wait | 14003 | 1146 | no spills, still slower than original ~885 us |
| Two-CTA cluster, broadcast specialization, TMA bias multicast | 14014 | 912 | bitwise equal but slower than unclustered 842 us |

The M64 candidates changed tile rounding/seed order and were assessed against
FP64 (initial RMS remained within 1.05 times the original). They are experimental
only. The selected specialization and R1 were bitwise equal to the original.
The clustered version remains a benchmark option, not a serving selection.

## Correctness evidence

Job 14018 passed 80 exact comparisons: L768/1024, contiguous and strided Q/K/V,
B2 with seven pair rows (CTA tails), noncontiguous key-mask storage, no mask,
dense/ragged/prefix/empty masks, one live key, a first bias chunk containing only
negative infinity, large logits, and forced SAFE recomputation. Both clustered
and unclustered candidates matched the original bitwise. Job 14014 also checked
nonzero-weight whole-block outputs and an independent FP64 attention-row reference
in both directions; the selected candidate's core and block deltas were zero.

## Installed results

Job 14025 verified default dispatch and opt-out, exact core/block outputs on five
mask patterns for both lengths and directions, and rejected flash_triattn fallback.
The prior broadcast-mask preparation remained enabled for **both** variants.

To reduce clock-state drift in the isolated profiles, job 14033 put the original
and new block into the **same CUDA graph**, reversed their order in the middle
round, warmed 100 replays, then profiled 50 paired replays per round. The main
attention kernels have distinct names, so their durations remain separable.
Medians of three rounds:

| Direction | L | Original calculation us | New calculation us | Reduction | Original block us | New block us |
|---|---:|---:|---:|---:|---:|---:|
| starting | 768 | 916.48 | 880.97 | 3.88% | 1428.28 | 1380.92 |
| starting | 1024 | 2072.72 | 2013.08 | 2.88% | 3000.08 | 2949.16 |
| ending | 768 | 906.93 | 872.20 | 3.83% | 1459.51 | 1443.17 |
| ending | 1024 | 2081.75 | 2022.34 | 2.85% | 3015.30 | 2951.77 |

Calculation values are paired-profile medians (job 14033); block values are
separate alternating graph-event medians (job 14025). They are different timing
experiments. Paired per-round reductions were 3.6–3.9% at L768 and 2.7–2.9% at
L1024. Whole-block reduction was 1.1–3.3% in the final installed run.

NCU job 14028 confirmed the new hot kernel at 845.728 us, SM throughput 57.70%,
tensor activity 32.72%, active-warps occupancy 21.26%, and 191,488 shared bytes
per CTA. Register/shared-memory residency limits remain one CTA per SM. DRAM
traffic was 607.719 MB; L2 traffic was 154,928,524 sectors (4.958 GB). This improves
the calculation time without claiming that the core has reached SOL90. The
older baseline capture was 880.352 us and SM throughput 55.58%; captures from
different jobs are supporting diagnostics, not the paired timing evidence.

Job 14034 passed **racecheck (zero hazards), synccheck and memcheck (zero errors)**
on the installed hot and SAFE CUDA kernels, with row tails, strided operands,
partial/empty batches, late seeding and forced SAFE recomputation. It also passed
all **17/17** shipped H100 package vectors. No expected outputs were regenerated.
The initial sanitizer harness in job 14026 stopped before testing: expanding a
single pair row did not create stride zero. The corrected harness uses two pair
rows; only job 14034 supplies sanitizer evidence.

Artifact SHA256: `f7fdfb954bc66f0e437f382725f0fcd1615020e4cde376c3bfe9f88644875505`.
The exported source patch is `codex_core_kernel.patch`, incremental over the
previous mask-preparation change. The tested shared library is installed at
`oc/opt_core/kernels/triattn_core_broadcast/triattn_broadcast.so`.

## Reproduction

```bash
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/core_tiles/build_broadcast.py
sbatch runs/triattn_20260921/core_tiles/check.sbatch
sbatch runs/triattn_20260921/core_tiles/broadcast_all.sbatch
# Installs the qualified artifact, then measures the actual default path:
sbatch runs/triattn_20260921/core_tiles/serving.sbatch
sbatch runs/triattn_20260921/core_tiles/sanitize.sbatch
sbatch runs/triattn_20260921/core_tiles/profile.sbatch
```

The installer preserves the generic binary and refreshes wrapper records.
Serialize installation against any jobs reading the same `oc/` tree.

To rebuild directly from the installed source package (or after applying the
source patch from within `oc/`):

```bash
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/oc/opt_core/kernels/triattn_core_broadcast/build_native.py
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/core_pipeline/refresh_core_record.py
```

The first command creates the new binary and its source/binary manifest. The
second repairs the native-wrapper record and the package integrity sums. The
source patch does not contain binary bytes. Existing generic core binaries and
all expected vectors are preserved.
