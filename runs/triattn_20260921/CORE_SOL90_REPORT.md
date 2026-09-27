# Core follow-up: CUDA/TMA pipeline and fused PV / row sums

> **Latest installation, job 14862:** L1024 core is 4.163%/3.971% faster in paired starting/ending measurements. Installed NCU: L768 795.904us / SM61.349%; L1024 1751.712us / SM65.249%. L768 machine words are unchanged from 4be. [Q1024_REPORT.md](Q1024_REPORT.md) records the package, 82 bitwise cases, full-shape sanitizers and serving verification. **SOL90 remains unachieved.**

2026-09-22, H100 80GB, C128/H4/D32 BF16. **SOL90 has not been reached.**
The installed changes preserve arithmetic and add repeatable improvements.
Approximation and alternative CTA experiments remain isolated in `core_sol90/`.

## Installed change

For square N=S768 or1024, H4 and the usual scale, the broadcast core selects a CUDA
instantiation with shape constants. It also packs the preceding probability
fragment before QK's hardware fence, then issues its PV after the current
exponentials. The source reuses that fence for PV; the current partial-Q
build's final register schedule still makes ptxas insert16 additional fences
(C7519). These are present in the installed SASS. PV remains the last committed
WGMMA group, preserving the established wait/release schedule. TMA Q/K/V and
bias rings, three consumer warpgroups, and dynamic32/160 register budgets remain.
The L768 specialization now assigns K to producer warp1 and V to warp2.
Q remains on warp1 and bias on warp0. Separate K-free and V-free barriers let
the two producers advance independently, while each full barrier still requires
both transactions before consumers read the stage.

The current core also fuses PV and its row sums into an N40 WGMMA. The first
32 output columns contain P*V; the last eight contain P*ones. V retains its
original global strides, SW64 shared layout, and TMA transactions. A shared
ones tile replaces the old512B tile. Each V descriptor points its next
N32 group at that constant tile, with the leading offset corrected for the
selected ring stage and key chunk. The latest version reuses a2KB ones tile
across chunks, reducing this allocation from8KB. No global V repacking or extra preparation kernel is used.
The arithmetic for both output and denominator remains bitwise equivalent.

The L768 hot path now uses constant K/V descriptor offsets. Its old per-period
`params.zero * p` offset was always zero; expressing that invariant directly
lets the compiler reuse descriptors. This changes no addresses, arithmetic,
barrier protocol, or preparation kernel. Other flags retain the original form.

The latest L768 specialization retains three of four Q fragments in twelve
packed registers: both K16 slices for the first query half, and the first
slice for the second half. The first half's QK performs RS(k0)/RS(k1); the
second performs RS(k0)/SS(k1). The reduction order is unchanged. Groups of
four in-place FP32 FMA/EX2 operations keep this within the160-register consumer
budget. The mathematical operations and probability rounding are unchanged.

The preceding976 artifact's source-counter capture, job14536, attributes16581 of
16791 MIO stall samples to MUFU.EX2 and reports zero excessive shared-memory
wavefronts. This supports further work on instruction overlap; the sample
share is not a fraction of kernel wall time. See
`core_sol90/installed-qsmall-source.*` and `qsmall-stall-comparison.json`.

The native binding selects hot1073741824 for the qualifying L768 shape/scale,
and hot536870912 for L1024. Other shapes/scales keep hot0. SAFE uses1024. No expected
test vectors, probability rounding, or floating-point operation order changed.

The artifact also corrects a pre-existing SAFE reset barrier defect exposed by
large fix lists. The old code passed `FirstUserBarrier+4` to CUTLASS's user API,
which adds the reserved offset itself, yielding effectiveID20 rather than an ID
in0..15. It also used aligned synchronization from distinct producer/consumer
branches. The replacement passes user-relativeID4 (effective12) and calls
`arrive_and_wait_unaligned()` at both rendezvous. These are the PTX barrier
contracts described in [NVIDIA's reference](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier).
Consumer named barriers also now use user-relative IDs0..3, mapping to hardware
IDs8..11. Their old base8 was being offset a second time by the same API.

This correction is in the broadcast core package. The separate generic M1
binary has not been rebuilt in this follow-up.

## Measurements and validation

Jobs14295/14298 alternated two captured graphs in16 paired rounds. Both core
and block outputs were bitwise equal. Compared with the preceding installed
broadcast core:

| Direction | Previous core us | Candidate core us | Median paired reduction | Block paired reduction |
|---|---:|---:|---:|---:|
| starting | 901.88 | 888.18 | 1.55% | 0.96% |
| ending | 903.99 | 889.49 | 1.59% | 1.00% |

Prototype NCU job14297 measured832.576us, SM58.265%, tensor33.208%, active-warps
occupancy21.281%. These are separate profiling measurements, not graph-event
times. The installed artifact's final profile and serving verification are in
`core_sol90/installed-fast-profile.*` and `core_sol90/serving-fast-*.json`.
The first installed NCU job14313 measured830.176us and SM58.447%.
The subsequent independent-producer change was measured against that version
in16 alternating rounds, with bitwise-equal core and block outputs:

| Direction | Previous core us | Current core us | Median paired reduction | Block paired reduction |
|---|---:|---:|---:|---:|
| starting, job14324 | 897.22 | 884.41 | 1.40% | 0.99% |
| ending, job14330 | 904.07 | 884.93 | 1.50% | 1.13% |

The preceding independent-producer NCU job14334 measured815.648us, SM59.220%,
tensor33.752%, occupancy22.299%, DRAM607.608MB and L24.955GB.
This profile is `core_sol90/installed-producer-profile.*`. Do not substitute
the separate graph-event measurements for the kernel's NCU time.

- Jobs14305/14306:60 bitwise cases across full N768/B2/H4 inputs and smaller
  N7/B2 inputs at L768/1024, both contiguous and strided layouts. Cases include
  ragged/prefix/empty masks, one live key, late seeding, large logits and forced SAFE.
- Job14302 reproduced SAFE's old reset failure on the previously installed
  binary at full shape. Earlier smallN sanitizer checks did not exercise enough
  SAFE entries per CTA to expose this defect.
- Job14308: full N768/B1/H4, strided inputs, five mask/SAFE patterns; racecheck
  reports0 hazards, synccheck0 errors and memcheck0 errors after the correction.
- Jobs14309/14310: the final module-name build passes20 full-shape bitwise
  cases and2 nondefault-scale cases, covering the guarded fallback to hot0.
- Job14312 verifies the actual installed default and opt-out on five mask
  patterns, both directions and L768/1024, with bitwise core/block equality.
  It confirms hot1073741824 at L768, hot0 at L1024, and no flash fallback.
  All17 shipped package vectors pass without regenerating expected outputs.
- Independent producers and corrected consumer IDs: job14325 passes20
  full-shape bitwise cases. Job14326 passes full-shape racecheck with0 hazards,
  synccheck with0 errors and memcheck with0 errors, five patterns each.
- Final namespaced producer package: jobs14329/14331/14332 pass20 full-shape,
  40 generic-layout and2 nondefault-scale cases.
- Installed verification14333 confirms default/opt-out bitwise core/block
  equality across five masks, both directions and L768/1024, expected hot flags,
  no flash fallback, and17/17 shipped package tests. Results are
  `core_sol90/serving-producer-*.json`.

The N40 descriptor change adds the following improvement over the independent
producer version. Each job used16 alternating graph rounds with bitwise-equal
core and block outputs. Percentage reductions use paired ratios, which can
differ from the ratio of the two timing medians.

| Length / direction | Previous core us | Current core us | Paired core reduction | Paired block reduction |
|---|---:|---:|---:|---:|
| 768 starting, job14351 | 890.84 | 881.74 | 1.02% | 0.72% |
| 768 ending, job14353 | 889.12 | 880.17 | 1.02% | 0.70% |
| 1024 starting, job14358 | 2003.67 | 1957.15 | 1.27% | 1.15% |
| 1024 ending, job14360 | 1992.31 | 1952.52 | 1.57% | 0.74% |

**Previous N40 installed NCU job14369:811.648us, SM59.943772%, tensor34.164844%,
occupancy22.310606%, issue38.283036%, DRAM607.598MB and L24.960GB.
SOL90 remains unachieved.** This is `core_sol90/installed-f40-profile.*`.
The separate prototype NCU job14352 measured805.728us and SM59.842244%; use
the latest installed profile below for current status and the paired jobs for incremental
speedup, rather than comparing isolated measurements from different jobs.

- N40 prototype jobs14354/14357 pass20 full-shape and40 generic bitwise cases.
- Job14356, full N768/B1 strided: racecheck0 hazards, synccheck0 errors,
  memcheck0 errors; five patterns each including forced SAFE.
- Final namespaced package jobs14363/14364/14365 pass20 full-shape,
  40 generic and2 nondefault-scale bitwise cases.
- Installed verification14368 confirms bitwise default/opt-out core and block
  outputs across five masks, both directions and L768/1024, expected hot flags,
  no flash fallback, and17/17 shipped vectors without regenerating expectations.
  Results are `core_sol90/serving-f40-*.json`.
- QK-only, EX2-only and output-PV-only ablations are deliberately incorrect
  diagnostics. Higher utilization in those runs is not an attention result.
- Merging QK and PV completion groups adds no measured benefit: job14366's
  paired core ratio is1.000086, and combining it with N40 in14367 matches
  descriptor-only fusion within noise. The installed code retains two commits.

The final descriptor-constant candidate was compared against the N40 package
in16 alternating graph rounds. Core and block outputs were bitwise equal:

| Length / direction | Previous core us | Candidate core us | Paired core reduction | Paired block reduction |
|---|---:|---:|---:|---:|
| 768 starting, job14384 | 867.83 | 857.45 | 1.06% | 0.97% |
| 768 ending, job14385 | 867.20 | 858.54 | 1.01% | 0.77% |
| 1024 starting, job14391 | 1935.92 | 1935.18 | -0.07% | -0.28% |

L1024 is unchanged within measurement noise; its hot flag0 source is unaffected.
**Previous installed NCU job14398:804.416us, SM60.630086%, tensor34.556008%,
occupancy22.349430%, issue37.601392%, DRAM607.963MB and L24.975GB.
SOL90 remains unachieved.** These are `core_sol90/installed-fixeddesc-profile.*`.
Prototype NCU14388 was795.136us/SM60.470721%; it is not the installed measurement.

- Prototype14386/14387:20 full-shape and40 generic bitwise cases.
- Final namespace package14389:20 full-shape,40 generic and2 nondefault-scale
  bitwise cases.
- Full N768/B1 strided sanitizer14390: racecheck0 hazards, synccheck0 errors,
  memcheck0 errors; five patterns each including forced SAFE.
- Installed serving verification14397 completed: both orientations, L768/1024,
  bitwise core/block checks and17/17 shipped tests PASS. Its outputs are
  `core_sol90/serving-fixeddesc-*.json`.
- Caching one M64 Q half plus in-place exponentials also passed initial bitwise
  checks, but its extra improvement over constant descriptors was uncertain.
  Same-process three-way comparison14396 found a0.23% median core difference;
  a paired-bootstrap95% interval crossed parity. That composite is not installed.

The K16 Q slice and smaller ones tile were measured separately, then combined
against the preceding6d52 package. Job14518 used16 balanced paired rounds in
each direction; captured core and block outputs were bitwise equal:

| Length / direction | Previous core us | Candidate core us | Paired core reduction | Paired block reduction |
|---|---:|---:|---:|---:|
| 768 starting | 874.35 | 866.18 | 0.92% | 0.52% |
| 768 ending | 873.30 | 865.10 | 0.94% | 0.64% |
| 1024 starting, job14521 | 1976.07 | 1955.20 | 0.42% | 0.48% |
| 1024 ending, job14521 | 1973.11 | 1950.35 | 0.33% | 0.56% |

L1024 rounds showed substantial clock variation; these paired results support
nonregression, not a precise large gain. For L768, paired-bootstrap95% core
ratio intervals are[0.99028,0.99109] starting and[0.98962,0.99080] ending.

**Previous installed NCU job14525:793.536us, SM61.041971%, tensor34.790760%,
occupancy22.264219%, issue37.997760%, DRAM607.537MB and L24.952GB.
SOL90 remains unachieved.** These are `core_sol90/installed-qsmall-profile.*`.
Shared LSU wavefront utilization is61.940%, down from the preceding68.993%.
Use the paired measurements above for incremental speedup; separate NCU runs
have different clock and profiling conditions.

- Final namespace job14520:20 full-shape,40 generic and2 nondefault-scale
  bitwise cases; full N768/B1 strided racecheck0 hazards, synccheck0 errors,
  memcheck0 errors, five cases each.
- Installed verification14524: both orientations and L768/1024, five mask
  patterns, bitwise default/opt-out core/block outputs, expected native hot
  flags, no flash fallback, and17/17 shipped tests. Outputs are
  `core_sol90/serving-qsmall-*.json`. Expected vectors were not regenerated.
- Spatial polynomial, shuffle-table exp2 and centered-V FP16-PV experiments
  were slower and remain isolated. None is part of the installed package.

Follow-up through job14685 added no qualified speedup. Three-score/full-Q
ordering removed serialization but took849 versus798us. A new M64 two-score,
two-P, N40/full-Q kernel passed a runtime two-CTA occupancy assertion and the
initial FP64 check, but took990 versus790us. Raw BF16 bias transport halved
bias TMA/LDS bytes and preserved initial full-output bitwise equality, yet
took869–879 versus793–797us after adding reconstruction work. Early P packing
still failed the compiler resource screen. Detailed designs, initial-check
limits and rejected results are in [core_sol90/STATUS.md](core_sol90/STATUS.md).
The installed binary and its existing qualification remain unchanged;
**SM61.042% was the installed result at that point, not SOL90.**

Further follow-up through job14777 tested shared probability storage and
cooperative CTA pipelines. All nine measured candidates passed the initial
full-output bitwise and FP64 row0 checks, but were slower. These are initial
screens, not sanitizer or broad numerical qualifications:

| Candidate | Hot us | Same-job installed us |
|---|---:|---:|
| Shared P, full Q, STSM stores | 1055.39 | 799.37 |
| Half shared P, full Q | 882.55 | 794.21 |
| Four cooperative WGs, two scores, paired P | 1241.42 | 799.32 |
| Two resident CTAs, paired P | 1106.45 | 793.14 |
| M64, four scores, shared P, deeper K/V ring | 1410.26 | 793.47 |
| M64, four scores, register P, two resident CTAs | 1160.64 | 797.62 |

The paired pipeline needed uniform WGMMA group counts from its first body:
an initial zero-P group removed C7514 serialization. It then compiled without
spills, but its two-score schedule lost QK/exponential overlap. M64 four-score
variants restored that overlap and passed actual two-CTA residency checks;
their extra control/transport costs still outweighed the savings. Shared P
made full-Q register allocation possible without improving latency. These
families are rejected; do not keep tuning their register budgets without
a new mechanism. Exact results, including all nine controls, are collected
in [shared-probability-results.json](core_sol90/shared-probability-results.json).
No new candidate was installed in those experiments. **SM SOL then remained61.042%.**

Follow-up through14802 also added no qualified speedup. The HFMA2 cubic and
quartic exp2 model passed the initial numerical limit on four pair rows in
both directions at Q/K strengths1/2/4 (maximum RMS ratio1.01174), but its
register microprobe took96–97us versus37us for native EX2. This is model and
microprobe evidence, not an attention timing. Three explicit P-ordering
dependencies still caused WGMMA serialization and were rejected before GPU.
A four-consumer/single-producer-warp CTA also failed the resource gate:
544*120 ignores the per-CTA register check's warp rounding; ptxas capped it
at96 registers and spilled heavily. Numerical/cost results and precise
qualification limits are in [half2-and-fence-results.json](core_sol90/half2-and-fence-results.json).
Installed976 and its existing qualification remained unchanged through14802.

## Three Q fragments: installed in job14836

The extra K16 slice saves half of the remaining shared-Q reads. Static hot
QK instructions change from20 RS/20 SS to30 RS/10 SS, with the same39
WARPGROUP.ARRIVE and288 EX2 instructions. The final hot kernel has3464 static
instructions, down from3472. It retains128 initial registers, zero spills,
and no WGMMA serialization warning; the existing16 C7519 inserted fences remain.

Job14833 interleaved serving976 and the two possible three-fragment choices
in72 balanced rounds,8 subrounds and2 graph replays per sample. The selected
first-half-full cache gives these paired results:

| Direction | Previous core us | Current core us | Paired core reduction | Paired block reduction |
|---|---:|---:|---:|---:|
| starting | 884.884 | 880.725 | 0.458% | 0.251% |
| ending | 891.563 | 888.067 | 0.453% | 0.277% |

Bootstrap95% core-ratio intervals are[0.994853,0.995662] and
[0.995120,0.995723]. The ordinary16-pair harness agrees in both directions.
Job14834 repeats the balanced comparison atL1024: core ratios0.999996/1.000018,
block1.000344/1.000234. These meet the0.5% nonregression criterion; no L1024
speedup is claimed. These graph-event times are distinct from NCU kernel time.

The timing harness audit found that captured block outputs could share an
output buffer. Both pair.py and q_three_interleaved.py now replay and clone
each result before another graph can overwrite it. Jobs14833/14834 use these
independent snapshots and pass full core/block bitwise equality in both
directions and lengths. Earlier timing samples remain usable, but their old
captured Tensor references alone do not establish correctness. The initial
bench.py comparisons already cloned their outputs.

Final package14832 passed20 full-shape,40 generic-layout and2 custom-scale
bitwise cases. Full N768/B1 strided racecheck, synccheck and memcheck each
passed five patterns: dense, prefix, empty batch, late seed and forced SAFE;
zero hazards/errors. Machine-word comparison proves the final namespaced hot
binary exactly matches the measured prototype. Generic hot0 and SAFE1024
machine words exactly match the preceding installed976 binary.

Job14836 installed the package and verified the actual default and opt-out
paths: both directions, L768/1024, five masks, full bitwise core/block equality,
expected native hot flags and no flash fallback. All17 shipped package tests
pass; expected vectors and the separate generic M1 binary were not regenerated.

**Previous installed NCU job14836:784.896us, SM61.347178%, tensor34.964713%,
occupancy22.276943%, DRAM607.538944MB and L24.975438080GB.
Shared LSU wavefront utilization is58.470807%. SOL90 remains unachieved.**
Use the paired measurements for the incremental gain, rather than the ratio
between this and an older isolated NCU run. Artifacts:
`core_sol90/installed-qthree-profile.*`, `serving-qthree-*.json`, and
[q-three-results.json](core_sol90/q-three-results.json).

A subsequent producer lifetime/K-V loop specialization with parity polling
passed the initial full-output bitwise check but took806.194 versus795.444us.
Its mixed24/168/160/160 register variant still hadC7512 and hot spills, so was
rejected before GPU. Both remain isolated;
[lean-producer-results.json](core_sol90/lean-producer-results.json) records them.

In parallel, full-Q with a scalar denominator spilled. Full-Q with two SM80
denominator MMAs per warp compiled clean and passed initial full-output/FP64
checks, but took831.180 versus791.807us. Splitting those two MMAs around E
took833.126 versus791.532us. Both were rejected before broader qualification;
they are absent from the installed package.

Installed SHA256:
`9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225`.
The immediately preceding package is saved in `core_sol90/before_q1024_install/`;
older packages remain in their earlier backup directories. The installer
replaces files atomically, preserving existing processes' mapped library
inodes. `codex_core_sol90.patch` includes the cumulative changes relative to
`before_fast_install/`.

## Rebuild

The installed package is the current source of truth; older `core_tiles/`
prototype installers predate this follow-up.

```bash
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/oc/opt_core/kernels/triattn_core_broadcast/build_native.py
bash runs/anthropic_adoption_20260919/env.sh python runs/triattn_20260921/core_pipeline/refresh_core_record.py
sbatch runs/triattn_20260921/core_sol90/verify_fast.sbatch
sbatch runs/triattn_20260921/core_sol90/profile_installed.sbatch
```

Detailed rejected experiments and ongoing work are in `core_sol90/STATUS.md`.
Do not install a candidate merely because its SM issue percentage is higher:
it must improve actual execution time and pass accuracy and synchronization.
