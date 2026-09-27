# Core phase timing and N16 CUDA candidates

The installed package remains SHA256 `9365a4f35e018afed4adba21590ae2c1734bbccf98d1e53904ebfe2537b3e225`.
Its last qualified NCU measurements remain SM61.349% at L768 and SM65.249%
at L1024. SOL90 is not achieved. No candidate in the results below is installed.

## Coarse phase trace, job 15063

The diagnostic kernel records 17 clock64 timestamps for three sampled CTAs
and their three consumer warpgroup leaders. Differences are taken within
each CTA, with matching SMIDs and monotonic timestamps checked. Instrumented
core/block outputs are fully bitwise equal before and after 24 balanced timing
rounds in both directions. Core overhead is 5.107% starting and 4.884% ending.

Later sampled CTAs spend a median 72.9–74.0% of their instrumented cycles in
the three stream bodies, 17.5–18.6% in startup, 2.8–2.9% in drains, and
5.8–6.0% in the final PV/epilogue. Register allocation alone is about 0.8%.
These separately computed medians need not sum exactly to 100%.
The first sampled CTA has a larger startup fraction, about 25%.

The measured overhead and limited sampling prevent treating these as exact
unmodified-kernel fractions. They support focusing on the stream body;
register allocation or drain removal alone has limited scope.
[Aggregated trace](core_sol90/current-phase-trace-summary.json) links back
to the two complete graph comparisons and raw phase samples.

## Four-score, two-P N16 stream

The prototypes retain native BF16, full Q in registers, fused N40 PV/denominator,
first32-key seeds, and finite-numerator/denominator SAFE checks. Independent
per-row K/V ready and free barriers allow separate TMA producer warps. Bias
uses paired ready barriers and releases from every owning consumer warp.
The stream overlaps QK(k+2), E(k), PV(k−1), and bias(k+3). The final body uses
wait1 because no phantom future QK is submitted.

The scope is B1, square L768, contiguous operands, standard scale, row-broadcast
mask. These are standalone prototypes, not serving-package replacements.

| Candidate | Emitted producer / consumers | Compiler result | Initial measured result |
|---|---|---|---|
| M64/R5 | 32 / 5×88 | C7512 and spills | No GPU launch |
| M128/R4 | 32 / 4×112 | C7512 and spills | No GPU launch |
| M64/R4 | 32 / 4×112 | No spills or serialization | Job15075, hot914.397 vs789.325µs |
| M128/R3 | 32 / 3×160 | No spills or serialization | Job15076, hot853.927 vs795.098µs |

Jobs15075/15076 complete successfully and initially match the entire core and
block outputs bitwise. Sampled first-row FP64 RMS equals the baseline,
0.00028883485479432215. Median core times over three initial rounds are
1042.672 vs886.133µs and977.723 vs873.383µs respectively. Both are rejected
on performance; broader numerical and sanitizer qualification was not warranted.

The new [N16 admission script](core_sol90/screen_n16_warp_latency.py) inspects
actual hot and SAFE SASS role allocations, multiplicities, initial pools,
spills, and serialization. M64/R4 has 61440 registers available/required;
M128/R3 has65536. This extends the admission approach to 640/768-thread CTAs
without reusing the older gate's fixed 512-thread assumption.

## Fence and register follow-ups

Removing source PV fences in the four-score streams changes no hot machine
words. ptxas injects44 fences for M64 and92 for M128; the variants retain
9888 and12592 words, identical to their measured controls. Duplicate GPU
benchmarks are skipped. [Equivalence evidence](core_sol90/n16-prefenced-equivalence.json).

The M128/R4 two-score stream retains four scores only during first32-key
bootstrap, then uses two live score and two P buffers. It issues PV(k) before
QK(k+2); wait1 leaves QK(k+1) outstanding across E(k). Current score and old P
users retire before reuse, and V is released only after its previous readers.
The first build eliminates serialization but still spills104 stack bytes,
220 store bytes and296 load bytes, so it is not launched. One K/V descriptor
base with constant stage/column offsets removes all hot/SAFE spills and
serialization. The resulting `m128n16r4s2p2qrf40fd` passes initial full core/block
bitwise equality in job15094, but is slower: hot859.529 vs787.573µs and
core990.237 vs881.567µs. It is rejected on performance.

NCU job15097 measures that candidate at864.480µs, SM56.017%, tensor31.346%,
and28.238% active-warps occupancy. Occupancy exceeds the installed kernel's
22.296%, while execution time and SM throughput are worse. Its L2 traffic is
4.564GB. Increasing consumers and lowering memory traffic did not produce a
faster kernel. These are separate profiling measurements, not paired timing.

The two four-score/fixed-descriptor controls with12 or8 cached Q registers
have now completed their CPU builds. Both fail with C7512 serialization;
Q3 additionally has16B stack/16B spill stores/768B spill loads. Q2 has no
spills but still serializes. Their actual role budgets are checked and valid.
Neither is launched on GPU. No further GPU experiments were submitted during
the user's status review or kernel-inventory request.

All source, binaries, compiler logs and initial timings are under
[core_sol90](core_sol90/). Installed manifest hashes are checked independently;
expected test vectors are not regenerated.
