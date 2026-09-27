# Query reuse and inference core follow-up

The best candidate is native CUDA `qfull4v2` at L1024. L384 retains
`h4kv_local1`; L768 retains `hot6t`. All paths use `front8` and the already
qualified `outproj_c1`. This is an explicit inference experiment. Production
dispatch and training are unchanged; four-head attention CTA work stays closed.

L1024 qualification19134 and actual-default verification19157 pass. The
experimental `candidate.load()` selects this change only at L1024.

## Actual-default complete inference FWD

Job19157, node02/normal_h100, B1/C128/H4/D32 BF16. Includes LN, bias, all
projections, attention, gate and input-preserving residual. Both directions
use64 AB/BA rounds x40 CUDA Graph replays. Original Anthropic's faster of
update-plus-add and copy-plus-fused-residual forms is selected per cell;
copy-plus-fused-residual wins all six here. Times are individual medians;
reductions use the median paired ratio.

| L | Direction | Anthropic ms | Ours ms | Less time vs Anthropic | Additional reduction vs18962 |
|---:|---|---:|---:|---:|---:|
| 384 | starting | 0.4609 | 0.3140 | 32.00% | unchanged |
| 384 | ending | 0.4671 | 0.3167 | 32.25% | unchanged |
| 768 | starting | 1.7435 | 1.5956 | 8.47% | unchanged |
| 768 | ending | 1.7219 | 1.6015 | 6.73% | unchanged |
| 1024 | starting | 3.5169 | 3.2546 | 7.43% | 2.22% |
| 1024 | ending | 3.4747 | 3.2693 | 5.87% | 2.23% |

The L1024 paired95% time-reduction intervals versus18962 are
1.92–2.36% starting and2.10–2.33% ending. An independent qualification run19134
measures2.62%/2.26% reductions. L384/L768 binaries and routes stay unchanged;
fresh absolute times are not additional gains at those lengths.

Use `load()` for the current inference experiment. Reproduce the prior
L1024 checkpoint with `load('hot4t', out_artifact='outproj_c1')`.
`load(out_artifact=None)` disables output fusion with the current cores;
explicit core names without `out_artifact` retain their historical cuBLAS tail.

## Selected algorithm

Each single-head CTA retains row-local K/V in shared memory and uses four
query warpgroups, with two TMA bias slots per group. For each 64x32 query
tile, two `ldmatrix.x4` instructions load Q into eight packed 32-bit registers
per thread before the key loop. Both K16 slices of every QK operation use
RS WGMMA, reusing Q registers across all key tiles. K remains in shared
memory. Accumulation order, BF16 probabilities, P*V/P*1, gated-output
rounding and the stable on-chip retry remain the same as `hot4t`.

The L1024 specialization uses 128 registers, zero stack/spills and 231424B
dynamic plus 1024B static shared memory. It has no compiler serialization
warning. The compiled L384 six-warpgroup specialization does spill and warns;
it is not selected. Q/K/V/gate/LSE still have no global save buffers at L1024.
This change reduces repeated shared-memory Q reads; it does not eliminate a
new HBM tensor or introduce a dedicated producer warpgroup.

Job19132 directly measures the native QKV/attention/gate boundary against
`hot4t`: 2740.27 to 2639.20 us, paired ratio 1.03813 (3.67% less time).
Both candidates use the same fused output boundary in complete-FWD tests.
Job19111 finds 3.01%/2.60% less complete-FWD time at L1024 starting/ending.
The final actual-default measurement below is the selection authority.

At L768, job19137 compares with the actual old `hot6t` core, using 64 AB/BA
rounds of 40 CUDA Graph replays. The reductions are only 0.43%/0.26%; ending's
95% bootstrap interval includes no improvement. Both outputs and changed
graphs are bitwise equal, but the gain is inconclusive, so `hot6t` stays.
Qualification19135's separate comparison to `hot4t` is a source-algorithm
control, not a gain over the old L768 selection.

## Sixteen controls

Complete inference FWD, 24 AB/BA rounds x30 graph replays on node02,
normal_h100. Every cell compares against the prior selected core for that
length, and both sides use `outproj_c1`. Values are paired time reductions
in percent, starting/ending; negative means slower. These are numerical and
graph pilots, not a claim that all sixteen received full sanitizer qualification.

| Candidate | L384 | L768 | L1024 |
|---|---:|---:|---:|
| pipeend32_c4 | -29.08/-29.73 | -24.33/-23.18 | -24.09/-24.41 |
| pipeend64_c2 | -37.79/-38.36 | -44.68/-44.03 | -41.38/-41.06 |
| pipeend64_c4 | -23.26/-23.51 | -16.23/-15.41 | -14.54/-14.49 |
| n40_hot4t | -9.72/-10.10 | -2.70/-1.94 | -0.27/-0.53 |
| n40_hot6t | -9.88/-10.26 | -0.98/-1.33 | -1.76/-1.92 |
| n40_qhalf4 | -13.35/-13.49 | -0.85/+0.73 | +1.01/+0.99 |
| qhalf4 | -13.95/-14.04 | -0.75/-0.83 | +1.55/+1.27 |
| n40_qhalf6 | -13.27/-13.36 | -2.63/-2.95 | -5.61/-5.70 |
| pack4 | -7.96/-8.14 | -1.80/-3.15 | -0.40/-0.28 |
| pack6 | -8.03/-8.44 | -2.97/-3.06 | -5.18/-5.90 |
| n40_pack4 | -7.71/-7.91 | -2.10/-0.54 | -0.14/+0.42 |
| n40_pack6 | -7.59/-7.99 | -0.79/-0.92 | -4.04/-3.64 |
| split6q | -8.35/-8.88 | -2.30/-2.31 | -1.44/-1.69 |
| split4q | -8.53/-8.78 | -3.22/-3.26 | -0.34/-0.43 |
| qfull4v2 | -19.66/-19.26 | +0.67/+0.52 | +3.01/+2.60 |
| qhalf6 | -13.98/-14.39 | -3.52/-3.66 | -7.24/-6.93 |

The end-wait pipeline fixes a real compiler problem: loop-backedge accumulator
copies had forced WGMMA serialization. Completion before the backedge removes
those warnings; SASS now issues next-QK HGMMA, current EX2, then completion.
Despite this overlap, its tile/lifetime/register costs make complete FWD slower.
N40 combines P*V and P*1 without a global V buffer, but offers no further gain
over the best query reuse variant. Streamed probability packing and separate
stable retry also lose. Job19132 compares the finalists directly: `qfull4v2`
beats `qhalf4` by another 1.24–1.33%, while adding N40 to `qhalf4` regresses.
See [design and rejected build prototypes](CORE_FOLLOWUP_DESIGN.md) and
[binary scheduling evidence](endwait-codegen-evidence.json).

## Diagnostic profile

NCU19136, L1024 native boundary, five warmups, cache-control none and
clock-control none. This is separate from the paired selection timing.

| Metric | hot4t | qfull4v2 |
|---|---:|---:|
| Kernel ms | 2.6547 | 2.5873 |
| SM throughput % | 49.26 | 49.59 |
| DRAM throughput % | 7.08 | 7.17 |
| Tensor pipe activity % | 28.27 | 29.56 |
| XU instruction throughput % | 45.13 | 47.20 |
| DRAM read MB | 358.63 | 351.16 |
| DRAM write MB | 271.67 | 270.88 |

The remaining core does not saturate HBM bandwidth. The small traffic change
is not an eliminated intermediate. Better Q reuse raises tensor/XU activity,
but softmax/scalar work and dependent waits remain. SM throughput around50%
is not SOL90 and should not be compared directly with earlier training-only
SOL measurements. NCU emitted the known Python-site export warning; both
reports/CSVs and the Slurm job completed successfully.

## Qualification and scope

`qfull4v2` native outputs are bitwise equal to the source `hot4t` algorithm in
eight fixtures at each L64/128/384/768/1024, including partial/full masks,
late keys, large logits and positive/negative bias offsets. Changed-input,
weight and bias CUDA Graph replays also remain bitwise equal. Independent
FP64 checks and stable-retry counter audits pass. All three sanitizers pass
at L128/L1024 in19134 and L768 in19135, each covering mixed masks, all-masked
inputs and large logits. The former log has12 zero-error mem/sync summaries
and6 zero-hazard race summaries; the latter has6 and3. Full-module checks at
L768/L1024 cover both directions:10 correctness fixtures,6 independent FP64
samples,2 changed-input/weight/mask graph checks,2 fullgraph checks and2
no_grad checks per length. Qualified intermediate variants `qhalf4`19109
and `n40_qhalf4`19098 pass the same L128/L1024 gates but are not selected.

All-masked semantics still differ from original Anthropic: ours produces a
zero attention update, whereas Anthropic produces mean-V attention. This is
not an unrestricted drop-in parity claim. Inference only; no backward saves
or gradient qualification. All48 training18246 source/binary files match
their snapshot. SOL90 remains unmet.

Machine-readable evidence: [core-followup-results.json](core-followup-results.json).
Actual-default19157 repeats all module checks at384/768/1024 and records
the actual candidate kernel names plus original Anthropic source/build
attribution. All owned experiment jobs are terminal, including the documented
failed build and cancelled prototypes; none remain queued or running.
Rebuild the summary with `summarize_core_followup.py --selected-job 19157`.
Individual JSON records retain raw paired ratios; summarized confidence
intervals use5000 bootstrap resamples.
