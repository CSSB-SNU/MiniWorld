# Backward SOL90 continuation — 2026-09-23

> **Current backward installation, 2026-09-25:** [Installed report](core90/README.md). Qualified `rs8_async_q4` dK/dV + bias and `rs_softmax_overlap` dQ pass installed-path job17211, including all input/parameter gradients. Complete backward improves21.92–23.23% and F+B14.63–16.00% versus job16663. Attribution17211 accounts for every16/17 backward kernels. SOL90 remains unmet. All new experiments use `node02 / normal_h100`.

The latest user direction targets dK/dV, dQ, then weight gradients after full
backward attribution. [priority_pass/README.md](priority_pass/README.md) was the
previous installed result, job16663: `ldmatrix_bias` dK/dV plus native CUDA/TMA
shared-input Q/K/V/gate weight gradients. It removes another4.09–5.30% of whole
backward time versus installation16580; F+B improves1.38–3.60%. L768 four-weight
gradient HBM traffic drops44.53%; its main kernel reaches HBM SOL85.27%. Full
backward has16/17 kernels rather than22/23. dQ experiments were slower, so the
qualified `vector_bias` dQ is retained. [Current attribution](priority_pass/ATTRIBUTION.md)
is job16664. dK/dV and dQ remain below80; no whole-backward SOL80/90 claim.

[below80/README.md](below80/README.md) preserves the preceding installation16580:
vector_bias dK/dV and dQ plus warp_packed projection/LN. Projection/LN reaches
HBM SOL84.94%/82.84% starting/ending; gate and final bias reduction also exceed80.
Those stages stay unchanged under the user's80% cutoff. Forward remains closed.

The remaining sections preserve the earlier SOL90 search and its named
artifacts. Its baseline is preserved q32_four + original LN/residual + old dQ +
old gate, not the immediate baseline in below80/README.md. Do not add the two
reports' percentages or treat the following job16333 table as the latest install.

## Historical installed combined result (job16333)

B1 BF16 C128/H4/D32, H10080GB, all input/parameter gradients, masked keys,
12 balanced AB/BA rounds, 15 graph replays. Timing dropout0; correctness covers
mixed/all-masked, dropout.25, SGD updates, frozen parameters, fullgraph Inductor,
BF16 autocast, and FP16 dispatch refusal. Times are ms; reductions use median
paired ratios. No sum of independent optimization percentages.

| L | Direction | Bwd before | Bwd after | Reduction | F+B before | F+B after | Reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 384 | starting | 1.394 | 1.268 | 9.06% | 1.910 | 1.780 | 6.80% |
| 384 | ending | 1.464 | 1.331 | 8.99% | 2.080 | 1.949 | 6.33% |
| 768 | starting | 7.811 | 6.892 | 11.71% | 10.365 | 9.411 | 9.24% |
| 768 | ending | 8.098 | 7.117 | 11.99% | 11.022 | 10.044 | 8.77% |
| 1024 | starting | 16.681 | 14.667 | 12.08% | 22.026 | 20.131 | 8.39% |
| 1024 | ending | 17.147 | 15.169 | 11.46% | 23.155 | 21.291 | 7.91% |

Evidence: `sol90_installed/paired-16333-L*.json`, `guards-16333.json`,
`amp-correctness-16333.json`, `final-16333.log`. Ten opt-out/metadata checks,
fresh eager first backward, fresh compiled first backward and BF16 autocast pass.
All five fusion flags are enabled by default. Each adopted kernel passes memcheck,
racecheck and synccheck; qualification jobs16210,16233,16257,16259,16324.
The cold-context correction qualified16265 is retained in n64_staged;
fresh eager and first compiled backward both pass16333. Versioned binaries and atomic replacement
prevent truncating mapped extensions. Unsupported inputs retain existing dispatch.

## Definition and measured progress

Use NCU hardware-peak utilization per actual bottleneck, alongside full-module
backward and forward+backward time. Do not substitute a measured copy ceiling or
an average of unrelated utilization metrics for hardware SOL90.

- q32_solpack: cooperative four-WG dBias, reuse LSE/delta, packed BF16 shared
  stores. Corrected paired job16156 shows roughly4–5% additional module backward
  improvement over installed q32_four. NCU16161: fused SM53.21%, DRAM30.95%.
- q32_async: overlap independent score/dP and dV/dK WGMMA groups. Job16164: about
  5.05–5.39% at768,4.36–6.06% at1024 vs installed. Zero spills/serialization.
- q32_vec: float4 final bias reduction. Job16174: about5.88–6.45% at768,
  4.58–5.87% at1024. NCU16186 L768 reduce_bias: **DRAM92.637%**,0.592192ms;
  fused dK/dV SM53.38%, DRAM30.82%; dQ SM63.41%. Only reduction reaches SOL90.
- q32_combo combines async and vector reduction; fully qualified16196.
  q32_combo_small adds a more parallel L384 reduction; fully qualified16210, adopted.
- q16_twocta spills112 bytes and serializes WGMMA; rejected before GPU launch.
  q16_smemgrad moves three gradient tiles per consumer to shared memory. It fits
  112640B shared and compiles with no spills,80 initial registers,24/104 producer/
  consumer allocation (29696 registers per CTA <=30720 initial budget). Job16193: L768 core5.97->7.57ms, L102413.36->17.52ms. Rejected; remaining module timing canceled.

## LN + residual fusion

Native CUDA projection dgrad + LN backward + residual includes all LN parameter
gradients. Eliminates dZ and dXbranch HBM buffers; keeps rounding boundaries.
Initial scalar-residual, TMA-residual and TMA-all versions are numerically correct
but slow (L768 around2.05/1.84/1.73ms vs0.86ms before fusion). Not adopted.
NCU16167 found long-scoreboard stalls and very low issue utilization. SASS of
TMA-all contains generic LD.E and hundreds of address calculations; manual_shared
uses explicit shared-memory instructions, a rolled LN loop and hoisted ending
coordinate division. Job16195: L768 starting0.863->0.753ms, ending1.048->0.815ms.
Full-module/compile/optimizer and all three sanitizers pass16199. The main kernel
is retained in installed split_norm; the old parameter reducer is superseded.
NCU16225 L768 starting: fused720.928us, SM32.70%, DRAM44.33%; parameter
old reduction44.992us, DRAM6.27%. This reduction profile predates split_norm.
Fusion is faster but not SOL90.

## Evidence caveats

Job16149 imported two CUDA extensions with the same module name. Importlib
returned the same object for candidate and installed paths. Candidate-vs-stock
Triton correctness/timing remains valid; the purported installed-vs-candidate
comparison is INVALID. native.py now supports distinct module names; bench_sol.py
and paired_current.py assert different module objects. Job16156 and later use
unique names; CPU probe verified both file paths and object identities.

Job16165 failed to open a renamed diagnostic script. It has no GPU measurement;
job16166 is the valid replacement. Avoid script names profile.py or inspect.py,
which shadow Python standard-library modules.

BF16 packing follows NVIDIA PTX high/low operand ordering and local CUTLASS4.2
numeric_conversion.h. See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.

## Native dQ continuation

Query-owned Q64/K64/D32 CTA, resident Q/dO/stats, double-buffer TMA K/V/bias,
WGMMA score/dP and dQ accumulation, packed BF16 dS in shared memory. No global
P/dS buffers or dQ atomics. First256-thread version (2CTA/SM) essentially matches
stock timing, pilot16212. NCU16221 SM50.60%, DRAM25.92%, occupancy15.09%.
3CTA/SM variant uses80 initial registers, producer24/consumer128, zero spills.
Pilot16226 passes FP64/masks/repeatability; dQ alone speeds up about7% at384,
9.6% at768/1024. NCU16232 SM58.76%, DRAM29.39%, occupancy22.67%.
Full qualification16233 passes module/all gradients/optimizer/fullgraph and all
three sanitizers. Installed16247. `_fuse_dq_backward=False` retains the old dQ.
Four-CTA K64 spills144/84 bytes and serializes WGMMA; not launched. K32 four-CTA
pilot16239 is slower than three_cta; pipelined variant16241 also offers no clear
advantage. Both are unadopted. q32_reduce_overlap16231 is unadopted: only
0.1-0.4% bwd change and no F+B benefit.

Gate/output dgrad + delta native TMA fusion is installed from gate_delta/context_ready.
It avoids rereading O/dO for preprocessing (nominal0.302GB at768), preserving
BF16 dO rounding. See the completed gate/LN qualification below.

## Further unadopted refinements

LN register-only epilogue spills; register_recompute is spill-free and correct
but slower than manual_shared (pilot16248: L7680.847/0.840ms vs manual0.753/0.815).
A two-stage parameter reduction subsequently qualified as split_norm and is installed.
Gate/delta pilot16236 is correct but slow (L7681.431ms vs0.481ms). SASS shows64
LDL/66STL instructions from dynamically indexed eight-float local accumulators,
plus64 reciprocal calls. Approximate reciprocal16246 removes calls but stays slow.
Packed stores + constant array index16249 eliminates the local stack and gives
0.738ms, still slower than the baseline. TMA stores subsequently qualified and are installed; N64 regressed and was rejected.
These earlier slow gate variants remain unadopted.

## Installed hardware profile (job16251, L768)

One warmed core graph replay, all four kernels included; clocks/caches not fixed.
This profile predates gate fusion: the default full module now computes delta
inside gate/output backward, so there is no separate delta-preprocessing kernel.

| Kernel | Time ms | SM SOL | HBM SOL |
|---|---:|---:|---:|
| delta preprocessing (existing Triton) | 0.133 | 35.39% | 70.51% |
| grouped dK/dV + bias partial (CUDA) | 3.272 | 53.38% | 30.91% |
| final bias reduction (CUDA) | 0.595 | 4.17% | **92.20%** |
| dQ (CUDA, three_cta) | 1.405 | 58.70% | 29.76% |

Whole-core DRAM reads+writes: 6.943827GB. Native dQ is faster but increases its
DRAM traffic to1.401545GB; do not reuse the old q32_four whole-core6.3435GB figure
for this installed path. Compared with the previously measured stock whole-core
9.4255GB, traffic is about26.33% lower; these are separate warmed NCU runs, not a
controlled streaming comparison. The single-run paired module timing is the speed
evidence. Bias reduction alone achieves SOL90; dK/dV and dQ do not.

## Completed gate/LN installation and context correction

- split_norm qualifies16257 and16259 (the latter includes finish_norm in the
  sanitizer filter) and is installed. It adds a16KB second-stage parameter
  scratch, reducing the old ~45us low-occupancy norm reduction.
- gate tma_store qualifies16254: additional whole-module backward2-3%, including
  all gradients/dropout/optimizer/fullgraph; three sanitizers pass. NCU16255
  gate/delta318.048us, HBM83.849%, SM23.967%, read453.081MB/write440.940MB.
- Complete combined job16262 passes warmed module/compile/AMP and timings, then
  aborts in a separate fresh-process guard test. Reproduction16264 locates the
  failure at the FIRST gate backward in a new autograd worker, before any opt-out:
  cuTensorMapEncodeTiled returns201 (CUDA_ERROR_INVALID_CONTEXT). Device guard
  alone can leave a driver context unset on this thread.
- Corrected host entry context_ready explicitly calls cudaSetDevice after
  CUDAGuard. Cold eager and compiled tests pass16265; actual installed combined
  verification passes16266. GPU SASS is unchanged. Gate fusion is enabled by default.
- bias_transpose16258 and bulk_partial16260 regress whole-module time and are
  rejected. The latter adds16KB shared double buffers and bulk partial stores;
  this store substitution helps gate fusion but not grouped dK/dV.
- dQ query12816263 reduces input duplication but spills28 bytes; numerically
  correct, slower than the selected three_cta. Gate cooperative16261 is essentially
  tied with tma_store and not adopted.

## Continued HBM reduction experiments

- r2_twocta16267 passes initial numerical comparisons but regresses core time:
  L7685.313->6.063ms, L102412.047->13.935ms. Doubling the partial-buffer size
  outweighs the CTA change; the build also spills32 bytes. Rejected.
- r2_warp/r4_warp compact-producer builds spill88 bytes and are not launched.
- cluster8 (16284) and cluster8_partial (16285) combine two CTAs through DSM,
  keeping FP32 group partials with no new rounding. Partial scratch halves again
  and reduction time drops from~0.584ms to~0.298ms at768, but the grouped kernel
  slows substantially. Core time regresses38-43% and17-20%, respectively.
  All eight FP64/mask cases and repeated exact determinism pass. Not adopted.
- producer_reduce (16288) double-buffers dS and assigns the three unused producer
  warps to local bias reduction, removing the512-thread consumer barriers.
  Correctness passes, but producer TMA address calculations spill56 bytes;
  core time regresses substantially. Not adopted. The128-thread unrolled
  alternative spills64 bytes and is not launched. producer_rolled16293 eliminates
  spills but still regresses26-34%; this producer/reduction schedule is rejected.
- r8_two_consumer keeps four rows of dK/dV per WG to halve partial scratch within
  one CTA. It spills216/396 bytes (store/load), so it is not launched.
- r6_three_consumer16296 preserves FP32 and reduces scratch by about1/3, with
  three consumer WGs keeping two rows each. It is spill-free and passes all eight
  FP64/mask cases. Core timings are effectively tied/slightly worse (+0.1-0.7%).
  It is not installed. r6_overlap16299 is also spill-free/correct but offers no
  core improvement. Full module paired16300 retains all five default fusions
  and measures R6 versus current R4: backward changes range from0.38% slower
  to0.67% faster, varying by length/direction; L768 is only~0.14% faster in both
  directions. There is no consistent material speed gain to justify adoption.
- bf16_partial16292 passes random FP64/mask tests and makes the core4.1/4.8/6.7%
  faster at384/768/1024, but FAILS the zero-sum row-adjoint test16295: reference
  and stock are exactly zero; candidate RMS1.121e-4 exceeds the unchanged2e-7
  zero-reference allowance. The current FP32 candidate passes with exact zero
  in control16297. Additional BF16 rounding is rejected, not installed.
- lossless24 stores only24 bits when the omitted FP32 byte is exactly zero across
  a32-value segment, else stores full FP32. Scratch capacity remains FP32-sized
  plus1 byte per32 values; only transfer bytes can shrink. All four gradients
  match the current kernel bitwise in12 tests (including40-bit exponent spread),
  FP64/masks and the cancellation test pass16298. Encoding/decoding overhead
  nevertheless regresses core time8-16%. Not installed.

`bias_fusion/continued-fusion-pilots.json` records these diagnostic core timings.
They include native dQ and separate delta preprocessing, so they are not full
module measurements with gate fusion. All candidates have distinct binary names;
none of these pilots has changed the qualified installed package.

## Additional gate tile experiment

`gate_delta/tile128` doubles rows per CTA while retaining native TMA/WGMMA and
FP32 delta with rounded BF16 dO. It reuses gate/out shared tiles for dG/A after
their last reads; no extra global tensor is added. Initial pilot16304 catches
incorrect delta row mapping at M128; dO/dG/A match exactly but delta fails, so
that artifact is rejected. tile128_delta derives the head/row accumulator indices
from the constant MMA fragment coordinates. Corrected pilot16312 matches all
four outputs bitwise, but is22-26% slower, so it is not adopted.
`n64_tma` revisits half-width output tiles with TMA stores and adjacent CTA pairs
sharing dY through L2. The older n64 pilot used direct global stores and a
nonadjacent N-tile traversal; it does not measure this new schedule.

N64/TMA pilot16307 matches all four current outputs bitwise at every tested
length and passes FP64 checks. Direct gate-kernel paired reductions are5.69%,
1.59%,0.96% at384/768/1024. Full qualification16310 passes module gradients,
dropout, optimizer changes to every parameter, frozen parameters, fullgraph and
all three sanitizers. Complete-module timing changes are small/noisy (including
an implausibly large isolated backward gain at1024); do not advertise them as a
reliable1.8% module improvement. NCU16311:307.840us, HBM86.6769%, SM25.0492%,
reads453.077504MB/writes441.296128MB. Actual DRAM reads do not double despite
split output channels: adjacent CTAs reuse dY through cache. Resource limits
permit4CTAs/SM. Gate SOL90 remains unachieved.

`n64_staged` tries reusing the retired weight tile for gate/out, reducing shared
memory from50176 to33792 bytes with a cooperative128-thread CTA. It compiles
with80 registers and no spills/stack; NCU confirms6CTA resource limits. All four
outputs match the previous installed gate bitwise across seven shapes; FP64
checks pass. Pilot16321:8.60%,1.96%,1.21% direct kernel reductions at384/768/1024.
Full qualification16324 passes input/all parameter gradients, dropout/masks,
optimizer changes to every parameter, frozen parameters, fullgraph and all three
sanitizers. NCU16325:308.896us, HBM86.410%, SM24.506%, reads453.562624MB,
writes441.062144MB. Selected and installed; cold/default combined16333 passes
all ten opt-out/metadata cases, first compiled backward, BF16 autocast and FP16
native refusal. The five package source/binary/API manifests pass hash verification.
The four-CTA variant is essentially tied at large sizes; the staged variant has
better L384 latency. Full-module incremental changes are small and noisy; do not
promote the isolated~1.3% L1024 backward median to a robust module gain.

## Current disposition

The additional installed change in this continuation turn is n64_staged gate/delta.
Bias/core experiments remain isolated. The combined table compares with the
preserved q32_four start of the broader SOL90 campaign; it is not an incremental
claim for gate tuning alone. Gate-specific direct timing is the evidence for the
new kernel improvement. All owned GPU jobs and builds are terminal, including
intentionally rejected precision/indexing failures. Complete backward SOL90 is
still not attained: latest core measurements remain SM53.38% (dK/dV) and58.70%
(dQ), while selected gate HBM is86.41%. Only the final bias reducer exceeds90%.
