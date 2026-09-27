# Selected inference FWD: bottleneck and next algorithm

2026-09-26. The user closed the four-head **attention CTA** direction. Keep its results as
negative evidence; do not reopen it as the default search direction. The selected L384
joint four-head **KV projection** is a different operation and remains selected.

No kernel or dispatch change is made here. Fresh node02/H100 profiling18909 completed;
the current inference entry remains384 `h4kv_local1`,768 `hot6t`,1024 `hot4t`.

## Diagnosis

The long-length core is not saturating HBM. It has a mixture of non-matmul work,
dependent compute/memory stages, limited ready warps and expensive on-chip resource
lifetimes. Aggregate counters do not establish one exclusive causal bottleneck, but
they rule out treating every remaining operation as HBM-bandwidth-bound.

Fresh NCU18909, five warmups, cache/clock control none,20 passes per kernel. L384 has a
separate KV projector; the table's attention row includes Q/gate. At768/1024 it includes
all Q/K/V/gate projections. Times below are native kernel diagnostics, not full FWD.

| L / kernel | Time ms | DRAM GB | HBM TB/s | HBM peak % | L2 peak % | Tensor pipe % | XU pipe % | Occupancy % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|384 KV projection|0.0439|0.1108|2.523|75.3|69.9|22.8|0.0|17.7|
|384 attention/QG|0.1831|0.1548|0.845|25.2|70.5|23.8|41.2|24.4|
|768 fused core|1.2020|0.4216|0.351|10.5|46.1|27.9|43.2|37.5|
|1024 fused core|2.6433|0.6201|0.235|7.0|45.8|28.3|45.1|25.0|

Tensor columns use `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`;
XU uses `sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed`, which includes
exp and other special operations. These counters are neither additive time percentages
nor a claim that exp alone accounts for43–45% of latency. L2 sector read+write volume is
1.294GB for384 attention,5.504GB for768,11.247GB for1024, well above DRAM volume.

Previously qualified actual-selected full-module profiles18745, starting direction:

| L | LN+bias us | KV us | Attention/QG or QKV us | Output GEMM us | Residual us |
|---:|---:|---:|---:|---:|---:|
|384|53.2|42.5|187.1|22.1|41.3|
|768|197.3|included|1233.8|104.7|151.2|
|1024|333.7|included|2680.2|186.5|263.3|

The core is about73–77% of summed stage time at long lengths. These independently
synchronized stage medians need not sum exactly to paired graph FWD latency. Ending
direction is similar; all raw stage records remain in `head4-results.json`.

## Why D32 changes the priority

There are `4*L^3` score elements:1.812 billion at768 and4.295 billion at1024. Every
normal-path score still needs an exponential. QK plus PV use only `4*D=128` multiply/add
FLOPs per score whenD=32. The denominator's P*1 WGMMA adds16 FLOPs per score; projection
work has a different, quadratic dependence onL.

For perspective, the FA3 authors give H100 nominal rates of989TFLOP/s for FP16 matmul
and3.9T special operations/s. AtD32, the ideal exp time is roughly twice the QK+PV
matmul time, before counting other softmax/conversion work. This is a throughput model,
not a prediction at the measured unlocked clocks. It explains why another reduction
in Tensor Core arithmetic or switching matmul precision alone need not help.
[Primary source: FA3 author explanation](https://tridao.me/blog/2024/flash3/).

The current fast path already removed per-key-tile running maxima and output rescaling,
and uses identical rounded BF16 P in PV and P*1. Recommending those changes again would
repeat completed work. Remaining normal-path work includes bias load/conversion, exp,
BF16 packing, dependent matrix operations, waits and loop/address instructions.

## Priority1: a real QK/softmax/PV pipeline with a register budget

The current consumer broadly executes:

`QK(t) -> wait -> bias/exp/pack(t) -> PV and P*1(t) -> wait -> next key tile`.

Other warpgroups provide some natural overlap. Within a consumer, however, QK is issued
with `flash::gemm<true,0>` and PV ends in `warpgroup_wait<0>`. Moving one wait alone is
insufficient. Prior `async4/6` only left PV outstanding until the next QK wait, based on
the older stable resident kernel. It did not create a separate next-score accumulator
that lets `QK(t+1)` run while exp processes score(t); both controls were slower.

The next distinct hypothesis is:

1. Retain one head per attention CTA, planar bias, and resident KV at long lengths.
2. Issue `QK(t+1)` into a different register fragment while applying bias/exp to score(t).
3. Preserve each BF16 P fragment until its asynchronous PV/P*1 reads retire. Schedule
   independent consumer groups so their special-function work and matmul phases overlap.
4. Pipeline bias delivery only where it shortens the critical path; TMA instruction
   throughput itself is nowhere near saturation. Extra producers/buffers must earn
   their thread, register and shared-memory costs.

The implementation constraint is severe. L1024 currently has512 threads x126 registers
and231424B dynamic shared memory, plus1024B static shared memory. It cannot simply add
32 score registers per thread or another large shared tile without changing occupancy
or spilling. L768 has768 threads x80 registers and an80-byte stack frame.

A concrete first feasibility variant is two32-key score fragments instead of one64-key
fragment: each fragment has16 FP32 score registers per thread, so two together match
the current32-register score footprint. This is only a budget starting point: probability
fragments, pointers, projection live ranges, extra loop/TMA instructions and smaller
GMMA shapes still need generated-code checks. Another option is fewer consumer groups
with larger per-consumer register budgets; past simple group-count sweeps are already
negative controls, so it must be tested together with actual overlap.

Success requires demonstrated overlap without a compensating spill/traffic increase,
current FP64/mask/retry/graph/sanitizer gates, and paired full-FWD gains. This is a
proposed algorithm, not an implemented or measured speedup.

## Priority2: remove a real HBM intermediate after attention

The current tail is `A @ Wo -> BF16 temporary -> residual/ending transpose -> Y`.
Fuse output projection and residual/orientation in the GEMM epilogue. This retains the
successful attention organization and removes the output-GEMM temporary's write+read.
It does not require combining attention heads in one CTA.

Let `S = 2*L^2*128` bytes. Ignoring the small cached weights, the tail logically moves
about5S: readA, writeT, readT, readX, writeY. A fused epilogue needs about3S: readA, readX,
writeY. Potentially removable traffic is75.5/302.0/536.9MB at384/768/1024. A simple3TB/s
model gives25/101/179us, approximately5–7% of current full FWD. These are modeled
opportunities, not measured gains or hard upper bounds; cache reuse, GEMM efficiency
and ending-direction store layout can change the result.

The epilogue must preserve `BF16(X + BF16(A @ Wo))`. A generic FP32 GEMM beta=1 epilogue
does not preserve the inner BF16 rounding. Keep out-of-place residual semantics and
verify ending-direction addressing before comparing complete FWD with Anthropic.

## What the bias actually costs

`score[i,h,q,k] = Q[i,h,q] dot K[i,h,k] / sqrt(32) + bias[h,q,k]`.

Bias is shared across outer rowi, not across heads. At1024 the unique BF16 bias is8MiB,
but its logical consumption across all outer rows is8GiB. Much of this is served by
cache: the entire selected core measured only0.620GB DRAM, including projections and
output. Flash-style avoidance of the full score/probability HBM tensor is still valuable.

An algorithm exploiting bias reuse must therefore group outer rows or multicast bias
to separate outer-row CTAs. Two-CTA multicast was already tested (`cluster2s`) and lost
after fixing its shared scratch ownership barriers. Full resident KV for two outer
rows would itself need256KiB at1024, beyond one CTA's budget before Q/bias scratch.
Do not prioritize this again without evidence that L2/bias delivery is the main limit
and a concrete lifetime design that avoids the previous synchronization cost.

Precomputing exp(bias) would not remove exp(QK), because exp(QK+b)=exp(QK)*exp(b).
It also changes rounding and extreme-value behavior. No low-rank/sparse bias structure
is assumed, and no approximate attention/model change is proposed here.

## Disposition and evidence

Close four-head attention CTA work. Preserve all selected source/binary artifacts and
training behavior. Focus the core investigation on the dependent non-matmul/matmul
pipeline; output-GEMM/residual fusion is the clearest remaining HBM-bound opportunity.
Neither diagnosis nor this plan is a SOL90 claim.

[Machine-readable measurements and operation counts](bottleneck-analysis.json),
[profiling script](bottleneck_profile.sbatch), [prior full-FWD evidence](HEAD4_KV_RESULTS.md),
[discarded head-layout variants](CTA_HEADS_RESULTS.md).
