# Next distinct architecture to evaluate: producer computes Q/K/V projections

Implemented and measured; all fused variants below were REJECTED for latency.
Current installed6d52 remains804.416us/SM60.630086%; no SOL90 claim.
Standalone STSM96 projection passed10 bitwise Q/K/V/G/bias checks and
memcheck/racecheck/synccheck atL384 in both orientations (job14405).

| Variant | Job | hot us | whole block us | installed block us |
| --- | --- | ---: | ---: | ---: |
| p96z |14411|1541.121|2040.788|1395.448|
| p96loop |14415|1610.581|2096.739|1368.450|
| p128sszm128 |14417|1510.556|2033.851|1389.713|
| p128sszm128cl4 |14422|1874.448|2355.786|1359.585|

Each passed the initial five mask patterns/FP64 accuracy screen. All four
built without spills or C7512; no production installation or full fused
sanitizer/ending qualification followed, given their clear regressions.
The shared-zero descriptor dependency in z variants prevents hoisted
projection descriptors from spilling. The loop variant rolls K16 projection.
The m128 variant projects both row halves together. The cl4 variant groups
four head CTAs and multicasts X with rank0-issued TMA and distributed
source-lifetime barriers. Its traffic reduction was NOT profiled; only its
latency regression is established.

NCU14416 for p96z:1534.496us, SM/tensor34.653835%, occupancy17.988633%,
issue29.837028%. DRAM read613.627648MB/write147.707392MB; L2 traffic8.561676224GB
versus installed4.975161312GB. Projection recomputation and repeated input
reads eliminated the expected bandwidth benefit. No serving files changed.

The remainder records the original design and implementation history.

The current attention already has low Tensor Core utilization while waiting on
scalar exponentials and shared reads. A potentially useful next step is to
remove global Q/K/V materialization and put their linear projections into the
attention producer warpgroup. The complete block, including a new preparation
pass, must be faster. Additional projection work must never be used merely to
inflate a utilization counter. Report fused and unfused work separately.

## Exact input arithmetic is available

Installed `oc/opt_core/kernels/triattn_surround_tma/prologue.cu` computes LN with
16 lanes per row,8 adjacent channels/lane and explicit rn reduction operations,
then rounds LN output to BF16 before WGMMA. Q/K/V/G projections accumulate over
8 K16 chunks and are rounded to BF16. Bias is also rounded to BF16 then stored
in FP32. Copy this arithmetic and layout exactly; do not replace LN with a
different reduction or keep unrounded Q/K/V in attention.

A preparation pass can emit normalized BF16 input X[i,j,128], existing BF16 G,
and existing FP32 bias. It replaces the Q/K/V/G/bias preparation. Starting and
ending input orientation must follow the current TMA prologue descriptor.

At L768, one Q/K/V/G or X tensor is about151MB. Replacing Q/K/V output with X
saves about302MB of writes and another302MB of attention input reads if the
input stays cached across heads/tiles. Actual DRAM traffic must be measured.

Attention is about232 GFLOP (QK+PV). Materialized Q/K/V projection is about58
GFLOP. Recomputing K/V for all6 query tiles costs about232 GFLOP for K/V plus
19 GFLOP for Q, so the fused kernel does about483 GFLOP. This is a tradeoff:
more Tensor Core work in exchange for fewer global passes, not a FLOP reduction.
The rough compute roof may match the EX2 work; shared-memory traffic could
still prevent useful overlap. These are estimates, not performance evidence.

## First bounded proof

Before integrating attention, write a standalone CUDA producer projection:
TMA X and per-head weights, WGMMA Q/K/V with the same K16 accumulation order,
BF16 conversion, then direct STSM into the exact existing Q/K/V shared layouts.
Validate emitted Q/K/V bitwise against the installed prologue for both
orientations, including random and large-value inputs. This proof should also
verify a64-register producer can compile without spills or serialization.

Use R2/M128 attention with two216-register consumers and one64-register
producer (384 threads). The initial pool must reserve at least63488 registers;
168/thread gives64512. The producer is a complete warpgroup and performs MMA
uniformly; do not run only one warp through a WGMMA or setmaxnreg operation.

Tentative shared storage, BF16 except FP32 bias:

- Q:2*128*32*2 =16384 bytes.
- K/V two-stage ring:2*2*128*32*2*2 =65536 bytes.
- Eight64x32 bias halves =65536 bytes.
- Two128x128 X stages =65536 bytes.
- Per-head [W_K;W_V] weights64x128 =16384 bytes.
- Separate8x32 denominator ones tile =512 bytes.
- Total229888 bytes before barriers; static_assert <=232448 is mandatory.

Use separate denominator MMA initially: the installed N40 fusion's8KB ones
storage would exceed this plan. W_Q32x128 can temporarily occupy the weight
storage before W_K/W_V replace it, after Q projection is fully drained.

One producer WG projects M64xN64 halves for K/V, accumulates32 FP32/thread,
rounds to BF16 and writes directly into K/V stages. Avoid an extra temporary
output tile: form a strided/swizzled output view where N's first32 channels map
to K and second32 map to V (static distance equal to the whole K array). Verify
this view independently with memcheck/racecheck before attention consumes it.
Q projection is M64xN32. The installed prologue's layout/copy code is the model.

The producer must interleave bias TMA, X prefetch and projection; all four
producer warps participate in MMA. A scalar warp cannot independently stream
bias forever while its warpgroup is doing WGMMA. Projected K/V are ordinary
shared stores, so fence the async proxy before signaling consumers. Empty
barriers must protect every ring reuse until the last PV consuming V retires.

## Required semantics and integration

Start with B1,N=S768,H4,D32 and broadcast boolean masks. Preserve the exact
power-of-two hot rescaling and SAFE running-max recompute. SAFE must use the
new R2 CTA mapping; the old R3 fix-list mapping is incompatible. Fully masked
rows must average the rounded V values, not project the mean X (rounding makes
those different). A dedicated rare fallback may materialize V and reuse the
established uniform mean kernel, but normal cases must not pay that bandwidth.
Do not omit all-masked or late-seed behavior just to benchmark a candidate.

The public core API currently receives Q/K/V. A fused path therefore needs an
explicit block-level dispatch using raw features/weights and the new preparation
pass, with the existing prologue/core as fallback for unsupported shapes. Keep
existing out-of-place residual, gate and epilogue semantics. Count preparation,
fused attention, fallback checks and epilogue in paired whole-block timing.
Only integrate after meaningful speedup and full numerical/sanitizer checks.


## Initial implementation evidence (2026-09-22 05:30 KST)

`projected/check-scalar-14402.json` and `check-stsm-14403.json` each contain10
bitwise Q/K/V/G/bias cases (both orientations, random/large/constant/tiny/outlier).
The direct STSM K-to-V gap view is numerically correct. K/V weights need a
static per-head [64,128] cache: concatenating two independently tiled [32,128]
SW128 shared tiles does NOT equal a [64,128] SW128 layout. `layout_probe.cu`
shows the different K64 stride. This was corrected before the passing checks.

Standalone scalar64 has1040B stack,1136/1412B spills. STSM64 reduces this to
176B stack,260/260B spills but gets C7512. Both are correctness proofs only.
STSM96 job14405 uses90 registers, zero spills and no C7512;10 cases pass.
Memory/race/sync checks completed with zero errors (job14405, both orientations).

`make_fused.py`, `fused_producer.cuh`, `fused_host.cuh`, `uniform_body.cuh` now
implement an isolated L768 R2/two-stage attention path. The producer projects
Q/K/V, overlaps two X row buffers with computation, interleaves bias TMA and
signals consumers after STSM/proxy fencing. Consumers release KV only after
the last PV retires. SAFE uses one fix flag per new CTA. The all-masked path
projects rounded V into the output allocation then averages it in place.
Whole-block timings must count this launch and preparation/epilogue.

Initial64-producer/216-consumer build14404 has560B stack,1440/1448B spills
and C7512 in hot/SAFE; it was NOT launched. Job14406 builds96-producer and
200-consumer instead (63488 dynamic registers,168 initial/thread reserve).
These initial build failures were superseded by the measured variants above.
No serving files have changed since6d52. SOL90 remains unachieved.
