# Training attention forward: measurement and CUDA design

2026-09-25. This document preserves the initial design investigation. The
subsequent [implementation and installed results](README.md) select native
CUDA/TMA `cooperative_q2` (promotion17310). Range-certified max-free and the
two-consumer variants were slower and were not installed. SOL90 remains unmet.

## Measured starting point

Job **17262**, node02, normal_h100, completed 0:0. The harness observes the actual
training module's projected operands and captures its current attention core.
BF16, B1/H4/D32, starting direction, every seventh key masked. Both lengths
dispatch `_attn_fwd`, M64/N64, four warps, 128 registers/thread. This is the
training Triton baseline, not the historical standalone CUDA forward.

| Measurement | L768 | L1024 |
|---|---:|---:|
| CUDA event median, core only | 1.6024 ms | 3.6958 ms |
| NCU kernel duration | 1.6149 ms | 3.7594 ms |
| HBM throughput | 11.34% | 8.70% |
| L2 throughput | 49.36% | 50.36% |
| L2 hit rate | 88.61% | 91.30% |
| SM issue-active / reported SM throughput | 68.83% | 69.06% |
| Tensor pipe active | 14.68% | 15.06% |
| Active warp occupancy | 24.74% | 24.82% |
| HBM reads | 457.753 MB | 814.440 MB |
| Size of Q + K + V + bias | 457.703 MB | 813.695 MB |

The HBM reads are already approximately one read of each unique input. This
does not prove that every tensor is read exactly once, but it rules out a large
HBM-bandwidth saving as the primary explanation for an anticipated speedup.
HBM saturation is not the current core's bottleneck. Instruction overhead,
dependency latency, and on-chip traffic are plausible targets; these aggregate
counters alone do not identify a single instruction as the bottleneck. In
particular, historical CUDA MUFU stall samples must not be attributed to this
Triton kernel.

Raw reports, metric units, operand strides and source digest are in
[profile-summary-17262.json](profile-summary-17262.json). The NCU CSV export
emitted an embedded Python encoding diagnostic, but exported complete metric
rows for both kernels; both application records and the Slurm job completed.

## The useful structure

For one head, with outer pair row i, query j and key k:

    S[i,j,k] = scale * dot(Q[i,j,:], K[i,k,:]) + B[j,k]
    O[i,j,:] = softmax_k(S[i,j,k]) @ V[i,k,:]

B is shared across every i. At L768 its BF16 storage is 4.5 MiB. Counting a
separate logical read for every i gives 3.375 GiB, but those are not HBM bytes.
Avoiding materialized S/P still avoids cubic intermediate storage and traffic.
The small D32 makes scalar work and operand movement significant relative to
each score's two dot products.

The previous M1 CUDA already has three-row bias sharing, fragment-order FP32
bias prepacking, max-free streaming with a repair pass, TMA producer warps,
overlapped QK/PV, and a tensor-core P-times-ones denominator. These are reusable
starting points, not new discoveries. Its ~0.8 ms historical core number is
not a training replacement result: it does not emit the required LSE and its
normalizer/masked-row behavior needs changes.

## First implementation: training-correct asynchronous CUDA

Start with one known-good streaming ownership pattern, rather than an R/M/N
sweep. Reuse M1's producer/consumer transport where appropriate:

- Each consumer owns complete output query rows; Q and FP32 O/statistics stay
  on chip until the last key. S and P never become global intermediates.
- Dedicated producer warps issue Q/K/V/bias TMA. K and V stages have separate
  reader-completion rules; a QK wait does not authorize overwriting V before PV.
- Bias preparation, if retained, computes fragment-order bias/scale once per
  call in FP32. This trades a quadratic buffer for less cubic conversion and
  layout work. Include its cost and lifetime in every timing comparison.
- Issue independent QK ahead of scalar softmax, and PV after probability
  packing, with delayed waits. Raw bias layout and operand fragment selection
  must make the same scheduling usable for dK/dV.
- Initially retain stable online normalization as the correctness baseline.
  Output projection-layout BF16 O plus contiguous FP32 **base-2 LSE**. Sum the
  unrounded FP32 probabilities for normalization/LSE; do not silently substitute
  M1's sum of BF16-rounded probabilities from P-times-ones.
- Preserve finite-zero fully masked output and the current guarded LSE
  convention (the installed forward clamps at -1e38). The old standalone
  uniform-mean-V empty-mask behavior is not the training contract.

Warp specialization and interleaving matmul/softmax follow the general
[FlashAttention-3 approach](https://arxiv.org/abs/2407.08608). Its published
utilization/speedups are not predictions for this D32 pair-bias workload.

## Algorithm experiment: fixed shift certified by cheap bounds

The dependent online-max/rescale recurrence prevents arbitrary overlap across
key tiles. Test whether it can be skipped for rows with a cheap conservative
range certificate. This is a refinement of the old max-free method for the
training contract, not a claim that fixed-shift softmax is new.

For live keys, define:

    bmax[j] = max_k B[j,k]
    knorm[i] = max_k ||K[i,k,:]||_2
    A[i,j] = scale * ||Q[i,j,:]||_2 * knorm[i]

Cauchy-Schwarz gives, in real arithmetic:

    bmax[j] - A[i,j] <= max_k S[i,j,k] <= bmax[j] + A[i,j]

Thus c = ceil((bmax + A) * log2(e)) gives p[k] = exp2(S[k]*log2(e)-c)
with p <= 1 and max(p) >= 2^(-(2*A*log2(e)+1)). For a sufficiently tight
certificate, at least one probability is safely normal and every key tile
uses the same scale. The loop can accumulate FP32 l=sum(p) and O without
per-tile maxima or rescaling, then emit:

    O_final = O_acc / l
    LSE2 = c + log2(l)

Use an integer exponent shift, actual norm bounds, rounding margins and
explicit finite/mask checks. An initial conservative 32-bit bound-width cap
is an experimental choice, not a proven optimal threshold. A rejected
certificate takes stable online softmax. All-masked rows take their explicit
existing contract. Q norms can be computed while Q is resident; K norm maxima
cost one quadratic K read initially and can later be fused into projection.
Bias maxima can join bias packing. Initial extra reads at L768 are roughly
151 MB for K plus the small bias pass, so preprocessing may erase the gain.

The bounds certify range in real arithmetic, not bitwise equivalence or all
FP32/BF16 rounding behavior. Tiny-tail underflow, accumulation order, extreme
V, LSE reconstruction and gradients still require numerical qualification.
Retain FP32 normalizer accumulation. No polynomial exp, low-rank bias or
attention sparsification is part of this proposal.

## Secondary experiment: keep a full bias strip resident

A CTA owns (head, query tile, outer-row shard), loads B[j:j+64, all k] once,
and iterates outer rows in small batches. Reuse grows without simultaneously
keeping dozens of O accumulators live. Two consumer WGs can own two outer
rows while a producer WG streams Q/K/V for the next work.

Candidate data allocation for M64/N64/R2 with three K/V stages:

| Allocation | L768 | L1024 |
|---|---:|---:|
| BF16 full bias strip | 96 KiB | 128 KiB |
| Q, two outer rows | 8 KiB | 8 KiB |
| K/V, three stages, two rows | 48 KiB | 48 KiB |
| Data subtotal | 152 KiB | 184 KiB |

This excludes barriers, alignment and any output store staging. The measured
device opt-in shared-memory limit is 232448 bytes; compiler/register fit is
unproven. O/stats remain in registers. Six row shards yield 288/384 CTAs for
L768/L1024 on 132 SMs. This reduces repeated bias transport but not the
per-score exponentials or shared-to-register bias reads. It may lose occupancy
and Q/K/V cache locality; actual HBM reads could increase. Against historical
M128 CUDA, M64 also doubles logical K/V reads across query tiles. Given the
measured HBM and L2 headroom, this is secondary to the asynchronous core.

## What transfers to dK/dV, and what does not

Bias fragment layout, TMA staging, asynchronous score/gradient MMA scheduling,
and the separation of simultaneous rows from sequentially processed rows are
transferable. dK/dV already has saved LSE, so eliminating a forward online-max
recurrence does not itself speed backward.

For dK/dV the bias-resident orientation becomes B[all queries, key tile]. Each
consumer must finish all queries before writing its final dK/dV. Crucially,
the existing fused dB reduction must survive: a full FP32 dB accumulator for
a 64-key strip needs 192/256 KiB at L768/L1024, in addition to 96/128 KiB of
BF16 bias. Both cannot simply reside in one CTA's shared memory. Writing more
dB partials or splitting dK/dV introduces extra HBM traffic and reductions.
The forward design therefore cannot be copied wholesale into backward.

## Selection order and stopping criteria

1. Establish native CUDA O+LSE with correct masks and the current training
   layout, then measure full FWD and F+B against frozen checkpoint17211.
2. Isolate the range-certified normalization experiment. Count norm/bias
   preparation, certificate failures and stable fallback time.
3. Compare full bias residency only with the same arithmetic/pipeline baseline.
4. Qualify both directions and L384/768/1024 with nonzero output weights,
   independent FP64/mask cases, all input/parameter gradients, CUDA graph and
   sanitizer gates before dispatch changes.

Reject a transport change that merely improves a utilization counter, an
O-only kernel that damages LSE/gradients, or a core win erased by preprocessing.
Do not claim a speedup from this document: only the baseline profiling has run.
