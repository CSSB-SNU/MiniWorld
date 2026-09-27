# QKV projection plus attention: proposed ownership and traffic

2026-09-25 design; implementation follow-up 2026-09-26. **Seven native CUDA fusion
candidates have now been implemented and measured.** See
[QKV_FUSION_RESULTS.md](QKV_FUSION_RESULTS.md) for measured outcomes and current
installation status. The ownership/traffic discussion below preserves the
original design rationale. Target: native CUDA, H100, BF16,
B1/H4/D32/C128, L384/768/1024, training and both attention directions.
The completed baseline for this next experiment is installed checkpoint17628
(`cooperative_head2`); see [current measurements](README.md).

## Audited current path

`cuda/ln_backward.py:Front.forward` computes canonical Z=LayerNorm(X), then five
separate `F.linear` calls for Q/K/V/gate/bias. Its backward saves Z, original X,
normalization statistics and weights, and uses the installed fused projection/
LayerNorm/residual dgrad plus joint Q/K/V/gate weight gradients.

`cuda/gate_backward.py:AttentionGate.forward` calls the native attention core,
then the gate/output projection. It saves Q/K/V, bias, LSE, gate, weights and O.
Both native dQ and native dK/dV currently consume the saved Q/K/V. Therefore
projection-attention fusion cannot simply delete Q/K/V training saves.

For a fixed outer row i and head h:

    Q[j] = BF16(Z[i,j,:] Wq[h,:,:]^T)
    K[k] = BF16(Z[i,k,:] Wk[h,:,:]^T)
    V[k] = BF16(Z[i,k,:] Wv[h,:,:]^T)
    O[j] = softmax(Q[j] K^T / sqrt(32) + Bias[h,j,:]) V

Bias[h,j,k] is projected from Z[j,k,:], not Z[i,k,:]. It is shared across every
outer row i. Keep bias/mask preparation before attention; the outer-row-local
QKV producer cannot generate all of its required bias from its own input row.

## Avoid recomputing K/V for every query CTA

Keeping the existing 64-query ownership and inserting K/V projection into its
key loop would recompute the same K/V L/64 times. This is a structural cost,
not an implementation detail that TMA can hide.

| L | Query CTAs per row/head | Current QKV projection GFLOP | Naive per-query fusion projection GFLOP | QK+PV GFLOP |
|---|---:|---:|---:|---:|
| 384 | 6 | 14.50 | 62.81 | 28.99 |
| 768 | 12 | 57.98 | 483.18 | 231.93 |
| 1024 | 16 | 103.08 | 1133.87 | 549.76 |

These are algebraic operation counts (multiply-add=2), excluding softmax,
normalization, gate and bias. Preserve the BF16 rounding boundary after each
projection; carrying unrounded projection accumulators into attention changes
the numerical operation.

## Proposed first fusion

One CTA owns one `(outer row, head)` and all its query tiles. Produce Q/K/V once
with WGMMA, store them in resident shared memory, then let several consumer
warpgroups process disjoint 64-query tiles while reusing resident K/V.

    TMA Z tile + head weights
          -> WGMMA Q/K/V projections -> BF16 resident Q/K/V
                                         |             |
                              training saves to HBM    +-> attention WGMMA/softmax -> O/LSE

Use adjacent-head CTA ordering so the four head CTAs can reuse normalized input
cache lines. A possible later control multicasts Z tiles across the four heads;
it must earn its cluster/barrier cost in measured complete-workload timings.

The first prototype keeps the training Q/K/V stores once per element and consumes
the on-chip values directly. It adds no global score/probability buffer and does
not materialize a second QKV representation. Q/K/V stores must finish before their
shared storage is overwritten (including any reuse for output staging).

| L | One-head complete BF16 QKV | Four consumers, one 8KiB bias slot each | Four consumers, two bias slots each |
|---|---:|---:|---:|
| 384 | 72 KiB | 104 KiB | 136 KiB |
| 768 | 144 KiB | 176 KiB | 208 KiB |
| 1024 | 192 KiB | 224 KiB | 256 KiB, does not fit |

The latter columns are QKV+bias storage only, **not final kernel allocations**.
Include aligned barriers, weight/input projection scratch and store lifetimes.
Projection scratch can reuse the later bias region after projection finishes;
one 64x128 BF16 Z tile plus one 32x128 BF16 weight tile is 24KiB. Double buffering
that scratch or overlapping its lifetime with bias needs an explicit budget.

H100 permits 227KiB per CTA, with 228KiB per SM. L1024 is close to the per-CTA
limit even with one bias slot per consumer; L768 resident-QKV designs permit
only one CTA/SM. Multiple consuming warpgroups must compensate for the loss from
the current six smaller CTAs/SM. Query group count and bias staging therefore
need shape-dependent decisions. These are feasible storage sketches, not an
occupancy or speed claim. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/archive/12.6.3/hopper-tuning-guide/index.html)

## Traffic that actually disappears

At L768, QKV contain 452.985MB in total. With existing backward retained:

- Their forward writes remain as backward saves.
- Attention's global Q/K/V reads disappear from the fused kernel. The current
  two-head schedule reads 457.77MB total HBM in profile17561, including more than
  just QKV; this is measured core traffic, not a guaranteed net saving for fusion.
- Repeated attention K/V requests to L2 also disappear: current64-query CTAs
  reload K/V for each query tile. The nominal QKV payload delivered to attention
  at L768 is F + 2F*(L/64) =3.775GB, with F=2*L^2*128. This is logical payload,
  not measured L2 sectors or HBM bytes.
- Each head CTA reads the128-channel Z row. Four head CTAs request 4F logical Z
  bytes versus 3F for three separate projections; cache/multicast reuse affects
  physical traffic. Count this replacement input traffic, weights, and training
  stores in the fused measurement. Do not report the eliminated QKV reads alone
  as a net end-to-end saving.

For inference, QKV saves can disappear too. For training, dropping saves would
require recomputation in backward and a new complete F+B comparison. Merely
recomputing QKV into global memory once before backward shifts their write cost;
it does not prove a full-workload traffic reduction.

The recent head-order control halved core HBM reads (910.76 ->457.77MB) but only
modestly improved full FWD. The attention core is also limited by L2 traffic,
softmax/issue work and synchronization. Fusion is attractive because it can
remove repeated L2 K/V loads as well as the HBM interface, but projection GEMM
efficiency and CTA occupancy can still outweigh that benefit.

## Autograd and validation boundary

A new combined front+attention custom-autograd boundary is needed. Leaving
`Front.forward`'s QKV linears active and calculating them again inside attention
would not implement the intended fusion. Keep LN, gate and globally prepared bias
at first; move only QKV projection into native attention.

Save the same X/Z/statistics/weights, QKV/bias/LSE/gate/O values needed by existing
backward kernels. In backward, reuse gate+delta, dQ, dK/dV+dBias, fused projection/
LN/residual dgrad, joint four-weight gradients and output weight gradient. Preserve
the residual branch and ending-layout transformations. Audit frozen/partial
parameters, dropout and BF16 AMP before dispatch changes.

Measure **projection + attention**, full FWD and full F+B against the installed
native baseline; isolated attention timings would hide the moved projection
work. Require independent FP64 references and gradients, both directions, masks,
changed-input graph replay, fullgraph, memcheck/racecheck/synccheck, real dispatch
and installed-path validation before selection. No speedup is asserted by this
design document.
