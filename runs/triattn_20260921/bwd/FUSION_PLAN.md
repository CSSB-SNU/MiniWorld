# Backward fusion design: minimize intermediate memory traffic

> **Current backward installation, 2026-09-25:** [Installed report](core90/README.md). Qualified `rs8_async_q4` dK/dV + bias and `rs_softmax_overlap` dQ pass installed-path job17211, including all input/parameter gradients. Complete backward improves21.92–23.23% and F+B14.63–16.00% versus job16663. Attribution17211 accounts for every16/17 backward kernels. SOL90 remains unmet. All new experiments use `node02 / normal_h100`.

> **Priority4 now implemented:** [priority_pass/README.md](priority_pass/README.md) records native CUDA/TMA joint Q/K/V/gate weight gradients, including FP32 partial-buffer traffic. NCU16659 measures44.53% less L768 HBM read+write traffic for those four gradients; installed16663 includes all correctness and full-module performance checks. Historical design/status paragraphs below predate this update.

> **Newer installed result:** [SOL90_PROGRESS.md](SOL90_PROGRESS.md) records the qualified grouped core, native dQ, and projection/LN/residual updates, including L384; the latest gate/context status is recorded there. Historical qualification and traffic data below remain tied to their named artifacts. Complete backward SOL90 is not reached.

2026-09-23. The analysis below defines the baseline and proposed fusion boundaries.
Priority1 is now implemented: [bias_fusion/README.md](bias_fusion/README.md) records
the selected native CUDA/TMA R4 kernel for L768/1024. It halves bias scratch;
NCU job16104 measures32.7% less whole-core DRAM traffic at L768. Installed
job16112 gives0.23–0.32% /2.74–2.99% additional whole-module backward gain at L768/L1024.
This paragraph describes the older job16112 installation. The current result in
SOL90_PROGRESS supersedes it: priorities2 and3 are installed, and L384 also uses
the selected grouped core. Joint weight-gradient scheduling (priority4) remains
unimplemented; the remaining sections preserve the original design analysis.

The user's explicit priority is to choose fusion boundaries first. In this
workload, avoiding HBM round trips has usually been worth substantially more
than optimizing an isolated kernel. Prefer on-chip reuse and, where beneficial,
recomputation. Count every consumer and replacement buffer before claiming a
memory saving. New implementations are native CUDA C++ with TMA and CTA
producer/consumer scheduling.

## Baseline dataflow and ownership

Let X be the pair input, Z=LN(X), G the pre-sigmoid gate, O the attention output,
A=sigmoid(G)*O, Y=X+dropout(A*Wo^T). Linear weights use Torch's [out,in] convention.
Indices i/j/k are outer pair row/query/key, h is the attention head.

```text
dY -- saved dropout mask / ending layout --> dYbranch
    -- output projection + gate backward --> dO, dG, A
                                               |       +--> dWo
                            O + dO --> delta   |
                          /                    |
            dQ kernel     + dK/dV kernel --> per-i dS --> sum_i --> dB
                 \            /                                  /
                  dQ, dK, dV, dG, dB ---------------------------+
                      |                               |
                      +--> five weight gradients      +--> summed dZ
                                                           |
                                                      LN backward
                                                           |
                                                      dXbranch + dY
```

Important existing fusions: output-projection dgrad and gate derivatives already
share `_dgrad_epi`; the five projection input gradients already share the new
CUDA kernel. The following proposals extend those boundaries.

Current consumers:

| Tensor | Consumers that must still get the value |
|---|---|
| dO | delta, dQ, dK/dV |
| dQ/dK/dV/dG | projection input gradient **and** their weight gradients |
| dB | input-gradient bias term **and** bias projection weight gradient |
| A | output-projection weight gradient |
| dZ | LN input gradient and LN scale/shift gradients |
| dXbranch | residual addition |

## Traffic accounting and priority

B1, H4, D32, C128. Let F=2*L²*C bytes, one full BF16 activation.
The figures below count explicit global-memory tensor writes/reads, in decimal
GB. They are an algorithmic traffic model, **not newly measured DRAM counters**;
caching, repeated GEMM tile loads, allocator behavior and parameter partials
require separate accounting.

| Priority | Fusion | L768 nominal traffic removed | L1024 nominal traffic removed |
|---|---|---:|---:|
| 1 | dK/dV + local row-group dBias reduction, R=4 / R=8 | 3.624 / 5.436 GB | 8.590 / 12.885 GB |
| 2 | projection dgrad + LN backward + residual | 0.604 GB = 4F | 1.074 GB = 4F |
| 3 | output/gate backward + delta | 0.302 GB = 2F | 0.537 GB = 2F |
| 3 extension | fold dropout derivative and ending layout into producer loads/final stores | route dependent | route dependent |
| 4 | joint weight-gradient schedule reusing Z and gradient tiles | determine after partial-buffer accounting | same |

These are individual opportunities, not promised timings or a sum of independently
measured benefits. Priorities 2/3 preserve the large multi-consumer gradient
buffers initially; priority 4 considers whether those can actually disappear.

## 1. Reduce dBias across outer rows before writing it

For each head, the derivative of the shared pair bias is

    dB[h,j,k] = sum_i dS[i,h,j,k].

The pre-fusion dK/dV CTA owns one (i,h,key tile), loops over queries, and writes
every dS element to a BF16 buffer. This is 2*B*H*L³ bytes: 3.624GB at L768 and
8.590GB at L1024. Writing then reducing it costs at least two tensor traversals:
7.248GB and17.180GB, excluding the small final output.

Proposed ownership: one CTA owns (batch, head, **R outer rows**, one key tile).
It loops over **all query tiles**, retaining a separate dK/dV accumulator for
each owned outer row. Within each query tile, combine those R rows' dS values
on-chip and write one FP32 bias partial. dK/dV remain final outputs, not split
partials. No elementwise global atomic is needed for this arrangement.

```text
own i in [group*R, (group+1)*R), fixed h and key tile
keep dk[i], dv[i] accumulators
for every query tile j:
    load the shared bias[j,k] tile once
    db_tile = 0 (FP32)
    for each owned i (parallel or interleaved consumer warpgroups):
        form P and dS using Q, K, V, dO, LSE and delta
        dk[i] += dS^T * Q; dv[i] += P^T * dO
        db_tile += bf16(dS) promoted to FP32
    store db_partial[group,h,j,k] once
store final scale*dk[i], dv[i]
```

The BF16 rounding of each dS contribution can remain in registers/shared memory
to preserve the existing bias-reduction inputs. Summing unrounded dS is a separate
numerical variant, not a free equivalence. FP32 partials avoid another BF16 rounding
at the group boundary; reduction-order differences still need validation.

For lengths divisible by R, the FP32 partial buffer is
4*B*H*L³/R bytes. Compared with the current BF16 buffer, its write+read traffic is
**2/R**, not1/R. R2 does not reduce these bytes; R4 halves them and R8 quarters them.
At L768/R8 the scratch becomes0.906GB and its write+read becomes1.812GB.

TMA producers stream row-specific Q/dO and shared bias; K/V reuse across query
tiles is planned within the CTA resource budget. Consumers own row-specific
accumulators. The bias reducer consumes their shared dS tiles. A stage becomes
reusable only after **both** WGMMA readers and the bias reducer finish.

Start resource analysis at R4, then R8. For key tile64 and D32, persistent FP32
dK/dV alone require16KiB*R across the CTA. dS storage, input rings, register
distribution, shared-memory capacity and resident CTAs must all fit; R8 is a
candidate, not a claim of an already viable launch configuration. Choose the
warpgroup mapping from this resource ledger, not an arbitrary CTA count.

Keep dQ separate at first. Final bias reduction can fuse mask-gradient application
and write projection-friendly [B,L,L,H] layout, removing subsequent masking/copy
passes without changing the mathematical reduction.

The old v8 global-atomic experiment was slower at C512/D128/L384. It demonstrates
the cost of an atomic per dS element, not the impossibility of local row grouping
at C128/D32. Its conclusion is not a reason to retain today's multi-GB scratch.

## 2. Finish LayerNorm and residual in the projection dgrad epilogue

The installed CUDA CTA already produces **all128 channels** of each owned token's
dZ. This is precisely the ownership needed for the two LN feature reductions:

    xhat = (X - mean) * rstd
    t = dZ * gamma
    dXbranch = rstd * (t - mean_C(t) - xhat * mean_C(t*xhat))
    dX = dXbranch + dY

Compute these while dZ is on-chip. Eliminate dZ's store/read and dXbranch's
store/read: 4F. The final residual dY read and final dX write still exist.
Load original X, saved statistics and gamma; do not confuse affine Z with xhat.

LN parameter gradients must also be produced: dgamma=sum_tokens(dZ*xhat),
dbeta=sum_tokens(dZ). Emit bounded FP32 parameter partials and reduce them, or
prove a low-contention alternative. For token tile64 at L768, two128-channel
partials per CTA occupy about9.44MB, whose write/read must be deducted from the
nominal saving. They are not full activation-sized tensors.

To isolate the fusion effect, initially retain BF16 rounding of dZ and dXbranch
on-chip at the same mathematical boundaries as the installed path. A fully FP32
chain is a later numerical/performance variant.

For ending attention, store final dX in the original input coordinates and combine
with the residual there. Use tiled address conversion; assess coalescing rather
than adding a full intermediate transpose.

This requires a shared autograd boundary exposing LN saved statistics, projection
weights and the residual gradient. Merely placing launch calls next to each other
does not remove the tensors between existing autograd Functions.

## 3. Compute delta where dO is produced

The current output/gate epilogue already loads O and produces dO. Add

    delta[i,h,j] = sum_d O[i,h,j,d] * dO[i,h,j,d]

there. This removes the separate preprocess pass reading O and dO: 2F. Keep the
small FP32 delta array because both attention backward consumers need it. For
baseline-compatible arithmetic, reduce using the rounded BF16 dO that the next
attention kernels will read, not the unrounded projection accumulator.

For training dropout, apply the **saved forward mask/scale** when loading dYbranch;
fold ending layout conversion into these loads. An eliminated dropout-only
intermediate can save another2F when that route actually materializes it. This is
not a saving in the dropout=0 timing case.

A, dG and dO still have independent consumers. Preserve them until a subsequent
fusion accounts for those consumers. In particular, recomputing A=sigmoid(G)*O
inside a separate dWo GEMM removes its allocation but introduces G/O reads; it
does not automatically reduce total bytes.

## 4. Weight gradients and full attention fusion: account for new partials

Weight gradients are dWq=dQ^T*Z, dWk=dK^T*Z, etc. A grouped CUDA schedule can share
Z loads across projections without packing a new [tokens,516] gradient matrix in
HBM. Combine this with the input-gradient consumer only if its weight partials
and resource footprint beat the saved gradient reads.

Counterexample: producing a complete five-projection FP32 weight partial for
each64-token CTA at L768 creates about2.435GB of weight partials, before the final
reduction. That can cost far more than the read it was intended to eliminate.
Use a much more amortized schedule, reduced output ownership, or shared-memory
producer/consumer reuse with a properly bounded partial count.

Likewise, a key-owned kernel computing dQ/dK/dV together does not magically own
final dQ. Writing FP32 dQ partials per key tile of width N costs
4*B*H*L³*D/N bytes. At D32/N64 this is exactly as large as today's BF16 dS buffer:
3.624GB at L768. Its write/reduce can undo the entire traffic benefit, while the
row-group bias partials are still required. N128 changes that accounting but also
changes resources. Global atomics or a CTA/cluster owning the complete reduction
are other candidates with explicit costs, not assumed solutions.

Retaining two local P/dS computations for dQ versus dK/dV can therefore be the
better memory-first algorithm. Also, fusing attention with a projection must
account for all heads, all query/key reduction contributions, and dW consumers
before claiming that dQ/dK/dV buffers disappear.

## Next implementation sequence and acceptance

1. CUDA row-group dK/dV + dBias partial reduction, with unchanged dQ as a control.
2. Extend the installed CUDA projection epilogue through LN and residual.
3. CUDA output/gate backward + delta, with dropout/layout folded into the same
   producer path when applicable.
4. Use the resulting traffic profile to choose the weight-gradient sharing and
   larger attention fusion schedule. Do not pack an intermediate solely to make
   a GEMM interface convenient.

Measure complete backward and forward+backward, both directions and all parameter
gradients. Record DRAM reads/writes as well as timing; distinguish warm graph/L2
reuse from streaming traffic. Check scratch peak, spill traffic, atomic costs and
barrier waits. A nominal memory saving is not a speedup until measured. Preserve
the current installed path until a candidate passes numerical, graph/compile,
dropout/optimizer and sanitizer checks. Forward SOL90 remains closed.

Source anchors: `triangle_attention/triton/main.py` (`_attn_bwd_dkdv`, `_tri_attn_bwd`),
`bias_only_attention/triton/gate_out.py` (`_dgrad_epi`, `_FusedGateOut.backward`),
`layernorm/triton/main.py` (`layer_norm_bwd_dx_fused`), and the installed
`triangle_attention/cuda/projection_dgrad.cu`, all under the engine's kernels.
