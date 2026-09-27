# Resident QKV with tensor-core softmax reduction

The `hot6t` experiment changes the arithmetic inside the inference QKV/attention/gate kernel.
It keeps the previous fusion boundary: normalized Z and bias enter; gated attention A leaves.
Q, K, V, gate logits, probabilities and LSE have no global tensor outputs. The front, output
projection and residual are unchanged.

## CTA dataflow

One CTA owns one head and one outer pair row. Six warpgroups cooperate at production lengths.
They first compute K/V and retain the complete row in shared memory. Each query batch then
computes Q and gate, runs attention, applies sigmoid gate and stores gated A through TMA.
L768 has two batches of six 64-query tiles; L1024 has two full batches and one four-tile tail.
Projection weights and Z are loaded with TMA; QKV projection, QK and PV use WGMMA.

Projection scratch shares storage with Q/gate and bias scratch. CTA barriers retire every
projection reader before the union changes purpose. The output TMA source is also retired
before the next query batch can overwrite it. This candidate uses cooperative warpgroups;
there is no dedicated producer warpgroup or register redistribution.

## Common-case softmax

For a row with logits s, the common path forms p = BF16(exp2(s * log2(e) - 16)).
It accumulates n = sum(p * V) and d = sum(p), then forms BF16(n / d). The fixed offset
cancels in the ratio. It avoids an updated maximum and rescaling of n at every key tile.

Both reductions use the **same BF16 p**. The earlier `hot4` attempt used unrounded FP32 p for
d and BF16 p for n. A single live key exposed that mismatch: its output should equal V,
but dividing BF16(p) * V by FP32(p) introduced an error. That attempt was rejected without
changing the FP64 threshold.

`hot4r` and `hot6r` fixed the mismatch with scalar BF16 probability sums, but were slower.
`hot6t` instead issues P*V and P*1 WGMMA operations in the same committed group. The ones
operand is an 8x16 BF16 tile reused for all four K16 fragments of a 64-key tile. The eight
output columns duplicate the denominator; each thread reads the two row sums from its
accumulator. No per-key-tile scalar denominator reduction or final shuffle sum is needed.
The 256-byte ones tile fits the alignment gap beside the retry flags, so it does not increase
the selected allocation relative to `hot6r`.

This changes probability rounding relative to the original online softmax. It is checked
against the existing FP64 and full-module tolerances, not advertised as bitwise equality.
Attention-output BF16 rounding before gating remains explicit.

## Stable retry and synchronization

After the common pass, a warpgroup retries if either denominator is outside [1e-16, 1e16]
or any numerator is nonfinite. Four warp votes are exchanged through shared flags and a
named barrier. The entire warpgroup then takes the same branch. The stable path uses the
previous running-max/rescale algorithm and FP32 denominator sum. Q, K, V and gate remain
resident during the retry, so it does not create QKV HBM intermediates.

Bias transaction-barrier phases advance after **every pass**, including retries. A retry in
one warpgroup must not shift another warpgroup's phase. Runtime tests cover a single retrying
query tile among normal tiles and subsequent query batches, and replay the same CUDA Graph
while changing inputs between normal and retry paths. L64 uses a one-stage bias ring so its
single key tile also advances the phase correctly.

All-masked inputs retry and return zero update, preserving our existing inference contract.
Anthropic's all-masked mean-V convention remains a documented difference. Large-logit and
extreme-offset inputs exercise the slower stable path; reported timing uses the mixed mask
with valid keys and ordinary random nonzero weights.

## Controls and measured limits

- Static query groups (`splitq2/4`) remove repeated query-batch transitions but recompute
  resident K/V in multiple CTAs. Their total FWD is slower.
- Joint N128 projection (`joint4`) produces K,V,Q,gate from one Z load, but computes unused
  Q/gate for other query groups. It does not recover the repeated-CTA cost.
- Leaving PV outstanding until the next QK (`async4/6`) preserves operand-register lifetimes
  with explicit fences, but does not improve the measured whole FWD.
- Two-CTA bias multicast needs ownership barriers for the entire projection/bias scratch
  union as well as per-slot reader barriers. `cluster2` omitted the former and failed L768
  finite-output checking. Corrected `cluster2s` passes numerical pilots but is slower.

The selected production-length kernels still use 80 registers with 80-byte stack frames at
L768/1024 and 112 bytes at L384. Removing named global QKV tensors does not remove register
spill traffic. NCU at L768 measures less time for `hot6t` despite slightly more HBM traffic;
the scalar control moves less HBM data and is slower. Whole-FWD paired timing, not nominal
memory savings or SOL alone, determines selection.

See [results](QKV_ALGORITHM_RESULTS.md) for the qualified timings and all rejected controls.
