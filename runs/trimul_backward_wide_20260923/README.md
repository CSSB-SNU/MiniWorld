# MiniWorld wide backward: first B1 candidate

2026-09-23. H100, BF16, B=1, H=2D, fixed dropout 25%, residual included.
This is an explicit development experiment, not engine dispatch or autograd integration.
D256 and D384 improve; retain the existing D512 B1. B7 is unchanged.

## D128 forward comparison

The existing D128 inference optimization reduced original-kernel time by 12.3%
(L384: 280.2 → 245.6 us) and 10.3% (L768: 1118.3 → 1003.2 us).
See `.engine-release-2.0.0/docs/operations/anthropic-trimul-payload.md`.
This is a different baseline and scope from the new wide training forward's
10–19% reduction versus the previous MiniWorld CUDA implementation.

Existing saved/dropout training-forward records in
`runs/trimul_cuda_widths_opt_20260923/latest-D128-L{384,768}.json` show
292.896 / 1164.368 us for CUDA and 555.872 / 2181.216 us for Triton.
Thus D128 is 1.90x / 1.87x faster than that Triton baseline; wide forward is
1.23–1.39x in its own recorded sessions. These are not a fresh matched-session
D128-versus-wide experiment or evidence of identical hardware efficiency.

## Change and tensor lifetimes

`prepare.cu` applies our wide forward output kernel to B1 recomputation:
channel-first triangle → on-chip transpose → output LN → streamed projection
and gate products → dProjection/dGate. Projection reads the shared LN tile.
The normalized triangle is still written once because dWproj consumes it later.
Mean and reciprocal standard deviation are saved for output LN backward.
This removes projection's global normalized-triangle rereads, not all B1
workspace or all activation traffic.

`finish.cu` retains the existing dNorm, split-K dW, output LN backward and weight
reductions, with its original norm/projection/gate preparation removed.
The separate launch provides ordering between preparation and its consumers.
Both kernels use TMA/WGMMA. The finish cooperative grid is bounded by occupancy.

`plan.B1(training)` borrows an already prepared `h100_width.Training` plan.
Its dy pointer is fixed at construction; do not call `training.bind_dy` and
reuse this B1 object. Construct a new B1 plan when bindings change. `tail`
executes the existing contraction gradients and B7. The check uses saves from
our new `h100_wide_forward.Forward`, so measured gradients exercise their
compatibility with the existing backward layout.

## Results

Same-process alternating CUDA graph measurements, 150 samples per path.
Both full paths use the SAME new forward; the comparison isolates the B1 change.
Times exclude allocation, compilation, CPU dispatch, and RNG.

| D | L | B1 old → candidate (ms) | Entire backward old → candidate (ms) | Fwd+bwd old → candidate (ms) | Decision |
|---:|---:|---:|---:|---:|---|
| 256 | 384 | 1.935 → 1.803 | 4.302 → 4.166 | 5.238 → 5.104 | retain candidate |
| 256 | 768 | 7.984 → 7.275 | 17.518 → 16.794 | 21.220 → 20.474 | retain candidate |
| 384 | 384 | 3.180 → 3.038 | 7.585 → 7.472 | 9.409 → 9.224 | retain candidate |
| 384 | 768 | 13.185 → 12.369 | 31.379 → 30.543 | 38.496 → 37.633 | retain candidate |
| 512 | 384 | 4.440 → 4.696 | 11.180 → 11.416 | 14.026 → 14.306 | reject candidate |

All 11 gradients pass the existing relative-L2 < 0.01 criterion against compiled
PyTorch in all five cases; maximum error is 0.006552. Differences from the
existing backward on identical forward saves are below 0.001846.
The reference uses the established width fixture and seed; this is not an
exhaustive numerical test. Paired timing exercises CUDA graph capture/replay.

D512 preparation alone costs about 2.108 ms at L384, versus 2.503 ms for its
finish. Two, three, four, and five warpgroups were compared; four remained best,
but none rescued the end-to-end regression. The underlying cause has not been
established, so no D512 improvement is claimed and L768 was not promoted/tested.

Results and raw samples: `result-D*-L*.json`, `summary.json`.
Compute jobs: 16682 (L384), 16691 (L768), 16687 (phase timing),
16690 (D512 warpgroup sweep). Sanitizer job 16696 checks new preparation and
finish at D256/D384 L384: both memcheck runs report zero errors and both racecheck
runs report zero hazards/errors/warnings. See `memcheck-D*.log` / `racecheck-D*.log`.
Earlier sanitizer job 16693 failed at CLI filter parsing before running kernels.

## Remaining backward work

B1 still materializes norm, dProjection, dGate, and dNorm. Removing these
requires accounting for dWproj, dWgate, dTriangle, and the later input-gradient
consumer of dGate; the current candidate does not do that.

B7 still materializes the full four projection/gate derivative planes and
reads them for both dX and weight gradients. A tile/ring producer-consumer
design must bound workspace while ensuring both consumers finish before reuse.
This is the larger remaining structural change, especially for D512, and has
not been implemented in this first B1 experiment.

Reproduce: `sbatch runs/trimul_backward_wide_20260923/check.sbatch`;
use `--array=0-1 --export=ALL,LENGTH=768` for the retained L768 candidates.
Run `sanitize.sbatch` for the scoped memory/race checks.
