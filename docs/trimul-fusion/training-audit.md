# Bidirectional TriMul training audit — 2026-09-16

## Measurement contract

The experiment calls the engine's unmodified
`bench_module_triangle_multiplication_bidirectional` function and uses its module
YAML. The runner is copied into the run directory so its `sys.path` modification
cannot silently import the older external engine checkout.

- H100 80GB HBM3; one GPU, no concurrent experiments in that allocation.
- B=1, L=384, d_pair=d_hidden=128, one bidirectional layer.
- BF16 input and linear weights; FP32 LayerNorm affine parameters.
- `mask_prob=0`, training row dropout p=0.25, residual enabled.
- `model.compile()` on both implementations, as in the runner. Its default is
  `dynamic=None`; only L=384 is evaluated in each fresh process.
- Whole forward and backward, including input and all parameter gradients.
  Optimizer, loss construction and data transfer are excluded.
- Five repetitions of the runner's 10ms warmup / 100ms measurement, median of
  the five reported medians. Compilation and tuning are excluded.
- `cudagraph=disabled` clears gradients to None per timed iteration.
  `cudagraph=manual` uses the runner's static gradient accumulation buffers.
  Compare implementations within each column; these two columns have different
  gradient-buffer handling as well as different launch overhead.

The runner passes `conf.dropout` only to the `miniworld` implementation; its
PyTorch and Triton constructors use the module default of 0.25. This experiment
explicitly sets `conf.dropout=0.25` so all three execute the same dropout workload.

The runner's built-in accuracy calculation draws independent dropout masks for
the implementation and the reference. Those raw accuracy fields are retained
for reproducibility but are **not correctness evidence**. The separate
`check_trimul_module_reference.py` check uses identical explicit dropout scales
and compares output, input gradient and every parameter gradient, including a
zero-scale case where only the identity residual must remain.

## Baseline before this audit's tuning

| Implementation | Compile, no manual graph | Compile + manual graph |
| --- | ---: | ---: |
| PyTorch | 4.0651 ms | 4.0219 ms |
| Triton after F567 and backward fusions | 1.8095 ms | 1.6753 ms |
| PyTorch / Triton | 2.247× | 2.401× |

These are new module-harness measurements. Earlier 1.79→1.71ms measurements
compared two Triton implementations and included explicit gradient resets;
they were not PyTorch comparisons.

## Fusion boundaries

The present boundaries are a sound baseline for this shape, but the measurements
do not prove a global optimum over all algorithms or shapes.

- F567 keeps output projection, gate projection, sigmoid, dropout and residual
  together. Their inputs are already materialized and both GEMMs share output
  tile coordinates.
- B9+B10 merges the two gradients of normalized input; B11+B12 merges input
  LayerNorm backward with the identity-residual gradient.
- The contraction is a full length-axis reduction. Fusing a consumer into its
  epilogue requires that the consumer see the completed reduction.
- B1's two derivatives feed multiple GEMMs. B7's packed derivative feeds both
  weight-gradient and input-gradient GEMMs. Removing these buffers requires
  recomputation or redesign of both consumers, and can sacrifice the existing
  fast GEMMs. Fewer launches alone is insufficient evidence of a win.
- B3 output-projection dgrad → B4 output-LN backward remains a possible research
  boundary. A GEMM output tile and a whole-feature LN reduction have different
  data requirements. It must be compared as a complete backward, with the same
  saved values and numerical precision.

## Profile findings

The baseline manual-graph trace contains 49 GPU kernels per training iteration,
including framework gradient accumulation and RNG work. Kernel durations below
are averages over three profiled iterations; unprofiled whole-module timings
above are the performance measurements.

| Stage | Time | Observation |
| --- | ---: | --- |
| F2 input projection + gates | 225.8 µs | Usable current-implementation cache was missing; grouped 2-D tiles tested below |
| F567 output epilogue | 109.8 µs | Already fused and tuned |
| B1 output gate derivative | 64.7 µs | Reads/writes roughly 189 MB |
| B4 output LN backward | 110.4 µs | Per-tile atomic parameter-gradient accumulation |
| B7 front gate derivative | 246.5 µs | Reads/writes roughly 755 MB, about 3.06 TB/s effective traffic rate |
| B8 packed input weight gradient GEMM | 128.7 µs | Large reduction, roughly 340 MB operand reads |
| B9+B10 input gradient | 134.8 µs | Two GEMMs and gradient sum already fused |
| B11+B12 input LN + residual gradient | 76.4 µs | Keeps the residual outside the LN derivative |

PyTorch's trace includes twelve large layout-copy kernels per iteration,
accounting for approximately 1.93ms in the profiled non-graph run. This is a
major source of the difference from the packed Triton path.

## F2 cache diagnosis

The cache file had a matching L384 training entry, but its implementation hash
was unusable. The kernel body's `op_identity` is unchanged (`46b51ccf88c7`).
Triton 3.6's `JITFunction.cache_key` also hashes the function's starting line:
the old function began on line 69; the current one begins on line 47.

Recomputing the current function's JIT hash with only that line number changed
reproduces the stored hash exactly. The runtime therefore searched a heuristic
24 of 864 configs despite the old results file existing. This audit rebuilds
measurements for the actual current implementation; it does not relabel old
measurements as new ones.

## Final L384 measurements

| Implementation | Compile, no manual graph | Compile + manual graph |
| --- | ---: | ---: |
| PyTorch, repeated baseline | 4.0611 ms | 4.0228 ms |
| Triton, corrected shape routing and rebuilt caches | 1.8703 ms | 1.6559 ms |
| PyTorch / Triton | **2.17×** | **2.43×** |
| H100 native path, before audit | 4.0828 ms | 1.7153 ms |
| H100 native path, after audit | 2.8928 ms | 1.7137 ms |

A separate alternating-order comparison retained all three compiled manual graphs
and replayed them in 12 rounds of 30 iterations. PyTorch: **4.053832 ms**;
Triton with previous routing: **1.682561 ms**; Triton with corrected routing:
**1.670700 ms**. PyTorch / corrected Triton is **2.426×**. The routing/cache
change itself improves this paired result by **0.71%**, not by 2.4×.
The separate non-graph runs do not establish an additional Triton speedup:
the initial 1.8095 ms run was faster than the final 1.8703 ms run.

The native H100 results use the currently declared defaults for
`trimul_inproj_masked_sm90_cute` and `trimul_output_bwd_rows_sm90_cute` because
their native caches fail the source/environment validation. Both before and
after runs have that limitation. They are not measurements of a fully tuned
CuTe implementation. The native CPU optimization reduces non-graph latency by
29.1% (1.41× throughput); graph replay changes by only 0.1%, as expected for a
host-side optimization. Native source identities remain strict: old results
are not relabeled and native caches need a matching build to become valid.

## Adopted changes

1. **Output-LN workload key**: bidirectional Triton and H100 wrappers now pass
   `both_key(M)` to output LayerNorm forward/backward, including FP32 and native
   fallback branches. Previously `None` selected the smallest row bucket even
   for large inputs. At L384/N256 the packed key changes from `2148533018` to
   `2473902211866`. The normalization math and fusion boundaries are unchanged.
2. **Current-implementation Triton cache**: rebuilt F2 for L384 and output-LN
   forward/backward for the affected shapes. Runtime checks confirm correct
   row keys and five cached candidates at L128/384/768.
3. **Lazy native config selection**: avoid constructing hundreds of kwargs and
   canonical dictionaries on every runtime launch. Misses convert only the
   default; valid caches reuse immutable candidate-membership signatures;
   actual builds still materialize and search the complete declared grid.
   Selected winners are not memoized. Cache publication, source/environment
   invalidation and grid narrowing are still checked on every selection.

No experimental GPU algorithm was substituted into the production L384 path.
The measured alternatives below did not improve its component times.

## Tuning and rejected alternatives

All times here are component CUDA-graph times, not whole training speedups.
Candidate errors compare the selected configuration with the previous result;
this does not mean every candidate underwent a numerical check.

| Existing operation | L | Searched / declared | Finite | Before → after |
| --- | ---: | ---: | ---: | ---: |
| F2 packed front | 384 | 864 / 864 | 854 | 0.25223 → 0.22400 ms |
| F4 output LN | 128 | 1440 / 1440 | 1434 | 0.006219 → 0.006398 ms |
| F4 output LN | 384 | 1440 / 1440 | 1434 | 0.060234 → 0.060254 ms |
| F4 output LN | 768 | 1440 / 1440 | 1434 | 0.232632 → 0.214699 ms |
| B4 atomic LN | 128 | 288 / 288 | 288 | 0.015813 → 0.016059 ms |
| B4 atomic LN | 384 | 288 / 288 | 288 | 0.111905 → 0.112223 ms |

The canonical persistent B4 search at L768 reached 384 of 400 configurations
before the 1800-second task limit; the final large tiles compiled too slowly.
Its measured checkpoint is retained with **partial coverage**, and its best
measured candidate is checked before publication. This is not a completed
400-config build or a claim that all engine caches are built.

| L384 alternative | Existing | Best candidate | Decision |
| --- | ---: | ---: | --- |
| Grouped 2-D front tiles, 48 configs | 0.22043 ms | 0.22744 ms | Keep current front |
| LN partial buffers + two final reductions, 48 configs | 0.11190 ms | 0.11551 ms | Keep atomic B4 |
| Canonical LN with 4/8 warps, 16 configs | atomic 0.11193 ms | 0.12285 ms | Keep atomic B4 |

The front alternative searches tile sizes, GROUP_M and warp choices; the
partial-buffer alternative includes its final reductions in timing. Separate
atomic/specialized-persistent/canonical screening at L384 measured
0.11487/0.12531/0.14523 ms, respectively, under the current cached/heuristic
choices. That screening is not an exhaustive search of all three spaces.

At L768 a 4/8-warp canonical screen found 0.46165 ms versus 0.54843 ms for
its then-current canonical choice; atomic measured 0.42909 ms. This is a
promising large-shape scheduling result, but was not promoted into the global
persistent grid or advertised as an L384 improvement. It needs managed-grid
expansion and a whole-module comparison against the recovered canonical
winner. No broad dispatch replacement was made from this small screen.

## Validation and reproduction

- One H100 allocation, Slurm job 13195. GPU experiments run sequentially.
- The unchanged benchmark runner snapshot has SHA-256
  `e0c1b2d34b60a5b4c36c87f9118bd2f0bd26eabb64c4346fd0bf170078d349ce`;
  the external checkout HEAD was `b4c3f6c34ea54d1f4053c68a9424a8193c617c4c`.
- Matched-dropout checks use 20% pair-mask holes, the same explicit p=0.25
  dropout scale, identical weights/input/upstream gradient, BF16 and FP32
  PyTorch references, all ten parameter gradients and the input gradient.
- L384 Triton worst relative L2 error across those tensors is 0.768% against
  BF16 PyTorch and 0.728% against FP32 PyTorch. The pre-audit baseline was
  0.769% and 0.727%, respectively. This is existing backend numerical
  variation, not a new 0.01% gradient-accuracy guarantee.
- Zero dropout scale gives exactly the identity output/input gradient and zero
  parameter gradients. L128/384/768 compile and graph runtime checks pass.
- H100 native matched-reference checks also pass at L128/384/768; the largest
  observed relative L2 error is 0.779% against BF16 PyTorch and 0.747% against
  FP32 PyTorch.
- L384 Triton compute-sanitizer memcheck reports **0 errors**.
- Native selection/build/cache publication regression checks: **27 passed**.

The archived `training-audit/` directory contains raw JSON results, the harness
snapshot and the sanitizer/test logs. Large Chrome traces stay in
`runs/trimul_training_audit_20260916/`. Raw runner accuracy fields are retained
with their independent-dropout warning; use the separate reference reports.

Example, from the project root on an allocated H100:

```sh
.pixi/envs/cu128/bin/python scripts/benchmark_trimul_module_audit.py \
  --run docs/trimul-fusion/training-audit --implementation triton \
  --graph disabled --output /tmp/trimul-triton.json
.pixi/envs/cu128/bin/python scripts/check_trimul_module_reference.py \
  --length 384 --implementation triton --output /tmp/trimul-reference.json
```

Use `--implementation pytorch` for the baseline and `--graph manual` for the
runner's graph setup. The hash-checked patch and activation manifest are in
`patches/miniworld-engine-trimul-training-audit.patch` and
`patches/trimul-training-audit-manifest.json`, mirrored into `libs/team-gm`.

### Installation

Activated in the local Pixi engine and added to both root/team-gm installer
stacks. Native source identity after activation:
`74b088cde6e546049a0018f7c3ff100fea6662d18928a791c0aa078d96eb97bc`.
Patch replay, idempotence, baseline patch hashes and all ten changed file hashes
were checked before activation. `activation.json` and `patch-validation.json`
record the exact files and identities. Existing GPU fusion diagrams remain
valid because these changes do not alter fusion boundaries.

Fresh processes using the installed package passed both L384 Triton and native
matched-reference checks. The installed native non-graph benchmark repeated at
**2.7952 ms**.
