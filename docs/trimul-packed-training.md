# Packed bidirectional TriMul training — 2026-09-15

The installed H100 bidirectional training path now writes contraction results
directly into their final buffers. Two forward GEMMs fill one buffer; four
backward GEMMs fill two gradient buffers. This removes three pair-sized
concatenation copies while preserving the contraction and gradient equations.

## Final whole-training results

H100 80GB HBM3, PyTorch 2.10.0+cu128, BF16, batch 1, pair width 128,
hidden width 128 per direction. Training includes forward, backward, gradient
reset, dropout 0.25 and a mask with holes. All three paths use identical seeded
parameters/inputs and manual CUDA graphs in the same process/GPU. Each median
comes from 12 alternating-order rounds of 20 replays. Compile is explicitly
`dynamic=False`, `fullgraph=True`; compiler-owned CUDA graphs are disabled.

| L | Triton ms | Previous H100 ms | New H100 ms | vs Triton | vs previous H100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.31549 | 0.31544 | 0.29919 | 1.054x | 1.054x |
| 384 | 1.96634 | 1.94564 | 1.77399 | 1.108x | 1.097x |
| 768 | 7.49705 | 7.40165 | 6.76747 | 1.108x | 1.094x |

Jobs: 13085/13086/13087. The previous H100 control loads the preceding
`BidirBackHalf` implementation, using the same current helpers and native front
cache as the new path. Thus this final comparison isolates the contraction
change; it does not mix different front tuning configurations.

Explicit `dynamic=True`, L768 (job 13088): Triton **8.13145 ms**, previous H100
**8.99044 ms**, new H100 **6.76866 ms**. The new path is 1.328x the previous H100
path in this mode and essentially matches its own static result. This does not
establish that all dynamic shapes or an entire MiniWorld training step have
the same speed.

## Why it improves

The primary saving is memory movement. In the L768 static profiler, five
Inductor kernels named `fused_cat*` totaled about 0.598 ms previously; the new
path has two small remaining concatenations totaling about 0.005 ms. The
three large contraction copies are gone. Under dynamic compile the corresponding
previous total was about 2.219 ms, explaining the much larger improvement there.
These are single-step profiler observations; the paired graph timings above
are the performance measurements.

The initial search also compared packed cuBLAS, all six Quack GEMMs, six
individual replacements and a combined candidate. L768: packed cuBLAS took
6.80774 ms; all Quack took 6.77002 ms. L384: 1.78999 vs 1.77303 ms. L128:
0.29799 vs 0.29844 ms. Thus most of the gain is from removing copies, with a
small additional GEMM gain on the two larger measured shapes. This is a
measured candidate search, not an exhaustive proof of the fastest possible
algorithm or every Quack tile configuration.

## Runtime selection and installation

New engine file: `kernels/trimul_inproj/cute/contract.py`. It wraps the fixed
two-/four-GEMM launch sequences as opaque operations with fresh contiguous
outputs. The surrounding existing autograd function owns the backward formula.

- BF16 contiguous `[256,384,384]` and `[256,768,768]` operands, 128 channels
  per direction, on the measured H100 80GB HBM3: Quack/CuTe for all six GEMMs.
- L128, other shapes/layouts/dtypes/cards: cuBLAS. The builder's CuTe GPU
  policy is still checked before importing or launching Quack.
- Backend selection runs inside the real opaque launch. It does not depend
  on `torch.compiler.is_compiling()`, fake-tensor strides, a warmed Python
  dispatch cache, or runtime benchmarking. Logged final calls confirm all six
  Quack contractions for L384/L768, including dynamic L768; L128 uses cuBLAS.
- This replaces the bidirectional **contraction** dispatch. The existing
  generic dispatch for weight and input gradients remains separate. In compiled
  training those still use cuBLAS; huge-K weight reductions strongly favored
  cuBLAS in the preceding component audit.

`patches/miniworld-engine-trimul-packed-training.patch` is the sixth patch in
`scripts/apply_engine_audit_patches.py`. It includes the two source changes and
the remeasured native cache. MiniWorld and team-gm contain matching copies.

Changing kernel source changes the native cache identity. Jobs 13080/13081/13082
therefore remeasured all 18 declared front configurations for each of the 9
inference/single-training/bidirectional-training keys: **162 validated pairs**.
The merged cache retains the top 5 per key. No old measurements were relabeled.
All final benchmark native selections hit; no cache misses were recorded.

Final native source identity:
`998c938247b33f7711689559fc005de9c395f82037f39234634ba1333b6c62fc`.

## Verification

- **61 GPU tests passed**, job 13084: the preceding 53 mask/native/compile
  cases plus 8 packed-path cases. These cover forward and all six contraction
  formulas against independent FP32 autograd, odd/unmeasured shapes,
  noncontiguous fallback, input immutability and output non-aliasing.
- Cold static and dynamic fullgraph training match the FP32 module reference
  for output, input gradient and every parameter gradient at L384.
- Compiled forward/backward CUDA graph replay advances dropout RNG. Changing
  the mask to all-invalid preserves the residual output and input gradient
  exactly. Output LayerNorm bias retains its mathematically live gradient;
  masking the front must not incorrectly mask that bias derivative.
- Direct contraction reference tolerance is relative L2 < 0.004; whole-module
  BF16-to-FP32 tolerance is < 0.025. These are tensor-norm tolerances, not
  claims of bitwise equality or a 0.01% maximum-element error bound.
- **4 CPU patch-stack tests passed**. A clean archive of pinned commit
  `1bc0803e3b2fef3b963fdc383e090c0adcccdb43` plus all six patches reproduces
  the installed source identity and all 9 native entries. Reapplying is a no-op.

An initial test-only import failure was fixed. A second test incorrectly
required every all-invalid parameter gradient to vanish, including output
LayerNorm bias; that assertion was corrected using the original unmasked
output-LayerNorm equation. Neither failure required changing the kernel math.

## Reproduction and evidence

Run GPU commands inside an H100 Slurm allocation using the cu128 environment,
the installed MathDx path, and the standard project CUDA library environment.

```bash
python scripts/apply_engine_audit_patches.py
python scripts/benchmark_trimul_packed_training.py --length 768 --dynamic false \
  --baseline-source runs/trimul_dispatch_search/before/miniworld_engine/kernels/trimul_inproj/cute/bidir_training.py \
  --output runs/trimul_dispatch_search/recheck_L768.json
python -m pytest -q tests/test_engine_trimul_packed.py
```

For a new checkout, obtain the pinned engine source, apply the first five
`PATCH_NAMES` using `apply_patches(package, patches, names=PATCH_NAMES[:5])`,
and pass its `bidir_training.py` as `--baseline-source`. The benchmark imports
only that back-half class while retaining the final installed helpers/cache.
`search_trimul_training_dispatch.py` takes the same baseline argument and
repeats the nine candidate comparisons.

Evidence directory: `runs/trimul_dispatch_search/`. `summary.json` contains
the final table, search results, validation totals and baseline SHA-256.
`final_L*.json` retain paired samples, kernel summaries, selected backend calls
and cache observations; separate `*_trace.json` files contain profiler events.
`tuning/L*/` retain all 162 configuration measurements and numerical checks.
`reinstall_verification.json` records the clean-install check. Search jobs
13076/13077/13078 and final jobs 13084–13088 have logs under
`runs/v1.0.1/phase2b/`.
