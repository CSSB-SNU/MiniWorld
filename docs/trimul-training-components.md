# TriMul bidirectional training: corresponding kernels and compiler effects

> Follow-up: [packed training results](trimul-packed-training.md) replace the
> contraction concatenations and qualify the new installed H100 dispatch.

Measured 2026-09-15 on H100 80GB. **The native CuTe input projection is faster
than Triton. The previous roughly 12% whole-step loss is reproducible under
automatic/dynamic shape compilation, but not under the inspected production
training script's explicit `dynamic=False` policy.**

No production kernel, dispatch policy, or published tuning cache was changed.
The benchmark-only hybrid below substitutes the input projection to isolate
its contribution; it is not an installed backend.

## Conditions and scope

- Bidirectional TriMul, BF16 activations/linear weights, FP32 norm parameters,
  batch 1, input width 128 and per-side hidden width 256 (two directions).
- Holed pair mask; full training uses row-broadcast dropout 0.25 and residual.
- Full step includes forward, backward, and gradient-buffer reset, without an
  optimizer update. This is not a whole-model training benchmark.
- Whole-step variants share seeded model/input values and execute in alternating
  order on the same GPU: 12 rounds, 20 CUDA graph replays per sample.
- Lengths run on separate GPUs. `dynamic=False` and `dynamic=True` experiments
  use separate allocations, but Triton/CuTe/hybrid within each are paired.
- Raw front comparison uses identical input, packed weights, and output buffers.
  Masks have identical values; Triton reads the production BF16 mask, while CuTe
  reads the normalized FP32 mask used by its native launch. Mask conversion,
  weight packing, and allocation are excluded from these raw kernel timings.
- Each of 18 native candidates was checked, then measured in alternating order
  for 12 rounds of 50 graph replays. Other isolated primitives use 12 x 30.
  Isolated hot-buffer timings must not be summed as predictions of whole-step time.

## Input projection: same fused work, different kernel

Both fuse left/right and both directions: projection, sigmoid gate, multiply,
pair masking, channel-major stores, and saved raw preactivations for backward.
Milliseconds, lower is better:

| L | Triton cached | CuTe installed | Best of 18 CuTe candidates | Triton / installed CuTe |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.026526 | 0.025690 | 0.025690 | 1.033x |
| 384 | 0.213417 | 0.194434 | 0.191175 | 1.098x |
| 768 | 0.866935 | 0.741821 | 0.741821 | 1.169x |

The installed native tile is the measured winner at L128/L768. L384 has about
1.7% isolated-kernel headroom in this sweep; this is not a 12% whole-step tuning
failure. All final component runs recorded no tuning-cache misses. The native
candidate outputs/preactivations matched the Triton BF16 buffers in this test;
maximum output relative L2 error against FP32 GEMM/GLU was 0.001659. Invalid
rows were exactly zero. These are tensor-norm checks, not elementwise error bounds.

## Other corresponding matrix operations, L768

These compare the SAME operands and strides. Both production compiled paths
currently choose cuBLAS for these operations; Quack/CuTe is the alternative
available to the H100 eager dispatcher. Thus a slow Quack candidate below is
not a claim that the compiled H100 path actually launches it.

| Operation | cuBLAS ms | Quack/CuTe ms | Finding |
| --- | ---: | ---: | --- |
| Outgoing contraction, forward | 0.18777 | 0.18122 | Quack slightly faster |
| Incoming contraction, forward | 0.19696 | 0.19307 | Close |
| Outgoing contraction, left gradient | 0.20387 | 0.18932 | Quack faster |
| Outgoing contraction, right gradient | 0.21072 | 0.20337 | Quack slightly faster |
| Incoming contraction, left gradient | 0.19779 | 0.19009 | Quack slightly faster |
| Incoming contraction, right gradient | 0.23688 | 0.21933 | Quack faster |
| Output-gate weight gradient | 0.11703 | 2.54185 | cuBLAS about 21.7x faster |
| Input-projection weight gradient | 0.48303 | 2.55021 | cuBLAS about 5.3x faster |
| Merged input gradient, two GEMMs with fused add | 0.60969 | 0.61053 | Essentially tied |

Contraction preferences vary with size: at L128 cuBLAS was faster on all six
contractions; L384 was close. The Quack candidates use their existing library
configuration policy; this experiment does not exhaustively retune Quack's GEMM
implementation. Matrix-variant outputs passed relative L2 comparison (<0.025).

## Shared kernels: no separate CuTe implementation to compare

Both training paths call these exact helpers. Isolated L768 time:

| Shared stage | ms |
| --- | ---: |
| Input LayerNorm | 0.10655 |
| Output LayerNorm + projection (multiple launches) | 0.37384 |
| Output LayerNorm + projection backward (multiple launches) | 0.89278 |
| Gate projection + sigmoid/multiply/dropout/residual | 0.35872 |
| Gate elementwise backward | 0.24774 |
| Input gate/mask backward and packed-gradient stores | 1.00413 |
| Contiguous forward concatenation | 0.20775 |

The approximately 1 ms packed-gradient kernel is a substantial shared cost.
It is not an H100-only regression. Dropout scale generation is outside the
gate epilogue; the shared-stage microbench uses unit scale, while the complete
training benchmark uses real randomized dropout and checks advancing RNG.

## Whole-step controlled comparison

Explicit `dynamic=False`, milliseconds:

| L | Triton | H100 CuTe | H100 wrapper with only front replaced by Triton |
| --- | ---: | ---: | ---: |
| 128 | 0.315567 | 0.315514 | 0.315129 |
| 384 | 1.965706 | 1.944760 | 1.957426 |
| 768 | 7.505859 | 7.413253 | 7.487991 |

CuTe is effectively tied at L128 and about 1.1–1.2% faster at L384/L768.
Replacing just its front with Triton increases L768 time by about 0.075 ms.
This intervention changes the front's implementation/output allocation contract;
it does not claim to isolate only instruction throughput.

L768 with explicit `dynamic=True`:

| Triton | H100 CuTe | H100 wrapper with Triton front |
| ---: | ---: | ---: |
| 8.136363 | 8.992441 | 7.481836 |

The same fusion boundaries can produce very different surrounding copy kernels.
In the recorded compiled-step traces, the sum of Inductor kernels named
`triton_poi_fused_cat*` is:

| Compile policy | Triton ms | H100 CuTe ms | Hybrid ms |
| --- | ---: | ---: | ---: |
| Static | 0.598 | 0.599 | 0.600 |
| Dynamic | 1.298 | 2.222 | 0.601 |

These are observational single-step profiler durations, not the paired graph
timings. They identify large concatenation/materialization overhead and explain
the direction and scale of the regression; exact attribution still needs
individual compiler-layout interventions. The hybrid shows that changing the
front's output contract also changes downstream compiler behavior, so its
dynamic speedup must not be labeled a faster Triton GEMM.

## Why the earlier benchmark showed H100 losing

`scripts/audit_trimul_tuning.py` used `torch.compile(fullgraph=True)` without an
explicit dynamic policy, then ran L128/L384/L768 in one process. Replaying that
unchanged audit (job 13068) reproduced H100 compiled training times of 0.31547,
2.36394, and 9.17708 ms. This is consistent with the earlier 9.159 ms result.

A second replay with `TORCH_LOGS=dynamic` (13072) explicitly records symbolic
dimensions being created for `pair.size()[1]`, `pair.size()[2]`, and `mask.size()[1]`
when training reaches L384. This confirms automatic dynamic-shape promotion,
rather than merely inferring it from timings. The inspected production entry
`scripts/run_miniworld_diffusion_train.py` lines 510/518 explicitly compiles with
`dynamic=False`. Earlier bare "compiled" tables therefore must not be treated
as its production compile-policy result.

## Validation and artifacts

- Final jobs 13064–13066: component comparisons and 54 native candidate checks.
- 13069–13071: static whole-step comparisons at all three lengths.
- 13067: explicit dynamic comparison. 13068/13072: old-harness reproduction/logs.
- All completed whole-step cases checked finite outputs and every parameter/input
  gradient, and verified graph replay advances dropout RNG. This run does not
  add a full-model or complete backward-against-reference accuracy claim.
- Initial exploratory jobs 13061–13063 finished whole-step measurements but
  later failed a microbench's flattened LayerNorm shape check. Their initial
  front test also used a non-production FP32 Triton mask key. Those component
  measurements are excluded; corrected final runs use production mask dtypes
  and preserve the four-dimensional LayerNorm input.
- Raw JSON, samples, selected configurations, compiler traces, and summary:
  `runs/trimul_training_components/`. Slurm logs:
  `runs/v1.0.1/phase2b/full_wiring_<job>.out`.

Reproduce on an allocated H100 using the project cu128 environment:

```bash
python scripts/benchmark_trimul_training_components.py --phase components --length 768 --output runs/trimul_training_components/repeat_components.json
python scripts/benchmark_trimul_training_components.py --phase whole --dynamic false --length 768 --output runs/trimul_training_components/repeat_static.json
python scripts/benchmark_trimul_training_components.py --phase whole --dynamic true --length 768 --output runs/trimul_training_components/repeat_dynamic.json
```

Next optimization target is the dynamic compiled layout/copy behavior, followed
by shared backward memory traffic. Broad native input-projection retuning is
not supported as the explanation for the earlier whole-step regression.
