# H100 bidirectional TriMul output training optimization

## Scope

The new path targets BF16 bidirectional TriMul on H100 80GB HBM3, batch 1,
pair width 128, hidden width 128 per direction, L384/L768. L128 retains the
previous output path: the first whole-module comparison regressed there.
Other widths, shapes, dtypes and devices retain the existing implementation.

The production candidate uses Triton to compute and save normalized values
without the affine transform, cuBLAS for the forward projection and weight
gradient, and a new CuTe GEMM epilogue for the projection input gradient plus
LayerNorm backward. It does not make the complete output stage a single kernel.

## Algorithm

Let `xhat = (x - mean) * rstd`, with LayerNorm parameters `gamma, beta` and
projection weight `W` of shape `[128,256]`.

Forward folds the affine transform into the small weight and bias tensors:

```
y = xhat @ (W * gamma).T + W @ beta
```

Backward derives the two LayerNorm row corrections from the saved projection:

```
c1 = sum(dy * (y - W @ beta), output_channels) / 256
c2 = sum(dy * sum(W * gamma, input_channels), output_channels) / 256
dx = rstd * ((dy @ W) * gamma - c2 - xhat * c1)
```

The final line is the new CuTe dgrad GEMM epilogue. External row corrections
remove the old full-width epilogue reduction requirement, so independent
output-column tiles and cooperative GEMM configurations are valid. The output
is written directly in the channel-major layout required by the contractions.
The full `dx_normed` buffer is no longer written and reread.

The weight and affine gradients use one cuBLAS `T = dy.T @ xhat` plus small
reductions. No division by `gamma` is used; zero gamma values remain supported.
BF16 rounding points differ from the preceding implementation. FP32 reference
comparisons qualify the numerical difference; this is not bitwise equivalence
or a full-model convergence study.

## Alternatives measured

Initial output-stage forward+backward at L384, same inputs and compiled graphs:

| Candidate | ms |
| --- | ---: |
| Existing materialized LN + cuBLAS path | 0.3255 |
| Existing M1 + full-width CuTe LN backward | 0.5409 |
| M1 + existing backward with activation recomputation | 0.3919 |

A prototype added statistics output to M2. M2 plus the projection-aware backward
reached 0.3076 ms in a subsequent paired run (baseline 0.3254 ms). Saving normalized
values once, instead of recomputing them for backward, further improved this to
0.2964 ms (paired baseline 0.3249 ms). The M2 and full-width reduction prototypes
remain in `runs/trimul_output_opt_20260916/prototypes/`; they are not installed by
the production patch.

## Whole-module measurements

H100 80GB, BF16, batch 1, pair width 128, hidden width 128 per direction,
dropout 0.25, holed masks. Includes forward, backward and gradient reset;
excludes optimizer update. Fixed shapes with `torch.compile(fullgraph=True,
dynamic=False)`, compiler CUDA graphs disabled, paired manual CUDA graphs.
Medians use 12 alternating-order rounds of 20 replays on the same GPU.

First full-module comparison, before the final source cache build:

| L | Triton ms | Previous H100 ms | Candidate H100 ms | Decision |
| --- | ---: | ---: | ---: | --- |
| 128 | 0.29859 | 0.29875 | 0.30742 | Keep previous output path |
| 384 | 1.81300 | 1.77740 | 1.76348 | Enable candidate |
| 768 | 6.88290 | 6.78875 | 6.55796 | Enable candidate |

The input-projection native cache was stale in these runs; both H100 controls
used the same declared front configuration. This isolates the output change,
but is not a claim that every H100 kernel has been fully retuned.
Final cache-qualified measurements (new output native selections asserted):

| L | Triton ms | Previous H100 ms | New H100 ms | Reduction vs previous H100 |
| --- | ---: | ---: | ---: | ---: |
| 384 | 1.80942 | 1.77478 | 1.75814 | 0.94% |
| 768 | 6.89791 | 6.78962 | 6.55696 | 3.43% |

These are `verified_final_L*.json`. Earlier `final_L*.json` and
`cache_hit_final_L*.json` are exploratory runs; their recorded selections show
output cache misses despite the filenames. L128 retains the prior output path.

## Configuration management and validation

The new native op is `trimul_output_bwd_rows_sm90_cute`, with the full 512-candidate
plain CuTe GEMM space. It has a registry entry, build driver, source-sensitive
exact operand keys, CPU precompile contract and per-candidate measurements.
Its graph timing profile is explicit and distinct from the ordinary event-based
native builder profile. An ordinary build can therefore measure its own profile
without attributing these graph timings to another benchmark.

Evidence resides in `runs/trimul_output_opt_20260916/`:

- `rows_tune_L*.json`: exploratory per-candidate correctness and timings.
- `runtime_cache_L*.json`, `runtime_cache_L*.shard.json`: final source measurements
  and cache shards, 512 measured configurations for each actual runtime key.
  Dtype, shape and stride are checked against the compiled module trace first.
- `tasks/39_runtime_tests.log`: 25 passed; FP32 references, shifted/zero inputs, zero gamma,
  nonzero beta, tail rows, all parameter/input gradients, cold static/dynamic
  compilation, and mask/dropout/residual graph replay.
- `compile_contract.json`: isolated CPU builder contracts.
- `tasks/23_memcheck.log`: CUDA memory checking; these runs never publish timings.
- `verified_final_L*.json`: paired whole-module timing, profiler evidence and
  required native output cache hits.
- `source_patch_validation.json`: complete source/cache patch installation and
  idempotent reinstall in an isolated copy.

The launcher canonicalizes gamma to contiguous FP32 before cache selection.
Saved normalized activations are row-major. Earlier measurements with a
channel-major activation have separate cache keys and do not establish coverage
for production operands. No timings are relabeled across source IDs or layouts.

The maximum recorded standalone FP32-reference relative L2 error is approximately
0.304% (gamma gradient). This is a tensor-norm metric, not a maximum elementwise
error or a full-model accuracy guarantee. The memory checker reported zero
errors in four direct cases. Six CPU compile contracts passed. Targeted syntax
and undefined-name lint checks pass; the repository's strict style rules are not
fully satisfied by the experiment scripts.

## Installation while the Triton cache build is running

The 7-GPU Triton cache build uses an immutable snapshot and verifies the installed
source and patch stack before publishing. Changing either early would reject
that publication. Therefore this optimization is developed and measured in a
separate package snapshot, with only one H100 allocation (job 13182).

`scripts/activate_trimul_output.py` installs the reviewed source/cache patch after
the existing writer exits. It checks source and patch hashes, preserves newly
published unrelated caches, updates both MiniWorld and team-gm installers, and
verifies an idempotent reinstall. It never relabels old native timings with the
new source identity. Activation status is recorded in `activation.json` in the run.

Activation is scheduled as CPU-only job **13183**, dependent on cache job
13181 exiting and optimization job 13182 succeeding. At this report snapshot
the installed package still uses the previous output path.
