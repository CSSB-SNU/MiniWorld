# CuTe configuration tuning

## Scope

This update broadens the custom SM90 CuTe epilogues and gives all 14 native build
operations persistent, incremental tuning. Generic Quack GEMM keeps Quack's own
autotuner; the measured cuBLAS/Quack contraction backend policy is unchanged.

Candidate definitions live in `miniworld_engine/autotune/cute_config.py`.
`native.candidates_for(op, bucket)` exposes the exact declared grid without a GPU.
Run `python scripts/report_native_search.py` to inspect tiles and scheduling axes.

- Gated GEMM: 18 -> 448 candidates; 112 distinct compile contracts per workload.
- Plain GEMM: 22 -> 512 candidates; 128 distinct compile contracts.
- LN backward: D128 6 -> 48, D192 4 -> 32, D256 2 -> 16.
- M2: 23 -> 240 candidates, static scheduler only; 60 distinct compile contracts.
  The 16 M192/N128 pingpong variants failed the custom stats/gate epilogue and
  are excluded. That tile remains enabled for the other epilogues that passed.
- TM2: still four `tile_m` values before shared-memory pruning. Other fields are
  not consumed by this kernel and are not presented as tunable axes.

For ordinary GEMM epilogues, the expanded space combines existing cooperative and
pingpong tiles with five smaller pingpong tiles, clusters `(1,1)/(1,2)/(2,1)/(2,2)`,
static/dynamic scheduling and maximum swizzle `1/2/4/8`. LN backward fixes
`cluster_n=1` and full-width N reduction. Unsupported operand swapping, tile-K,
warp counts and gather modes remain excluded.

Swizzle changes the runtime scheduler argument. CPU precompilation therefore
compiles one binary contract for its four swizzle values, while performance tuning
still measures each value separately.

## Measurement and cache contract

- Keys include exact shape, stride, dtype, optional operands/epilogue values, GPU,
  compiler environment, implementation identity and benchmark budget.
- Changing only the CuTe search grid preserves compatible timings. Build generation
  includes the policy fingerprint so `--resume` reopens units after a grid edit.
- Every successful native timing is retained in the versioned JSON workload's
  `timings`. `entries` stays a bounded top-K for runtime use.
- Removing a winner can promote a previously measured runner-up. Adding a candidate
  measures only the missing candidate; removed candidates retain their history.
- Benchmark profiles are separate. `native_runtime_profiles` explicitly selects the
  latest published native profile, rather than choosing an arbitrary profile's winner.
- Per-candidate atomic journals under the build's round-cache directory survive
  process interruption. A workload lock prevents duplicate concurrent tuning.
- Deterministic compiler rejections count as searched. Timeouts, killed compilers,
  unknown errors and launch failures remain retryable and prevent a unit from being
  marked complete. A poisoned CUDA context aborts the unit.
- `--rebuild` bypasses reuse. A changed code/environment identity causes retuning.
- Native `cache-status` reports missing candidate/workload coverage after a grid edit.

Implementation identities intentionally remain conservative across the native
kernel tree. This update separates CuTe grid edits from code invalidation; it does
not claim fully independent per-kernel dependency hashes.

## Integration with the running build

Job 13096 uses the unchanged eight-patch snapshot in
`runs/h100_build_20260915_retry1/package`. The new source is developed and tested in
`runs/native_tuning_dev/package` and packaged as
`patches/miniworld-engine-native-tuning.patch`.

`activate_native_tuning.py` waits for the previous build's exit marker, validates
the exact source and patch hashes, then appends the new patch to both root and
team-gm installers. It leaves Triton cache files byte-identical. Old native runtime
winners can be preserved only when the full kernel/launcher tree is byte-identical
and the old identity matches the reviewed baseline. The compatibility patch records
that migration explicitly; old timings remain **legacy-unattributed**, never
claimed as measurements of the new benchmark profile or expanded search space.

The enlarged production shape grid requires a subsequent incremental native build.
Passing representative correctness tests does not mean that all production cache
entries have been measured over the enlarged grid.

## Verification

CPU regression tests cover cache/shard merging, complete timing retention, grid
addition/removal, interruption recovery, benchmark-profile isolation, rebuild,
timeout retry and deferred activation/reinstallation.

H100 qualification scripts compare each candidate with FP32 formulas using row
tails, BF16 operands, masked rows, saved preactivations, a fused output gate and
residual addition. Results live under `runs/native_tuning_dev/qualification/`.
The eight expanded epilogues have **2,800 candidate/workload correctness checks**
across representative shapes (M=264, generally K=128/N=256; squeeze K=2048/N=512, plus both LN backward paths at K=192/256).
The maximum relative L2 difference from FP32 formulas was 0.0030. This is a kernel
BF16-vs-FP32 check, not a whole-model output or end-to-end speed measurement.

The excluded M2 family had approximately 0.81 relative L2 error on the same tail
shape. Raw evidence, including those rejected candidates, remains in `linear.json`.
`verify_native_qualification.py` checks that **every currently shipped candidate**
of these eight epilogues is present among the passing rows.

CPU: 31 regression tests passed. GPU: 20 regression tests passed, including
fullgraph compilation, backward and CUDA graph execution. The pinned source plus nine source patches installs
cleanly, and applying the complete stack again changes zero files.

Final GPU regression/extra-width job: **13102**, completed successfully.
The final real tuning round measured 448 candidates; the warm round reused all 448
and measured zero. Expanded candidate list construction is memoized (about
4.28 ms -> 0.0008 ms per call on the development host); callers receive a fresh
list and cannot mutate the cached grid. This measures list construction only.
Deferred activation job: **13103**, depending on `afterany:13096,afterok:13102`.
The activation additionally checks the final validation identity and the prior
build's exit marker. Status is written to `runs/native_tuning_dev/activation.json`.
The current eight-patch installation remains unchanged until that dependency clears.

Before deferred activation, run the new CPU tests against the staged package:

```bash
PYTHONPATH="$PWD/runs/native_tuning_dev/package" .pixi/envs/cu128/bin/python -m pytest -q tests/test_engine_native_tuning.py tests/test_engine_native_activation.py
```

After activation, the ordinary installed package supplies these modules.
