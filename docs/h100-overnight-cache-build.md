# H100 overnight cache build — 2026-09-15

## Applied before the build

The Triton bidirectional training back-half now writes two forward GEMMs into
one final buffer and four backward GEMMs into two final gradient buffers.
`triton/contract.py` exposes these fixed cuBLAS launch sequences as opaque ops;
the existing autograd function retains the gradient equations. The training
path no longer concatenates the three pair-sized contraction outputs.
Inference is unchanged by this seventh patch.

Patch: `patches/miniworld-engine-triton-packed-training.patch`. MiniWorld and
team-gm have matching installer, patch and test copies. The original source
checkout outside MiniWorld was not edited.

Validation job 13089: **16 GPU tests passed**, including both implementations,
independent FP32 contraction/autograd references, all parameter gradients,
cold static/dynamic fullgraph compilation and dropout/residual graph replay.
The four patch-stack tests and four cache-publication tests passed. A clean
archive of pinned engine commit `1bc0803e3b2fef3b963fdc383e090c0adcccdb43`
plus the final eight patches reproduces the installed source and all 73 baseline H100
cache files. Reapplying the stack is a no-op.

Before the full cache rebuild, paired static training measurements were:

| L | Previous Triton ms | Packed Triton ms | Packed H100 ms |
| --- | ---: | ---: | ---: |
| 128 | 0.31486 | 0.29838 | 0.29840 |
| 384 | 1.97728 | 1.82219 | 1.77431 |
| 768 | 7.64304 | 7.03010 | 6.78907 |

Jobs 13090/13091/13092, H100 BF16, batch1, D128, dropout0.25 and holed masks;
forward/backward/gradient reset, fullgraph `dynamic=False`, paired manual
CUDA graphs. These are **pre-rebuild** numbers: changing source invalidated
native cache identity and the probe reported cache fallbacks. They demonstrate
the copy reduction within this run, not the final tuned H100/Triton comparison.
Detailed samples and profiler traces are under `runs/trimul_triton_packed/`.

## Preflight repair before the overnight restart

The first build, job 13093, successfully derived 5,750 invocations but its real
module preflight found a bias-free LayerNorm fallback failure: BF16 activations
with FP32 affine weights were passed directly to CUDA PyTorch LayerNorm. This
blocked the module sweep while allowing alternative driver units to run. That
job was cancelled; its logs/shards remain in `runs/h100_build_20260915/`.

The eighth patch, `miniworld-engine-layernorm-mixed-dtype.patch`, fixes the
standalone PyTorch fallback to calculate low-precision inputs in FP32 and
restore the activation dtype. FP32 affine gradients remain connected. This is
a dtype correctness repair, not a cache-key relabel. Job 13095 passed **20 GPU
tests** (the 16 packed-path cases and 4 mixed-norm cases), and the entire
module preflight completed with `ALL MODULE PREFLIGHT: []`. Restart job
**13096** started successfully after that validation via `afterok:13095`.

## Overnight job

- Slurm job **13096**, `mw-h100-cache`, node02.
- **7 H100 80GB GPUs**, 112 CPUs, 768 GiB requested memory, 24-hour limit.
- Started at 2026-09-15T12:24:07.309019+00:00 after job 13095 succeeded.
- Command: `python -m miniworld_engine.cli build all --gpus all`, default
  incremental/resume mode; 12 compile workers per unit, one unit per GPU,
  two-hour per-unit timeout, compiled Triton cache retained.
- Exact script: `scripts/submit_h100_cache_build.sbatch`.
- Source snapshot: `runs/h100_build_20260915_retry1/package/miniworld_engine`.
- Native source identity:
  `9994de3e8563aab036f7b099478cd0f428f8d80359c9cf8bfaf9cac0c2845363`.

The source snapshot isolates an overnight build from subsequent source edits.
The CLI regenerates and verifies its SM90 module plan before tuning, merges
successful shards and checks coverage. The initial plan contains 53 module
rows / 5,750 invocations to record. A running Slurm allocation alone is not
a declaration that tuning or cache coverage has completed.

## Results and continuation

All paths below are relative to `runs/h100_build_20260915_retry1/`:

| Path | Purpose |
| --- | --- |
| `build-13096.log` | Main build output, preflight, worker outcomes and final coverage |
| `job_started.json` | Confirmed allocated devices and source identity |
| `shards/` | Per-unit logs, measurements, completion markers and resume data |
| `plans/` | Source-specific SM90 derivation and its verification evidence |
| `package/miniworld_engine/autotune/data/` | Merged measured cache in the snapshot |
| `publication.json` | Created after the CLI exits; publication result and complete/partial state |
| `job_exit.json` | Written by the job exit trap with exit code and finish time |
| `baseline_data/` | Original H100 files for concurrent-edit checks and reproducible diffs |

`publish_engine_build_cache.py` runs after the build. If snapshot and installed
sources still match and nobody changed the original cache or patch stack, it
publishes valid changed H100 files to the installed engine using an additional
`miniworld-engine-h100-cache-build-20260915.patch`, then adds that patch to
both MiniWorld and team-gm installers. Publication itself is checked for
idempotence. A partial build can publish its valid measurements, but is recorded
as **partial**, not complete. Concurrent source/cache edits stop publication
and leave the measured snapshot/shards intact.

Morning checks:

```bash
squeue -j 13096
tail -n 80 runs/h100_build_20260915_retry1/build-13096.log
cat runs/h100_build_20260915_retry1/publication.json
cat runs/h100_build_20260915_retry1/job_exit.json
```

The final two files may not exist while their stages are still running. On a
failed or timed-out build, keep the snapshot and `shards/`; the same CLI command
and environment resume completed units. Do not reset the cache or discard
successful shards merely because some units failed. If a partial cache has
already been published, a later publication needs a new baseline/patch name;
the first publication patch is intentionally not overwritten.
