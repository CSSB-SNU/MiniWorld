# DataLoader analysis and optimization — 2026-09-24

## Scope

Production Phase1a, PF16, L384, atom bucket4096, MSA pool8192/sample1024,
random recycle1–4, eight H100s, accumulation32. Preserve sampling order,
cropping policy, tensor values/dtypes, model/engine and existing W&B run
`team_gm/MiniWorld/tapgki9e`.

## CPU profile

128 real samples across four rank streams, CPU-only Slurm16918 on node02.
These are **CPU wall ms per sample in one worker**, not GPU latency or exposed
training stalls. Mean292.46ms; slowest1028.05ms.

| Stage | Mean ms |
|---|---:|
| CIF loading |23.90|
| Crop/tokenization |15.58|
| MSA loading/assembly |92.04|
| MSA sampling |7.01|
| Templates |26.99|
| Batch construction |20.56|
| Bucket collation |88.12|

Slow-sample profiles show compressed-data decoding, many small NumPy array
reconstructions and MSA preparation. Fixed output shapes do not make the
original structures or their CPU preparation costs equal across ranks.

## Changes

1. **Pad directly into the final batch.** The old implementation allocated a
   real dummy batch, concatenated it and sliced it away. The returned view
   retained its dummy storage. A meta-device schema now supplies the same
   shapes and dtype promotion, with one allocation for each padded field.
2. **Reopen the Arrow catalog in spawned workers.** The 8.75GB Arrow table
   previously participated in worker serialization. Workers now receive a
   254-byte path/file-identity payload and reopen the same mmap. Changed files
   are rejected. In-memory catalogs retain the original serialization fallback.
3. Eight workers per rank and prefetch8 become affordable without copying the
   catalog into each worker. Ordered DataLoader delivery is retained.

## Validation and CPU benchmark

- Nine unit tests passed: B1/B2, T0/1/4, exact legacy padding values/dtypes,
  sparse fields, catalog roundtrip/change rejection/in-memory fallback.
- CPU-only Slurm16920: 64 actual batches, old/new execution order alternated.
  Every field and dtype matches exactly, including NaN positions.
- Collation **95.11 →20.76ms**, **4.58x** on these batches.
- Example aligned-sequence backing storage **50,331,648 →25,165,824 bytes**.
- Four real spawn workers successfully produced64 batches; job MaxRSS~5.1GiB.
  This is the CPU benchmark footprint, not an eight-GPU training memory claim.
- Targeted F/E9 lint passed.

Artifacts: [profile](profile-summary.json), [collation](collate-benchmark.json),
[spawn test](spawned-loader.json), [baseline epochs](baseline-epochs.json),
[live timings](live-summary.json).

## Production handoff and current validation

Full epoch477/step47700 checkpoint saved before replacing job16917.
Model, Adam, scheduler and EMA preserved; checkpoint SHA256:
`e8c41c9f63b213a66c9a78a569f23d5851e63efcce1cc54707d22cd98a7cbaf9`.

Job16921 failed its strict CUDA graph startup gradient check; all loss values
matched exactly. Repeated ordinary execution also showed gradient differences,
so this is not evidence that the loader changed model inputs. Diagnostics are
preserved under `preflight-failure/`. Job16922 repeats validation with NCCL
Ring/Simple to isolate collective reduction variability. No tolerance was
relaxed and failed validation did not update training weights/checkpoints.

Completed old-loader CUDA-graph epochs472–477 averaged **111.63ms/microbatch**.
The optimized loader's end-to-end speed is pending successful validation and
live measurement. CPU collation speedup must not be reported as training speedup.
