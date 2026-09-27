import hashlib,json
from pathlib import Path
root=Path(__file__).parent
repo=root.parents[2]/'.engine-release-2.0.0'
rows=[]
for part in range(4):
    rows.extend(json.loads((root/f'selected-{part}.json').read_text()))
for part in (0,2):
    override=json.loads((root/f'selected-rms-{part}.json').read_text())[0]
    rows=[override if (r['M'],r['D'],r['rms'],r['dtype'])==(override['M'],override['D'],override['rms'],override['dtype']) else r for r in rows]
linear=json.loads((root/'selected-linear.json').read_text())
(root/'results.json').write_text(json.dumps({'norm':rows,'linear':linear},indent=2))
files=[*sorted((repo/'src/miniworld_engine/kernels/norm_cuda').glob('*.cu')),*sorted((repo/'src/miniworld_engine/kernels/norm_cuda').glob('*.py')),repo/'tests/numerics/test_norm_cuda_gpu.py',repo/'tests/numerics/test_norm_linear_cuda_gpu.py']
manifest={'baseline_sha256':hashlib.sha256((root/'baseline.cu').read_bytes()).hexdigest(),'sources':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'jobs':{'correctness_matrix':19231,'managed_tests_and_stress':19269,'norm_performance':19230,'rms_retune':19255,'linear_performance':19242,'sanitizers':19252,'ncu':19263},'production_dispatch_changed':False}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
text='''# Native CUDA norm optimization — 2026-09-26

## Status and comparison scope

LayerNorm, RMSNorm, and LayerNormLinear have native CUDA candidates with automatic
schedule selection inside the explicit `norm_cuda` APIs. Existing engine/module
production dispatch is unchanged. This report supersedes the first native norm
candidate report for performance selection.

**Old CUDA below means the pre-optimization native candidate (v4), not engine
1.0.0 or the whole MiniWorld model.** Its launch configuration was re-tuned for
this comparison. `Engine` means the current installed-source dispatch called by
`layernorm_kernel` / `triton_rmsnorm`, with FP32 affine parameters. Those calls
may use different CUDA/Triton paths by dtype.

H100, CUDA graph replay, median of seven event measurements, ten complete calls
per replay. Training means forward plus all input/affine parameter gradients.
Tables use **milliseconds**. Contiguous inputs are used for timing; copies for
noncontiguous inputs are validated but these tables do not price those copies.
Triton baselines use the normal 24-candidate cache-miss search, **not a fully built
autotune cache**. PyTorch FP32-accumulation formula timings remain in JSON and are
not presented as native PyTorch module latency.

## Implementation

1. Explicit float reciprocal square root and packed loads/stores. Common widths
   specialize the exact register tile; D384/D768 no longer reserve padded columns.
2. D64/D128 use eight threads per row. Affine gradients are combined inside each
   CTA, reducing the partial buffer and the final reduction traffic.
3. D256/D384/D512 reuse CTA partial reduction across row warps when shared memory
   fits. Float64 D512 defaults to 128 threads to respect that footprint.
4. D1024..16384 use CTA-wide row reductions, avoiding large per-warp register
   arrays. Larger/unaligned/irregular cases retain the generic CUDA fallback.
5. LayerNormLinear has a native LN/WMMA forward with packed input/saved-activation
   loads, padded shared memory, shared weight tiles, and output-width-dependent
   warp assignment. Its backward keeps the rounded normalized activation,
   cuBLAS dX/dW, and the optimized native norm backward.
6. Auto LNLinear fusion is restricted to the measured low-precision
   M147456/K128/N16 cell. Other cells use native norm plus cuBLAS. Explicit fused
   calls remain available for experiments; unsupported shapes/dtypes compose.

Float16/BFloat16/Float32 statistics and affine gradients accumulate in Float32;
Float64 accumulates in Float64. Centered variance is preserved. RMS epsilon=None
uses input dtype epsilon. No global floating atomics are used in selected norm
paths. Partial gradient workspace is capped at 32 MiB. First-order gradients,
empty leading dimensions, mixed affine dtypes, and last-axis normalization are
supported. Higher-order CUDA gradients remain unsupported.

## Norm forward + backward

'''
for dtype in ('torch.bfloat16','torch.float16','torch.float32','torch.float64'):
    text+=f'### {dtype.replace("torch.", "", 1)}\n\n| Op | M | D | Old CUDA ms | New CUDA ms | Speedup | Engine ms | New / engine speedup | Schedule |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
    for r in rows:
        if r['dtype']!=dtype:continue
        t=r['times'];old=t['old_cuda']['train_us'];new=t['new_cuda']['train_us'];eng=t.get('engine_dispatch',{}).get('train_us');cfg=r['new_config']
        text+=f'| {"RMSNorm" if r["rms"] else "LayerNorm"} | {r["M"]} | {r["D"]} | {old/1000:.5f} | {new/1000:.5f} | {old/new:.2f}x | {eng/1000:.5f}' if eng is not None else f'| {"RMSNorm" if r["rms"] else "LayerNorm"} | {r["M"]} | {r["D"]} | {old/1000:.5f} | {new/1000:.5f} | {old/new:.2f}x | n/a'
        text+=f' | {eng/new:.2f}x' if eng is not None else ' | n/a'
        text+=f' | {cfg[0]}, t{cfg[1]}, r{cfg[2]} |\n'
    text+='\n'
text+='''The engine autotune shape key rejects widths >=4096. Engine Float64 timing is
omitted because its accumulation contract differs. A speedup below 1.0 means
that the new native candidate still loses that comparison. Small RMSNorm remains
launch/reduction sensitive, and several cells still favor the engine.

## LayerNormLinear

Old CUDA is independently re-tuned for the full composed workload. New uses the
public auto API. K=input width, N=projection output width.

| dtype | M | K→N | Old inference ms | New inference ms | Inference speedup | Old training ms | New training ms | Training speedup |
|---|---:|---|---:|---:|---:|---:|---:|---:|
'''
for r in linear:
    t=r['times'];old=t['old_cuda'];new=t['new_cuda_auto']
    text+=f'| {r["dtype"].replace("torch.", "", 1)} | {r["M"]} | {r["D"]}→{r["N"]} | {old["inference_us"]/1000:.5f} | {new["inference_us"]/1000:.5f} | {old["inference_us"]/new["inference_us"]:.2f}x | {old["train_us"]/1000:.5f} | {new["train_us"]/1000:.5f} | {old["train_us"]/new["train_us"]:.2f}x |\n'
text+='''
The native fused prototype is not universally faster than fused Triton. For the
BF16 M147456/K128/N16 inference cell, the new native path is approximately
0.031 ms; fused Triton is approximately 0.022 ms. Broad CUDA fusion promotion is
therefore not justified. Wider projections generally retain composition because
native WMMA fusion is slower. `linear-v1.json` through `linear-v5.json` preserve
all five iterations, including losses, with actual fused Triton/CuTe forward
measurements. Those fused-forward columns do not include backward or training
activation saves and must not be treated as full training comparisons.

## Validation and issues caught

- Job 19231: 432 shape/dtype/layout executions (216 per shard; some linear cases
  repeated across shards), output and all gradients; changed-input graph replay.
- Job 19269: 28 stress/FP64 gradcheck/compile checks and 93 managed GPU tests pass (final source).
- Jobs 19230/19255: 48 large-cell gradient checks against explicit accumulation
  math, then old/new paired timing. Job 19242: eight full LNLinear gradient and
  timing cells.
- A partial-row-group deadlock was found during broad validation. The eight-lane
  shuffle now uses the exact participating subgroup mask. Failed jobs 19198/19199
  were stopped; the corrected broad matrix passes. Frozen pre-fix candidates are
  research artifacts, not dispatch choices.
- Initial sanitizer runs reported a CUDA API invalid-handle at late module
  loading (`cuKernelGetFunction`), while numerical tests passed. Sanitizer
  initialization now preloads all extensions before autograd/graphs and uses
  `CUDA_MODULE_LOADING=EAGER`; no API-error suppression is used.
- Atomic affine accumulation was slower and rejected. Coalescing final reduction
  without enough CTAs was slower for narrow widths and rejected. Subgroup rows
  alone were insufficient; CTA-local affine reduction was necessary.

Sanitizer and final NCU completion details appear below.
No full-model training promotion, release commit, or remote push is included.

## Artifacts

Workspace: `/home/psk6950/MiniWorld/runs/norm_cuda_20260926/opt`.
`results.json` combines the final tables; `manifest.json` records source hashes
and job IDs. `baseline.cu` freezes the pre-optimization reference. Every rejected
experiment retains its own source/JSON/log. Full baseline formula timings,
error norms, old selected configurations, and individual forward timings remain
in JSON for audit.
'''
text+=(root/'completion_section.md').read_text()
(root/'REPORT.md').write_text(text)
(repo/'docs/development/norm-cuda-optimization-20260926.md').write_text(text)
