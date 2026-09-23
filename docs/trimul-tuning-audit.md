# TriMul tuning audit — 2026-09-15

> Follow-up: the native cache and driver gaps described below are fixed.
> [Applied H100 tuning](trimul-h100-tuning.md) verifies9/9 native keys and reports
> measured gains and remaining small-shape/compiled-path limitations.

## Verdict

TriMul is **not fully tuned** in the current installation. The existing Triton
pieces have valid, populated tuning entries for the tested shapes. The newly
added H100 masked input projection has no tuning file, and the current build
driver cannot produce most of its observed runtime keys. Compiled bidirectional
execution also bypasses backend calibration and has slower layout operations.

No production kernel, dispatch rule or persistent tuning cache was changed by
this audit. The previous upgrade tests established functionality and measured
speed gains; they did not establish completion of native tuning.

## Runtime verification

Three H100 80GB jobs ran in parallel: outgoing13040, incoming13042 and
bidirectional13041. All completed successfully. Each ran L128/384/768, width128,
batch1, in training and eval/no-grad modes, with and without torch.compile:
**36 complete cases**. Inputs and linear weights were BF16; masks had holes;
training used shared-row dropout0.25. All outputs and training gradients were
finite, and training CUDA graph replays advanced dropout randomness.

The audit observes the real cache selector return values without replacing their
candidates. Triton records are first-use lookup decisions, not a per-launch hit
rate: later calls can reuse an in-process tuner result without querying the file.
Native selections are also recorded, and every case has a Chrome GPU trace.

| Part | Observed status |
| --- | --- |
| Existing Triton kernels | 12 kernel IDs, 33 distinct dtype/bucket keys; **33 cache hits, no misses** |
| Triton measurement coverage | Matching implementation profiles record all current-grid candidates as searched for these33 keys; no current-grid additions remain unsearched |
| Triton validity | Runtime checks accepted grid, environment, key scheme, source and implementation fingerprints; static H100 scan also reported these files valid |
| New masked SM90 projection | 12 distinct native operand/save keys; **0 hits, all12 missing the cache file** |
| Native runtime fallback | Each lookup supplies18 candidate configurations; runtime selects the first declared configuration instead of measuring the18 |
| Backend calibration | Eager bidirectional training measures cuBLAS/Quack alternatives; the compiled branch directly prefers cuBLAS |

The Triton claim is limited to the measured widths/shapes/modes and current search
grid. A recorded searched candidate may have been rejected or failed compilation;
it does not mean every candidate produced a valid timing. Bucketed tuning also
does not prove the globally optimal schedule for every possible input.

The missing file is:

```text
.pixi/envs/cu128/lib/python3.10/site-packages/miniworld_engine/autotune/data/
  trimul_inproj_masked_sm90_cute/NVIDIA H100 80GB HBM3 (sm90).json
```

All observed native calls use the same declared fallback: tile_m256, tile_n128,
cluster_m1, cluster_n2, pingpongFalse, dynamicFalse, swap_abFalse, swizzle8.
The warning text says “full autotune grid,” but native.choose_config actually
returns the default on a cache miss when run_autotune=False. The audit checks the
selector itself rather than interpreting that warning as evidence of a sweep.

## Native build driver does not cover runtime

The actual masked_front_sm90 driver was invoked with meta tensors, replacing only
the kernel launch with a key recorder. At D128 and the same three row counts, its
emitted keys intersect **only3 of the12** observed native runtime keys. Those3
are the single-direction training keys shared by outgoing and incoming.

Two concrete mismatches explain the other9:

1. **Projection width:** inference runs a per-side [128,256] weight. The driver
   always constructs [D,4*hidden], with hidden=D or2D, so D128 emits widths512/1024
   even for save_preact=False. It never emits the inference weight shape256.
2. **Mask metadata:** the driver always uses a flat bool mask[M]. Inference
   presents bool[1,L,L] or bool[1,1,L,L]; bidirectional training presents BF16[M].
   masked_front converts all of these to contiguous FP32[1,M] before launching,
   but constructs its cache key from the **original** mask shape/dtype/stride.
   These equivalent launch masks therefore produce different tuning keys.

Merely running the existing native build cannot complete runtime coverage. The
launcher should key the actual normalized launch operands, or the driver must
cover every retained input form. It must also emit the real inference projection
width. After that, the relevant native configurations need actual measurements
and the runtime audit must demonstrate hits. No cache was relabeled or fabricated.

## Compiled bidirectional path needs separate attention

L768 complete graph times in milliseconds; training includes forward, backward
and gradient reset. These are sequential audit measurements on the same GPU per
family, not alternating paired benchmark rounds:

| Family | Eager inference | Compiled inference | Eager training | Compiled training |
| --- | ---: | ---: | ---: | ---: |
| Outgoing | 1.0366 | 1.0298 | 4.3366 | 4.4266 |
| Incoming | 1.0376 | 1.0345 | 4.3368 | 4.4190 |
| Bidirectional | 2.1049 | 2.6422 | 7.3527 | 9.1252 |

Eager bidirectional L768 calibration chose Quack for incoming forward contraction
and several contraction gradients, and CuTe for the merged input gradient.
The compile branch in trimul_inproj/cute/dispatch.py prefers cuBLAS instead of
consulting those winners. This is a distinct issue from missing tile caches.

The full timing gap is **not attributed solely to backend selection**. Traces
also show different concatenation/transposition/materialization kernels. For
example, compiled bidirectional inference spends about0.792ms in an Inductor
cat/transpose kernel; compiled training has cat/materialization kernels taking
about1.025ms and0.856ms. These need to be compared with the eager operator schedule
before assigning causality or promising a specific speedup from calibration.

## Work needed to finish TriMul

1. Align the new native launch key and build workloads, including save/no-save,
   per-side inference projection width and normalized mask representation.
2. Measure the masked projection's supported configurations at the runtime keys;
   publish only actual measured winners, then repeat the cache-hit audit.
3. Restore shape-aware compiled backend selection safely and address the
   bidirectional layout overhead, followed by paired complete-module timing.
4. Re-run existing mask/dropout/residual and compiled-gradient regression tests
   after any kernel/dispatch changes. This audit itself checked execution,
   finiteness and RNG advancement, not a new numerical reference comparison.

## Evidence and reproduction

Artifacts under runs/trimul_tuning_audit:

- outgoing_13040.json, incoming_13042.json, bidir_13041.json: cases, selector
  records, backend choices and actual kernel summaries.
- Corresponding *_trace.json: all36 GPU traces.
- runtime_cache_coverage.json: deduplicated keys and matching searched-grid data.
- native_driver_coverage.json: driver-emitted keys versus observed native keys.
- static_cache_status.json: source/environment status of installed H100 files.

GPU audit, once per family on separate allocated H100s:

```bash
python scripts/audit_trimul_tuning.py --family outgoing --output runs/trimul_tuning_audit/outgoing.json
python scripts/audit_trimul_tuning.py --family incoming --output runs/trimul_tuning_audit/incoming.json
python scripts/audit_trimul_tuning.py --family bidir --output runs/trimul_tuning_audit/bidir.json
```

CPU-only coverage and actual driver shape replay:

```bash
python scripts/summarize_trimul_tuning.py runs/trimul_tuning_audit/outgoing_13040.json runs/trimul_tuning_audit/incoming_13042.json runs/trimul_tuning_audit/bidir_13041.json --output-dir runs/trimul_tuning_audit
```

Relevant installed sources: autotune/cache.py::_cached_subset/select_config,
autotune/native.py::choose_config, kernels/trimul_inproj/cute/masked_front.py,
kernels/drivers/trimul_inproj.py::masked_front_sm90, and
kernels/trimul_inproj/cute/dispatch.py::pick.

## Explicit Triton backend timings

[Existing Triton TriMul benchmark](trimul-triton-benchmark.md) separately measures
implementation="triton" and verifies its actual kernels. The timings in this
audit are for the H100 auto route.
