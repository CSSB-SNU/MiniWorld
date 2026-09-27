# TriangleAttention backward: installed CUDA fusions

> **Current backward installation, 2026-09-25:** [Installed report](core90/README.md). Qualified `rs8_async_q4` dK/dV + bias and `rs_softmax_overlap` dQ pass installed-path job17211, including all input/parameter gradients. Complete backward improves21.92–23.23% and F+B14.63–16.00% versus job16663. Attribution17211 accounts for every16/17 backward kernels. SOL90 remains unmet. All new experiments use `node02 / normal_h100`.

> **Current installed backward:** [priority_pass/README.md](priority_pass/README.md), job16663, adds ldmatrix bias loads and CUDA/TMA shared-input Q/K/V/gate weight gradients. Whole backward improves another4.09–5.30% over16580; forward+backward also improves. [Current stage times](priority_pass/ATTRIBUTION.md) are profile16664. dQ remains below80; no complete-backward SOL80/90 claim.

> **Latest installed backward:** [below80/README.md](below80/README.md) records the qualified three-kernel update16580, additional 7.2–10.1% backward reduction, and projection/LN HBM SOL82.84–84.94%. Earlier results below are tied to their named artifacts.


> **Newer installed result:** [SOL90_PROGRESS.md](SOL90_PROGRESS.md) records the qualified grouped core, native dQ, and projection/LN/residual updates, including L384; the latest gate/context status is recorded there. Historical qualification and traffic data below remain tied to their named artifacts. Complete backward SOL90 is not reached.

2026-09-23 update: [bias_fusion/README.md](bias_fusion/README.md) records an additional
native CUDA/TMA dK/dV + grouped dBias fusion at L768/1024. Installed job16112
reduces whole-module backward0.23–0.32% at L768 and2.74–2.99% at L1024 **on top of** the
projection optimization below. Bias scratch halves (3.624 ->1.812GB at L768;
8.590 ->4.295GB at L1024), with measured32.7% lower whole-core DRAM traffic at
L768. L384 retains the previous path. `_fuse_bias_backward=False` opts out of
this new fusion. Its separate report contains installed-path verification.

The remainder documents the 2026-09-22 projection optimization and its baseline.
The actual engine training module uses a native CUDA/TMA projection
input-gradient kernel for qualified H100 shapes. This is a backward improvement;
the closed forward core SOL90 search remains closed. No backward SOL90 claim.

## Installed whole-module performance

Job15537, H100 80GB, B1 square BF16 inputs **and projection weights**, C128/H4/D32,
self-attention, no QK norm, dropout disabled for timing, broadcast key mask (every
seventh key masked), nonzero output weights. Both directions include residuals,
all input/parameter gradients and required layout copies.

Each comparison uses separate CUDA graphs in one GPU process, 12 rounds with
alternating AB/BA order, 15 replays per measurement. Tables show medians in ms and
time reduction from those medians; JSON also preserves every paired ratio.

| L | Direction | Bwd before | Bwd after | Bwd reduction | Fwd+bwd before | Fwd+bwd after | Reduction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 384 | starting | 1.540 | 1.388 | 9.86% | 2.047 | 1.907 | 6.80% |
| 384 | ending | 1.608 | 1.457 | 9.40% | 2.220 | 2.075 | 6.53% |
| 768 | starting | 8.424 | 7.850 | 6.81% | 10.977 | 10.415 | 5.12% |
| 768 | ending | 8.686 | 8.121 | 6.51% | 11.625 | 11.076 | 4.72% |
| 1024 | starting | 17.959 | 17.142 | 4.55% | 23.498 | 22.594 | 3.85% |
| 1024 | ending | 18.466 | 17.561 | 4.90% | 24.617 | 23.627 | 4.02% |

These are module measurements, not a whole training step. The existing H100
LayerNorm cache lacked these BF16/FP32 combinations and used its warmed heuristic
autotune subset in **both** arms. Compilation/autotuning is excluded from timings.
Raw measurements: `installed-paired-L384.json`, `installed-paired-L768.json`,
`installed-paired-L1024.json`. `paired-L*.json` is the earlier isolated control.

## Implementation and remaining bottlenecks

- One CTA handles 64 tokens by 128 input channels. 256 threads are launched;
  one producer warp issues TMA, one 128-thread consumer warpgroup executes WGMMA.
- Two shared-memory stages hold a 64x64 gradient tile and a 128x64 weight tile.
  Transaction barriers make data ready; a consumer barrier plus empty-stage
  arrival prevents reuse before WGMMA completion. Descriptors are grid-constant.
- The four wide projection products accumulate into one FP32 tile. The four-head
  bias projection joins it in the epilogue; the input gradient is stored BF16 once.
- Replaces five independent input-gradient GEMMs and four BF16 additions.
  L768 isolated input projection backward: **1.116ms -> 0.528ms** (job15519).
- Forward remains five ordinary Linear calls. All five weight-gradient GEMMs are
  retained; no detached/cached training weights and no state-dict format change.
- That projection-only change retained attention dQ/dK/dV, delta preprocessing,
  bias reduction, LayerNorm and gate/output-projection backward. Its L768 core
  remained about6.0ms, including dK/dV about3.2ms, dQ about1.6ms and bias reduction
  about1.15ms. The later grouped-bias fusion halves that3.624GB scratch to1.812GB;
  see the current update above.

The installed code is at
`../../trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda/`;
the dispatch is in the engine's `modules/triangle_attention/module.py`.
It activates for H100, BF16 inputs/weights, B1, L384/768/1024, C128/H4/D32,
self-attention, grad enabled, no QK norm. BF16 autocast is supported. Other shapes,
dtypes, devices, missing/incompatible binary artifacts and FP16 autocast retain
existing dispatch. The existing module itself does not support BF16 weights plus
FP16 autocast backward: job15540 failed in its gate kernel with mixed operand
dtypes before reaching the new path. This change does not repair that limitation.
`model._fuse_projection_backward = False` disables projection fusion. To reproduce
the historical table, also set `_fuse_bias_backward=False` in both arms; the paired
projection harness now does this explicitly.
The inference-only Anthropic CUDA forward path is separate and unchanged.

## Qualification

- 48 native cases: row counts1/63/64/65/127/128/129/1024, three scales, two CTA
  configurations. Independent FP64 products, finite values and repeatability.
  Relative L2 is about0.166%, consistent with final BF16 storage.
- 16 small full-module cases: both directions, no/mixed/single-key/all-masked masks,
  dropout0/0.25. Forward unchanged; every input and parameter gradient compared.
  FP64 module references for nonzero signals: maximum candidate relative L2<0.7%.
- A single live key has analytically zero dQ/dK/dbias. Existing BF16 attention
  leaves small residuals (largest tested parameter RMS about1.1e-5). Those are
  recorded with absolute errors and a no-regression comparison, not division by
  zero or a claim that the old attention residual was fixed. See `correctness.json`.
- Full L384/768/1024 module gradients, both directions, compared on graph replay;
  maximum baseline-relative L2 across these results<0.32%.
- Installed L384 dispatch verified by CUPTI, dropout, all-masked behavior, frozen
  parameters, and SGD updates changing entries of **every parameter**.
- `torch.compile(backend="inductor", fullgraph=True)` forward/backward passes in
  both directions. Generated backward calls `triangle_projection_dgrad_cuda`.
  BF16-autocast compiled execution is additionally checked by CUPTI.
- Job15520: memcheck0 errors; racecheck0 hazards/errors/warnings; synccheck0 errors.
  Installed binary/source are identical to this sanitized K64 candidate.
- The forward core manifest still matches all12 entries.

Evidence: `correctness.json`, `installed-correctness.json`, `amp-correctness.json`,
`sanitize-15520.log`, `installed-15537.log`, `amp-15543.log`, package `manifest.json`.

## Controls and invalid runs

cuEquivariance CUDA replacement for the attention core was slower: L768 module
backward8.44ms ->16.08ms starting,8.70ms ->16.36ms ending (job15480), including its
layout conversions. The profiler showed a cuDNN SM80 backward kernel; not adopted.
An addmm accumulation control gave only about3% module improvement. M128 CTA and
K128 staging supplied no additional usable gain. K64/M64 is the selected artifact.

Jobs15471/15472 exposed graph harness initialization/stream ownership errors;
15472's completed core measurement is retained, its module capture is invalid.
Early compile failures15509 and illegal-descriptor access15513 have no valid
performance results; grid-constant descriptor storage fixed the latter before
numerical/sanitizer qualification. Jobs15521/15524 used an invalid relative-error
criterion against exact-zero FP64 gradients; corrected tests record the baseline
residual explicitly. Job15535 exposed an lru-cache wrapper crossing Dynamo's
constant boundary; the installed availability helper fixes that. Failed evidence
is not counted as qualification.

## Reproduction

From the workspace root:

```bash
sbatch runs/triattn_20260921/bwd/installed.sbatch
sbatch runs/triattn_20260921/bwd/amp.sbatch
sbatch runs/triattn_20260921/bwd/sanitize.sbatch
```

Rebuild the installed artifact with the CUDA12.9/torch2.10cu128 environment:

```bash
MAX_JOBS=2 bash runs/anthropic_adoption_20260919/env.sh python \
  runs/trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/triangle_attention/cuda/build_native.py
```

CUTLASS4.2 is found at the existing sibling run path or via `CUTLASS_PATH`.
The builder refreshes the local manifest. No test vectors were regenerated.
