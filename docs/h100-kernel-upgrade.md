# H100 kernel upgrade — 2026-09-15

## Installed changes

Base engine: `1bc0803e3b2fef3b963fdc383e090c0adcccdb43` plus the three existing
compatibility/wiring patches. The fourth patch, `miniworld-engine-h100-upgrade.patch`,
updates 18 engine files. MiniWorld and `libs/team-gm` carry identical patches and
install helpers; `pixi run -e cu128 engine-setup` reapplies the full stack.

| Area | Final behavior |
| --- | --- |
| TriMul input projection | New SM90 CuTe gated GEMM applies the pair mask after GLU in the output epilogue and writes the layout consumed by the contraction. Single/incoming/bidirectional training and inference are connected. Saved preactivations and the normalized input for the output gate stay unmasked. |
| TriMul output | Existing gate, shared-row dropout, output mask and residual fusion remains connected. Backward retains its mask/dropout factors. Kernel initialization now runs before cold `torch.compile(fullgraph=True)` tracing. |
| Transition | New SM90 squeeze GEMM reads residual as a separate C operand; no output staging copy or separate add. Enabled for BF16 width 512, expansion 4, at least 16384 flattened rows in multiples of 128, contiguous same-dtype residual. Other shapes and mixed-dtype residuals preserve the existing matmul-plus-add calculation. Identity gradient is added exactly once. |
| LayerNormLinear M2 | Restored only for Hopper BF16 square width 128, 2-D inputs, aligned rows >=128, inference without saved statistics or prefolded operands. Opaque dispatch supports fullgraph compilation and CUDA graphs. Other cases use M1. |
| Builder | Both new GEMMs have registry entries, drivers, source-sensitive configuration keys and isolated native precompile contracts. This is not a complete rebuild of every shape's tuning cache. |

Transition has no dropout argument in this API; adding dropout to its GEMM would
change the model. TriMul keeps its existing shared-row dropout semantics. A mask
belongs on the contracted projections and output update, not on the output-gate
input or the identity residual.

## Which experiments remain

These decisions use engine git history, followed by current measurements where
we changed dispatch. Historical speedups below use the comparison named in that
commit, not the current complete module.

| Implementation | Decision and evidence |
| --- | --- |
| CUDA standalone expand + gate | Excluded from routine `build all` (`developed=no`); direct reproduction API retained. `b2ce8134`: 0.98x / 0.79x / 0.55x against CuTe at widths 128 / 256 / 512. |
| M2 `lnl_ws=1` | Historical opt-in only, default remains 0. `a806c93b`: 5.8 ms vs M1 0.61 ms at width 768, 262144 rows; only three load warps reading X caused a bandwidth bottleneck. |
| AdaLN CUTLASS TF32 | Kept outside production dispatch and routine builder. `71fee1c8`: token 0.91–0.94x and atom 0.58–0.59x vs the existing materialized implementation. |
| Ordinary M2 (`lnl_ws=0`) | Worth restoring selectively. `1d1f5fc9` disabled it for Quack 0.5 compatibility failures despite earlier speed gains. Current width 128 qualification passed; width 256 does not consistently win. |
| TM2 custom output GEMM | Retain for shape-specific comparison. `eebedbbc` fixed a deadlock and measured 0.0556 vs 0.0583 ms against cuEquivariance at L384. That does not prove a win against the current fused Triton back. |
| CuTe gate backward / split-back | Retain. `6f9854ee` reported a 22% whole-path gain at width 512; `5dcd6663` reported a 1.3–1.5x gate-kernel win. These are not isolated current end-to-end measurements. |
| DAB + LayerNorm backward | Retain as an experiment. `a8f0c566`: 1.311 vs 1.256 ms for stacked backward at L384/width128, from separate runs; insufficient evidence to discard every shape. |
| ConditionedTransition CUTLASS | Research only. `71fee1c8` lost on smaller inputs; large augmented atom cases approached parity, while token cases still lost. |

Unused wrappers are not counted as independent kernels or deleted merely because
they lack a current production caller. The low-value experiments above are kept
out of routine optimization; promising kernels require shape-specific evidence.

## H100 measurements

H100 80GB HBM3, PyTorch 2.10.0+cu128, BF16, batch 1, identical seeds and weights.
Each family used its own GPU allocation; baseline and candidate ran on the same
GPU within that family. Jobs ran concurrently across GPUs. These are CUDA graph
replay medians after warmup, excluding compilation/autotuning. Training includes
backward and gradient reset, with dropout 0.25 for TriMul. Pair masks contain holes.

| Module | L | Inference before → after (ms) | Speedup | Training before → after (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Outgoing TriMul, width128 | 128 | 0.0801 → 0.0539 | 1.49x | 0.2646 → 0.2529 | 1.05x |
| Outgoing TriMul, width128 | 384 | 0.5010 → 0.2839 | 1.76x | 1.3638 → 1.2639 | 1.08x |
| Outgoing TriMul, width128 | 768 | 1.9498 → 1.0378 | 1.88x | 4.9299 → 4.5595 | 1.08x |
| Bidirectional TriMul, width128 | 128 | 0.1706 → 0.1217 | 1.40x | 0.3739 → 0.3553 | 1.05x |
| Bidirectional TriMul, width128 | 384 | 1.0215 → 0.5903 | 1.73x | 2.1870 → 1.9972 | 1.10x |
| Bidirectional TriMul, width128 | 768 | 3.9118 → 2.1257 | 1.84x | 8.4257 → 7.6654 | 1.10x |

Final Transition width512 forward, paired alternating graph replays in job 13010:

| Rows | Separate residual (ms) | CuTe residual (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 16384 | 0.2240 | 0.2126 | 1.054x |
| 147456 | 1.9713 | 1.9073 | 1.034x |
| 589824 | 8.0131 | 7.5566 | 1.060x |

Profiler: the forward shrank from stats + fold + expand + squeeze + add to
stats + fold + expand + squeeze-with-residual, with no device-to-device copy.
An initial `torch.addmm` version retained a staging copy and regressed the small
shape by about 2%; it is not the installed implementation.

Final installed-package Transition rerun, job13013 (separate baseline/installed
processes on the same GPU; includes the complete module and backward):

| L | Mode | Baseline (ms) | Installed (ms) | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 128 | inference | 0.2247 | 0.2084 | 1.079x |
| 128 | training | 0.9383 | 0.9183 | 1.022x |
| 384 | inference | 1.9647 | 1.8955 | 1.037x |
| 384 | training | 7.7239 | 7.7085 | 1.002x |
| 768 | inference | 7.8132 | 7.4924 | 1.043x |
| 768 | training | 30.6768 | 30.3321 | 1.011x |

Training gains are small; individual L384 samples varied more than the measured
median difference. Use the paired forward measurements above for the clearest
residual-fusion comparison. Full training speedup needs a whole-model benchmark.

M2 width128 forward vs M1: 1.04x at 128 rows, 1.06x at 4096, 1.09x at 16384,
1.49x at 147456, 1.81x at 589824. Width256 ranged from 0.95x to 1.03x and stays
on M1. These kernel/module gains are not a measured whole-model training speedup.

## Verification and evidence

- 42 mask cases: outgoing/incoming/bidirectional, lengths128/384, missing/all-valid/
  holed/all-invalid masks, inference and all training gradients; custom epsilon
  and nonzero normalization biases expose masking the wrong gate input (job12995).
- Three compiled TriMul dropout tests against an FP32 reference passed, including
  backward and restored production RNG (job12995).
- 19 upgrade tests passed after the final Transition replacement (job13009):
  BF16 and FP32-master/autocast gradient checks, M2 layouts/epsilon/compile/graphs,
  masked partial tiles and preserved preactivations, changing dropout RNG and mask
  during graph replay. Relative norm error thresholds are 2.5%; this is not a
  maximum-coordinate or whole-model error bound.
- Actual installed package: all **62** upgrade/mask tests passed (job13012), including
  a cold fullgraph-compiled squeeze residual and changed-input CUDA graph replay.
- Existing engine wiring / compilation / AMP suite: 34 passed (job13005).
- Real epoch-93 checkpoint, L768 / 8192 padded atoms / MSA depth2048:
  outputs finite, all 685 trainable diffusion/projection parameter tensors had
  finite gradients, and no audited module used the PyTorch default (job13005).
  This squared-output probe is not a production loss or optimizer update.
  Evidence: `runs/publish_validation/h100_upgrade_runtime_13005.json`.
- Both new GEMMs passed CPU-only isolated native compilation on a compute node:
  masked-front save/no-save and squeeze-residual (three tasks, job13011).
  Builder plan verification finds 24 masked-front units, 20 squeeze-residual units
  restricted to width512, and zero routine units for the retired CUDA expand-only path.
- Reinstall simulation: pristine pinned engine + four patches matched all 28
  affected candidate files byte for byte. Actual installation updated 17 files;
  their SHA-256 values matched the tested candidate, and a second application
  updated zero files. Patch-stack CPU tests: four passed, including overlap and
  refusal without partial writes on an incompatible source tree.

Durable regression tests: `tests/test_engine_h100_upgrade.py`,
`tests/test_engine_h100_masks.py`, `tests/test_engine_patch_stack.py`.
Native compile reproduction: `scripts/check_engine_h100_native_compile.py` (run
on a compute node). Module benchmark harness: `scripts/benchmark_engine_h100.py`; select an isolated
pre-upgrade/post-upgrade package through `PYTHONPATH`, then use `--family outgoing`,
`--family bidir` or `--family transition`, with `--label` and `--output`.

Local raw artifacts are in `runs/h100_upgrade/`: family baseline/candidate JSONs
(jobs12996–12998), `m2_benchmark.json`, `transition_paired.json`, profiler event
lists, `squeeze_probe.json`, native compile task results and `installed_sha256.json`.
Job logs are `runs/v1.0.1/phase2b/full_wiring_<job>.out`. Initial Transition
benchmark JSONs from job12998 describe the superseded addmm candidate, not the
final CuTe epilogue; use job13010 for the final forward comparison.

Final installed Transition artifacts: `transition_final_baseline_13013.json` and
`transition_final_installed_13013.json` under `runs/h100_upgrade/`.

## Subsequent PyTorch comparison

[Direct PyTorch comparisons](h100-pytorch-comparison.md) include torch.compile,
three additional cold-compile fixes, and their regression results. The current
four-patch stack affects 28 source files, including 18 in the H100 patch.

## TriMul tuning audit

[Runtime cache verification](trimul-tuning-audit.md) confirms that the tested
Triton keys are tuned, but the new masked H100 projection has no tuning file and
its build driver misses most runtime keys. Native tuning is not complete.

## Applied TriMul native tuning

[The subsequent tuning pass](trimul-h100-tuning.md) fixes native cache keys and
build workloads, measures all candidates and adds a fifth patch carrying the
measured H100 cache. The earlier missing-cache audit is historical.
