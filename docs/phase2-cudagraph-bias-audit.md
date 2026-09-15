# Phase 2 CUDA Graph / ln_pair.bias audit (2026-09-15)

For the subsequent complete engine wiring changes and their validation, see
[engine-wiring-audit.md](engine-wiring-audit.md). The measurements below describe
the intermediate versions identified by their job IDs.

## CUDA Graph result

The frozen training trunk does use CUDA Graphs. Output addresses were not a
valid test of whether graphs were active. The old `ACTIVE/SKIPPED` pointer-based
verdict has been removed from `profile_trunk_vs_diffusion.py`.

Measured on one H100 80GB, PyTorch 2.10.0+cu128, epoch 93 / step 9300 **model
weights**, BF16 autocast, 768 tokens, 8192 padded atoms, MSA depth 2048, two
recycles. All arms compile the same `_condition_impl`; only `triton.cudagraphs`
changes for OFF vs Trees. Timings exclude compilation, autotuning and capture,
include output clones for the downstream training consumer, and synchronize the
GPU. Ten measured repetitions per arm.

| Version | Graph OFF | Trees ON | Replays per trunk call |
| --- | ---: | ---: | ---: |
| Before graph-break fixes, job 12952 | 0.48284 s | 0.48815 s | 334 |
| After fixes, paired alternating measurements, job 12957 | 0.48044 s | 0.47290 s | 3 |

The paired speedup is **1.01594x**, about 1.6%. The profiler independently recorded
9 `cudaGraphLaunch` calls for three trunk invocations after the fixes; the Python
replay counter recorded 30 for ten measured invocations. Measured recaptures: 0.
The Dynamo `graph_break` counter is empty after those fixes, but this alone was
insufficient: the later follow-up found a frame-abandonment reason under the
separate `unimplemented` counter. See the correction below.

OFF and ON outputs are **bitwise equal on the same tensors**. After modifying
reference coordinates in place, outputs change and OFF/ON remain bitwise equal.
This checks that replay consumes new input values. It does not establish a
performance result for other shapes, recycle counts, or the confidence model's
multi-step diffusion rollout.

The eager-vs-compiled differences in the JSON are separate from the graph toggle:
both OFF and ON use compilation, and the paired graph comparison is exact.

Evidence:

- `runs/v1.0.1/phase2b/cgaudit_12952/{off,trees,manual}.json`
- `runs/v1.0.1/phase2b/cgaudit_12957/trees.json`
- Corresponding `*.trace.json` files contain the CUDA runtime events.

Job 12957's OFF and Trees arms completed and saved the results above. Its
subsequent, redundant manual arm was cancelled during a training-catalog rebuild;
the earlier manual arm in job 12952 completed (0.47551 s, one replay per call).

### Follow-up: why three replays, and the missed compiler boundary

The job-12957 trace contains **two compiled graph bodies, invoked three times**:

| Replay | Region | CUDA kernels in the first profiled invocation |
| --- | --- | ---: |
| 1 | Input embedding, including three SWA atom-attention blocks | 114 |
| 2 | Template, MSA and 48-block Pairformer: recycle 1 | 1706 |
| 3 | The same trunk graph body: recycle 2 | 1706 |

The corresponding CUDA launch correlation IDs are 110, 179 and 227. Recycle
graphs have identical kernel-name sequences. Thus this was not three complete
forward passes. It was also **not** proof that the whole conditioning function
had compiled without interruption.

`_condition_impl` used `contextlib.ExitStack` inside its recycle loop. Dynamo
abandoned that frame and compiled `_embed` and `_trunk_step` independently.
A minimal reproduction on the installed PyTorch reports `unsupported
contextlib.* API: ExitStack` under `unimplemented`, while `graph_break` remains
empty. The previous report's implication that there were no remaining compiler
boundaries was incorrect.

The loop now uses `torch.set_grad_enabled`, preserving the outer gradient state
and allowing gradients only on the last recycle. CPU fullgraph tests cover
1, 2 and 4 recycles with gradients enabled and disabled. The audit now supports
`--fullgraph`, records `unimplemented`, and accepts an explicit read-only catalog
snapshot to avoid rebuilding the production training catalog.

Job **12967**, on the same H100/L768/two-recycle configuration, compiled with
`fullgraph=True` and measured **one replay per call** (six replays for six calls;
three CUDA graph launches in the three-call profiler). Both `graph_break` and
`unimplemented` were empty. After warmup there were no new captures. Paired
median OFF/ON times were **0.48337 / 0.47531 seconds**, a 1.01697x ratio. This is
an OFF/ON comparison of the new implementation, not a controlled speed comparison
against the earlier three-graph implementation on identical preprocessed inputs.

Unlike the earlier three-graph result, this run was **not bitwise equal** across
OFF/ON: single output was exact, while pair output relative L2 difference was
**9.8771e-5** (0.00988%; maximum absolute difference 48 on this checkpoint's
large-magnitude pair state). Modified-input comparison was 9.9987e-5, and the
changed input affected the result. Do not carry the earlier exact-equality claim
over to this new measurement. These are numerical checks, not a training-quality
or structure-quality result.

Repeat diagnostic **12973** confirmed bitwise-stable repeated calls **within**
each mode (OFF/OFF and ON/ON). The OFF/ON pair difference persisted at relative
L2 1.0087e-4. Thus it is a reproducible difference between the two execution
paths, not run-to-run variation in this check. The exact responsible kernel has
not been isolated. Input mutation still affects the output; both modes remain
finite. Its paired OFF/ON medians were 0.48217 / 0.47626 seconds (1.01241x).

### Follow-up: engine wiring

Configuration did not previously reach every engine-selectable layer:

- Four MSA OPMs and three MSA pair-weighted averaging modules omitted
  `implementation=eng`. Their ten LayerNorms therefore used the PyTorch default.
- Diffusion conditioning's two pair and two single Transitions also omitted the
  implementation selector.
- The optional MiniPairformer single-track Transition had the same omission
  (`use_single=False` in this phase-2 configuration).

These call sites now propagate their configured implementation. The OPM and
MSA averaging bodies still use engine-owned PyTorch einsums/softmax; their
selector controls the LayerNorm kernels, not a fused implementation of the
whole operation. MiniPairformer intentionally has no triangle attention.
The optional single-track AttentionPairBias has no implementation selector or
optimized attention dispatch in this installed engine version.

Other plain LayerNorms remain, including token DiT's `ln_pair` and the standalone
conditioning/encoder/decoder norms. Importing a class from miniworld-engine does
not mean every operation inside it uses a custom kernel. The original six
regression tests and frozen-trunk timing were not exhaustive integration proof.

Job **12970** passed all **14** tests in `tests/test_engine_wiring.py`: six
fullgraph/gradient-policy cases and eight GPU forward/backward cases for OPM,
MSA averaging and conditioning Transitions of widths 128 and 384, expansion 2.
The conditioning tests use FP32 parameters as in `DiffusionModel` and BF16
autocast. The MSA averaging pair-LN bias is a softmax-invariant direction; both
implementations' cancellation noise is bounded against the non-null LN scale
gradient at BF16 precision, rather than applying relative error to a theoretical
zero. All other compared gradient relative L2 errors are below 8%, and output
relative L2 errors are below 3%.

Job **12974** then ran the real checkpoint and L768 batch through the frozen
trunk, diffusion inference, and a diffusion backward pass. Runtime hooks verified
the ten MSA norms resolve to the engine CUDA LayerNorm family and execute twenty
times over two recycles. The diffusion profiler recorded:

| Operation | Actual engine custom-op calls |
| --- | ---: |
| Pair conditioning Transition | 2 |
| Single conditioning Transition (CuTe forward) | 2 |
| Atom SWA attention / QK norm + RoPE | 6 each |
| Token augmented attention | 24 |
| ConditionedTransition expand and squeeze stages | 24 each |
| AdaLN inference epilogue | 48 |

All **685 trainable parameters** received finite gradients for the denoising
output squared-mean probe loss (1.9299085); the output was finite. This validates
end-to-end gradient connectivity for this configuration and batch. It is not an
optimizer step, production-loss validation or convergence test. Runtime inventory
also explicitly found the **24 token-attention pair LayerNorms still using
PyTorch**, as described above. Evidence: `wiring_runtime_12974.json` and its trace;
reproducer: `scripts/audit_engine_wiring.py`.

The first runtime attempt, job 12972, saved its inference trace but was cancelled
during expensive profiler event-tree aggregation. The reproducer now counts the
exported custom-op events directly; job 12974 completed both forward and backward.

## Fixed causes

1. **Wheel import layout:** the CuTe loader walked upwards looking for a `src`
   directory, reaching `/` from an installed wheel. It now anchors on the
   `miniworld_engine` package directory. The source checkout has the same layout
   relative to that package.
2. **Missing cuBLASDx location:** the diagnostic submission now supplies and
   checks the existing MathDx installation. It invokes the existing environment
   directly and propagates Python failure as a nonzero batch exit status.
3. **Obsolete environment setting:** the current engine reads
   `settings.configure(transition_force_split=...)`; the old
   `MINIWORLD_TRANSITION_FORCE_SPLIT` export did not set it.
4. **CRC graph breaks:** constant-fold only the immutable axis-name checksum.
   Shape dimensions and validation still execute in `pack`; cache keys are
   unchanged.
5. **Native CUDA graph break:** register the b2b inference launch as a custom op
   with a fake implementation. Leave LN statistics, shapes and tensor transforms
   visible to the compiler.
6. **Mixed precision failure:** ConditionedTransition and AdaLN could hand
   BF16 activations and FP32 GEMM weights/saved operands to the same native call.
   They now pass a consistent compute dtype across forward/backward, retain
   FP32 master parameters, and use differentiable casts. AdaLN's norm affine
   parameter retains its precision.

## ln_pair.bias: what the evidence supports

Let `b` be the LayerNorm offset and `W` the subsequent projection. In exact
arithmetic, the offset adds the same `W b` to every key logit within a head.
`softmax(z + c) = softmax(z)`, so removing it is algebraically redundant. The
FP64 probe on actual pair inputs verifies the projected shift to about 1e-10.

That does **not** imply exact BF16 equivalence. The projection rounds differently
with and without the offset. The audit restores the offset **inside LayerNorm**,
rather than adding it after an already-rounded LN output, and isolates this one
change within the same new engine and checkpoint.

The earlier claim that three entire DiT blocks were dead/uniform was unsupported:

- The actual attention kernel accumulates dot products/logits in FP32. A large
  BF16 pair-bias offset does not imply that QK scores disappear.
- In the measured real input, QK standard deviations for the large-offset heads
  ranged from approximately 19 to 1216; assuming every head had O(1–10) logits
  was incorrect.
- There are **five heads across three blocks** with `abs(W b) > 1000`, not a
  measurement showing three whole blocks have stopped working.
- The BF16 spacing near 46324 is **256**, not 128. The script measures this with
  `torch.nextafter`.
- Growth of `b` alone does not prove Adam random walk, a particular gradient
  feedback mechanism, or that this caused a FoldBench deficit. Those causal
  claims need gradient/history or controlled training evidence.

Job 12956 initially found an approximately **0.82% relative L2 difference** in the
full diffusion output. The final mixed-precision implementation was rechecked in
job 12962 on a real sample from an explicitly selected, read-only catalog
snapshot (same epoch-93 model weights, one sampled noise level):

| Output metric | Job 12962 | Job 12963 |
| --- | ---: | ---: |
| **1386 valid atoms: relative L2** | **1.2736%** | **0.8148%** |
| Valid atoms: mean absolute difference | 0.01391 | 0.01036 |
| Valid atoms: maximum absolute difference | 0.1875 | 0.09375 |

These are separate diagnostic runs. The old/new bias comparison uses identical
inputs within each run. Exact preprocessed input tensors were not persisted,
so these are not presented as identical-input repeats across processes.

The actual kernel's attention probabilities were recovered with identity-valued
V inputs for blocks 0, 6 and 7, so the attention conclusion is not based solely on
a separate softmax implementation. This is a local numerical comparison, not a
loss/structure-quality measurement, and does not establish that deleting the
offset improves a trained checkpoint. The snapshot sampler uses its stored raw
weights; it is a diagnostic input source, not an assertion about training's
current source sampling proportions. Evidence: `biasaudit_12962.json` and the
final valid-query statistics in `biasaudit_12963.json`. Job 12963 excludes padding
queries and compares against a uniform distribution over the **167 valid keys**:

| Block / head | Effective attended positions, actual kernel | Total variation from uniform |
| --- | ---: | ---: |
| 0 / 3 | 1.0104 | 0.9938 |
| 0 / 14 | 1.7632 | 0.9835 |
| 6 / 3 | 1.0041 | 0.9940 |
| 7 / 0 | 1.0041 | 0.9940 |
| 7 / 10 | 1.0002 | 0.9940 |

Effective positions are `exp(mean attention entropy)` over valid queries.
Uniform attention over the valid keys would give 167 effective positions and
zero total variation. These heads are sharply concentrated, not uniform.

## Validation and reproduction

`tests/test_engine_cudagraph_regressions.py` covers unchanged shape keys under
fullgraph compilation, native CUDA transition fullgraph/replay with changed
inputs, and AMP forward/backward against the PyTorch reference at widths
128/128 and 768/384, using nonzero projection weights and FP32 master parameters.
Job 12961: **6 passed**.

Engine changes are preserved against the pinned engine revision
`1bc0803e3b2fef3b963fdc383e090c0adcccdb43` in:

- `patches/miniworld-engine-wheel-cute-path.patch`
- `patches/miniworld-engine-cudagraph-amp.patch`

They are applied to the local cu128 installation. The dependency revision in
`pyproject.toml` has not been repinned to an unpublished commit. After reinstalling
that environment, reapply the checked patches with:

```bash
.pixi/envs/cu128/bin/python scripts/apply_engine_audit_patches.py
sbatch submits/engine/audit_phase2_cudagraph.sbatch --paired
sbatch submits/engine/audit_ln_pair_bias.sbatch --catalog-snapshot \
  /home/psk6950/data/BioMolDB_20260224/catalog_trainitem_no_disordered_bioai_unified.arrow
```

The patch application is idempotent and refuses a source version that does not
match. `scripts/diag_cudagraph.py` now runs the measured audit instead of the old
pointer heuristic. It accepts `--ckpt` and writes timestamped JSON and trace files.

Relevant PyTorch documentation:
[CUDA Graph Trees](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html),
[numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).
