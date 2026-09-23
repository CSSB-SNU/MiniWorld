# Transition performance recheck — 2026-09-15

## Findings

The width512 training slowdown reproduces in an alternating, same-process,
same-GPU comparison. It is not a mislabeled PyTorch backend or a missing backward.
However, the earlier comparison omitted a substantial difference in activation
storage, and width512 does not represent every Transition used by the model.

All variants below use torch.compile(fullgraph=True), manual CUDA graphs,
BF16 inputs and linear weights, identical copied parameters/input/gradient,
batch1 and expansion4, with TF32 disabled. Normalization parameters remain FP32
in both modules; both paths cast their affine parameters for the BF16 computation.
Every output, input gradient and parameter gradient was checked against the
compiled PyTorch reference (relative L2 threshold0.025). Transition has no dropout.

## Same-GPU paired training times

Milliseconds per forward + backward + gradient reset. Four shape jobs ran in
parallel on separate H100 80GB GPUs. Within each shape, both implementations
resided in the same process/GPU. Twelve timing rounds alternate execution order;
each sample contains20 graph replays. Compilation/tuning is excluded.

| Shape | Compiled PyTorch | Engine | PT time / engine time |
| --- | ---: | ---: | ---: |
| Pair [1,768,768,128] | 4.7119 | 4.1109 | 1.146x |
| Pair [1,768,768,512] | 25.6411 | 30.3171 | 0.846x |
| Single [1,768,384] | 0.08044 | 0.21574 | 0.373x |
| Token [1,768,768] | 0.14514 | 0.19895 | 0.730x |

The active model configuration uses d_pair128 and d_single384, with token width768.
The last two rows are unconditioned Transition probes at those widths, not complete
ConditionedTransition or diffusion-block measurements. The previous width512
pair benchmark was a qualification case for the upgraded wide CuTe path, not a
measurement of the model's default pair Transition.

## Why width512 differs

`kernels/transition/cute/fused.py` saves the input and LayerNorm statistics. It
reconstructs normalized inputs and expanded gate activations during backward.
The PyTorch reference keeps expanded activations for backward. Same output
formula and precision do not imply the same activation-memory policy.

A non-reentrant checkpoint around the PyTorch module, compiled with
preserve_rng_state=False (this module has no randomness), provides a separate
recomputation comparison:

| Width512 pair case | Paired training ms | Extra live allocation after forward (GiB) | Extra peak allocation across step (GiB) |
| --- | ---: | ---: | ---: |
| PyTorch, saved activations | 25.6411 | 7.8794 | 10.1313 |
| Engine, recomputation | 30.3171 | 0.5669 | 11.8228 |
| PyTorch + checkpoint | 30.9665 | 0.5625 | 14.6294 |

Memory numbers subtract the already-live model/input/gradient buffers, warm both
side and measurement streams (cuBLAS workspace allocations are stream-local),
and are taken before allocating CUDA graph pools. These are allocated-memory
deltas for one module, not reserved VRAM or whole-model peaks. The engine retains
far less forward state, but its backward temporaries make this step's peak HIGHER
than ordinary PyTorch. A blanket claim of lower peak memory would be incorrect.
Checkpoint and engine recomputation do not perform an identical schedule of ops;
the checkpoint result is a memory-policy comparison, not a replacement for the
ordinary PyTorch baseline. Their approximately2% timing difference is small.

GPU kernel profiling attributes the width512 difference predominantly to backward:

| Sum of actual GPU kernel durations (ms) | PyTorch | Engine | PyTorch + checkpoint |
| --- | ---: | ---: | ---: |
| Forward | 8.691 | 7.282 | 8.658 |
| Backward | 15.461 | 20.487 | 20.438 |

These profile sums are separate from paired graph medians: they exclude launch
gaps, omit gradient reset, and may vary with profiling. GPU annotation spans are
excluded to avoid double-counting kernels. CPU CUDA-launch correlation IDs map
actual kernel events to forward/backward ranges.

The engine's `_transition_expand_gatebwd_kernel` alone takes about7.75ms while
recomputing expand/gate values and producing gradients. The fused forward is
faster; its gain is outweighed by backward recomputation. Existing CuTe backward
was also tested using the current explicit settings API:

- settings.configure(transition_large_d_training="cute")
- Observed the actual GemmDLnGated SM90 kernel in the trace; gradient checks passed.
- Engine31.8476ms vs compiled PyTorch25.6737ms in job13030.
- Extra step peak14.6377GiB. Thus simply switching this shape to CuTe backward
  does not fix the performance issue.

The default Triton backward was therefore not an accidental failure to connect
the faster existing CuTe alternative. No kernel or production dispatch was changed
in this recheck.

## Smaller token inputs

For [1,768,384], the engine chooses `_transition_b2b_ktiled_kernel`. Its forward
alone consumes approximately0.113ms of GPU kernel time, exceeding the entire
compiled-PyTorch step's0.080ms graph median. This is a concrete poor-performing
shape/dispatch combination, in addition to the recomputation cost.

For [1,768,768], both forward and backward kernel-duration sums are higher in the
engine. Saving fewer activations does not make the existing path the fastest
choice for these small row counts. Full-grid tuning was not rebuilt; this audit
measures the installed runtime policy and its available/heuristic configurations.

## Reproduce and evidence

On an allocated H100:

```bash
python scripts/audit_transition_performance.py --width 512 --layout pair --checkpoint --profile --output runs/transition_recheck/repeat.json
python scripts/audit_transition_performance.py --width 128 --layout pair --profile --output runs/transition_recheck/pair128.json
python scripts/audit_transition_performance.py --width 384 --layout token --profile --output runs/transition_recheck/single384.json
python scripts/audit_transition_performance.py --width 768 --layout token --profile --output runs/transition_recheck/token768.json
```

Use --engine-backward cute for the explicit alternative. The audit records the
active setting and asserts that its CuTe kernel appears when profiling. Old
MINIWORLD_TRANSITION_LARGE_D_TRAINING environment examples in kernel comments are
stale; current settings.py no longer reads that variable.

Local artifacts under runs/transition_recheck:

- pair512_13022.json, pair128_13023.json, single384_13024.json,
  token768_13025.json: paired timing and original traces.
- pair512_memory_13026.json, single384_memory_13027.json,
  token768_memory_13028.json: corrected memory-only measurements after warming
  the measurement stream. Initial PyTorch memory values included its first
  stream-local workspace allocation; do not use them as activation-memory counts.
- pair512_cute_13030.json: explicit CuTe backward, with runtime kernel verification.
- annotated_profiles.json: actual kernel durations re-derived from Chrome traces,
  excluding GPU user-annotation spans present in the initial event-list summaries.
- Job13029 used the obsolete environment switch and actually repeated the default
  Triton path. Its misleadingly named `pair512_cute` artifact is excluded from the
  CuTe comparison; job13030 is the verified result.

Job logs are runs/v1.0.1/phase2b/full_wiring_<job>.out. The audit creates no model
parameter updates and does not benchmark optimizer steps or full-model training.

## Forced Triton comparison

See [the four-way Triton comparison](transition-triton-comparison.md) for measured
fused/split Triton alternatives, H100 auto dispatch and compiled PyTorch at the
same four shapes.
