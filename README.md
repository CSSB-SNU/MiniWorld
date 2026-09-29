# MiniWorld

[Repository map, development branches, kernel status, and transfer checklist (2026-09-27)](REPOSITORIES.md).

## Training pipeline

v1.1 work: [distogram fixes and interchain weighting](docs/v1.1-distogram.md),
with a [diagram of current cropping](tmp_kernel/cropping/CROPPING.svg).
v1.2.0: [AF3-style MSA sampling — full-depth PDB pool and per-recycle re-draw](docs/v1.2-msa-sampling.md).
v2.0.0 (phase 2): [BiasOnlyTokenDiT + full-bf16 diffusion module, 1.9-2.4x diffusion step](docs/v2.0.0-dit.md).

The stages are distogram training (1a/1b), diffusion training (2a/2b), and
confidence training (3a/3b). See [the pipeline guide](docs/pipeline.md) for
configuration names, crop sizes, checkpoint handoffs, and version variants.

## Engine setup

**Kernel development (2026-09-27):** Our H100 inference and training kernels are
developed in miniworld-engine. The [repository map](REPOSITORIES.md) identifies
the current development worktrees, installed research runtime, and remaining
validation work. Upstream attribution and the earlier development direction are
recorded in [the historical kernel report](tmp_kernel/KERNEL_PROGRESS.md#개발-방향-전환--anthropic의-추론-최적화를-계승).

[Anthropic inference integration and H100 profiling results](tmp_kernel/ANTHROPIC_INFERENCE.md)
cover the first shared-kernel campaign. These historical records predate the
current training kernels.
The [interactive HTML dashboard](tmp_kernel/ANTHROPIC_STATUS.html) shows wiring, timings,
NCU bottlenecks, and remaining integration work.

Add `train.engine_backend=triton` to pin training to the engine Triton paths.
See [backend selection and scope](docs/triton-backend-option.md), and the
[full-model CUDA graph training comparison](docs/miniworld-training-cudagraph-ab.md).

Check out the pinned submodules, install the selected Pixi environment, and
check the pinned `miniworld-engine` package:

```bash
git submodule update --init --recursive
pixi install -e cu128
pixi run -e cu128 engine-setup
```

The `cu128` environment keeps its historical name; since miniworld-engine v2.2.0 it
installs torch 2.13.0+cu129 and cuequivariance 0.12.

Run `engine-setup` again after reinstalling the engine dependency. It verifies
the consolidated engine version, recorded Git pin and required source files.
The old 1.x patch stack is historical and is not reapplied to main. FlashAttention and MathDx kernel
prerequisites still need to be available for the selected GPU backend; see the
environment comments in [pyproject.toml](pyproject.toml).

The integration preserves the SWA block equations and checkpoint layout.
[The engine audit](docs/engine-wiring-audit.md) records H100 forward/backward
checks, numerical differences, CUDA Graph results, and their limits.
[The H100 upgrade](docs/h100-kernel-upgrade.md) records kernel selection,
mask/residual fusion, and measured speedups. See also the
[direct PyTorch comparison](docs/h100-pytorch-comparison.md).
[Packed bidirectional TriMul training](docs/trimul-packed-training.md) records the
latest H100/Triton comparison and the compile-independent contraction dispatch.
[The September 22 TriMul closeout](docs/trimul-fusion/closeout-20260922/README.md)
records the final measured B1/B7 candidate, reproducibility evidence, and the
remaining input-LayerNorm gradient validation issue. See the
[consolidated status and wiring](TRIMUL_STATUS.html).
[AdaLN and attention fixes](docs/engine-adaln-attention-fixes.md) cover the H100
shared-memory alignment failure and L8192 attention training memory.

## Evaluation and documentation

- [FoldBench evaluation conventions](docs/foldbench-evaluation.md)
- [Known issues and validation limits](docs/known-issues.md)
- [Technical report draft and build instructions](docs/paper/README.md)

Generated FoldBench results live under `eval_results/` and are excluded from Git.

## TriMul update (2026-09-17)

[Kernel changes, validation, and reinstall instructions](docs/trimul-release-20260917.md).
[Inference kernel wiring](TRIMUL_INFERENCE.svg) · [Wiring details](docs/trimul-fusion/inference.md).
