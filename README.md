# MiniWorld

[Repository map, pins and validation notes](docs/repositories.md) · [documentation index](docs/README.md).

| Path | Contents |
|---|---|
| `src/`, `configs/`, `tests/` | MiniWorld package, Hydra configs, tests |
| `libs/` | `patches/` for FoldBench (StructCooker, which builds the databases, is a separate repository) |
| `scripts/` | Python entry points and tools; `slurm/` holds sbatch launchers (mostly local, untracked) |
| `docs/` | Guides; `kernels/` kernel diagrams and status reports; `archive/` dated validation records ([index](docs/README.md)) |
| `runs/` | Every run output: training runs, `logs/`, `outputs/`, `eval_results/`, `wandb/` (ignored by Git). The H100 kernel-experiment records that used to be tracked here are under the tag `archive/h100-kernel-experiments-20260927`; `scripts/restore_kernel_experiments.sh` extracts them |

## Development

MiniWorld depends on `miniworld-engine` directly; the training loop, EMA, diffusion, blocks and layers that used to live in
team-gm are part of this package (`miniworld.training`, `miniworld.diffusion`, `miniworld.modules`, `miniworld.utils`).

```bash
bash scripts/check.sh          # ruff check, ruff format --check, pyright
bash scripts/check.sh --fix    # apply the safe fixes and the formatter first
pytest tests                   # needs a B200/H100 node; CI only runs ruff and a compile check
```

Known lint and type-check debt is listed in `pyproject.toml` (`[tool.ruff.lint] ignore`, `[tool.pyright]` warnings).
jaxtyping dimension names that are a single identifier are written with a leading space (`" B"`) so that linters
do not read them as forward references.

## Training pipeline

v1.1 work: [distogram fixes and interchain weighting](docs/v1.1-distogram.md),
with a [diagram of current cropping](docs/kernels/cropping/CROPPING.svg).
v1.2.0: [AF3-style MSA sampling — full-depth PDB pool and per-recycle re-draw](docs/v1.2-msa-sampling.md).
v2.0.0 (phase 2): [BiasOnlyTokenDiT + full-bf16 diffusion module, 1.9-2.4x diffusion step](docs/v2.0.0-dit.md).

The stages are distogram training (1a/1b), diffusion training (2a/2b), and
confidence training (3a/3b). See [the pipeline guide](docs/pipeline.md) for
configuration names, crop sizes, checkpoint handoffs, and version variants.

## Engine setup

**Kernel development (2026-09-27):** Our H100 inference and training kernels are
developed in miniworld-engine. The [repository map](docs/repositories.md) identifies
the current development worktrees, installed research runtime, and remaining
validation work. Upstream attribution and the earlier development direction are
recorded in [the historical kernel report](docs/kernels/KERNEL_PROGRESS.md#개발-방향-전환--anthropic의-추론-최적화를-계승).

[Anthropic inference integration and H100 profiling results](docs/kernels/ANTHROPIC_INFERENCE.md)
cover the first shared-kernel campaign. These historical records predate the
current training kernels.
The [interactive HTML dashboard](docs/kernels/ANTHROPIC_STATUS.html) shows wiring, timings,
NCU bottlenecks, and remaining integration work.

Add `train.engine_backend=triton` to pin training to the engine Triton paths.
See [backend selection and scope](docs/triton-backend-option.md), and the
[full-model CUDA graph training comparison](docs/miniworld-training-cudagraph-ab.md).

miniworld-engine is a git dependency pinned by `rev` in [pyproject.toml](pyproject.toml); there are no submodules.
Install the Pixi environment, fetch the CUTLASS headers the H100 kernels compile against, and check the pinned package:

```bash
pixi install -e cu128
pixi run -e cu128 setup-cutlass     # clones CUTLASS v4.2.1 into ./.cutlass once; the environment sets CUTLASS_PATH to it
pixi run -e cu128 engine-setup
```

The `cu128` environment keeps its historical name; since miniworld-engine v2.2.0 it
installs torch 2.13.0+cu129 and cuequivariance 0.12. FlashAttention-4 is in the lock, pinned to `4.0.0b32`: the engine unpacks
the four values `flash_attn.cute` returns in that beta, and a later beta returns more. Do not upgrade it by hand.
On the B200 boxes the engine is the local checkout
(`$W/mwenv`, see [docs/operations/b200-environment.md](docs/operations/b200-environment.md)) because they cannot fetch the GitHub pin.

Run `engine-setup` again after reinstalling the engine dependency. It verifies
the consolidated engine version, recorded Git pin and required source files, and says whether the engine kernels have been
prebuilt for this revision.

**After creating the environment or bumping the engine pin, prebuild the engine kernels on a GPU node of each architecture
(sm_90 H100, sm_100 B200).** The engine JIT-compiles its CUDA extensions with nvcc on first use, one at a time: without the
prebuild the first model call of every new environment / pin waits on nvcc (840-2000 s for the first FoldBench target on an
H100). `engine-prebuild` runs the production models (random weights) at every inference rung and training crop, then checks in
a new process that nothing compiles any more, and leaves a stamp `engine-setup` reads. On H100 it takes ~36 min with 8 workers
on 2 GPUs (~59 min with one); afterwards the first FoldBench call takes ~5 s instead of 840-2000 s:

```bash
sbatch scripts/slurm/engine_prebuild_h100.sbatch          # H100: 2 GPUs, 8 workers (--mode all)
pixi run -e cu128 engine-prebuild --jobs 8                # any GPU node (B200: python scripts/prebuild_engine.py --jobs 8)
pixi run -e cu128 engine-prebuild --verify-only           # only check that nothing would compile
```

The kernels land in the environment's own cache (`~/.cache/miniworld_engine_jit/py310-<env hash>`, torch extensions in its
`torch_extensions/`), so other environments (checkouts on other engine pins) neither share nor rebuild them. Shapes, timings and what is not
covered: [docs/operations/engine-prebuild.md](docs/operations/engine-prebuild.md).

MathDx and the other kernel prerequisites are listed in the
environment comments of [pyproject.toml](pyproject.toml).

The integration preserves the SWA block equations and checkpoint layout.
[The engine audit](docs/engine-wiring-audit.md) records H100 forward/backward
checks, numerical differences, CUDA Graph results, and their limits.
[The H100 upgrade](docs/h100-kernel-upgrade.md) records kernel selection,
mask/residual fusion, and measured speedups. See also the
[direct PyTorch comparison](docs/h100-pytorch-comparison.md).
[Packed bidirectional TriMul training](docs/trimul-packed-training.md) records the
latest H100/Triton comparison and the compile-independent contraction dispatch.
[The September 22 TriMul closeout](docs/archive/trimul-fusion/closeout-20260922/README.md)
records the final measured B1/B7 candidate, reproducibility evidence, and the
remaining input-LayerNorm gradient validation issue. See the
[consolidated status and wiring](docs/kernels/reports/TRIMUL_STATUS.html).
[AdaLN and attention fixes](docs/engine-adaln-attention-fixes.md) cover the H100
shared-memory alignment failure and L8192 attention training memory.

## Evaluation and documentation

- [FoldBench evaluation conventions](docs/foldbench-evaluation.md)
- [Known issues and validation limits](docs/known-issues.md)
- [Technical report draft and build instructions](docs/paper/README.md)

Generated FoldBench results live under `runs/eval_results/` and are excluded from Git.

## TriMul update (2026-09-17)

[Kernel changes, validation, and reinstall instructions](docs/trimul-release-20260917.md).
[Inference kernel wiring](docs/kernels/reports/TRIMUL_INFERENCE.svg) · [Wiring details](docs/archive/trimul-fusion/inference.md).
