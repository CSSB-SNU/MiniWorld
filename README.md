# MiniWorld

## Training pipeline

The stages are distogram training (1a/1b), diffusion training (2a/2b), and
confidence training (3a/3b). See [the pipeline guide](docs/pipeline.md) for
configuration names, crop sizes, checkpoint handoffs, and version variants.

## Engine setup

Check out the pinned submodules, install the selected Pixi environment, and
apply the patches to the pinned `miniworld-engine` package:

```bash
git submodule update --init --recursive
pixi install -e cu128
pixi run -e cu128 engine-setup
```

Run `engine-setup` again after reinstalling the engine dependency. It checks
source compatibility and is safe to repeat. FlashAttention and MathDx kernel
prerequisites still need to be available for the selected GPU backend; see the
environment comments in [pyproject.toml](pyproject.toml).

The integration preserves the SWA block equations and checkpoint layout.
[The engine audit](docs/engine-wiring-audit.md) records H100 forward/backward
checks, numerical differences, CUDA Graph results, and their limits.
