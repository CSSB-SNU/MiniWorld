# MiniWorld documentation

Current guides live at the top level of `docs/`; dated validation runs, audits and one-off experiments are kept under
`archive/` unchanged (paths inside them were rewritten, content was not).

| Where | What |
|---|---|
| [repositories.md](repositories.md) | repository map, engine pin history, validation notes |
| [pipeline.md](pipeline.md) | training phases and configs |
| [known-issues.md](known-issues.md) | validation limits and open problems |
| [kernels/](kernels/) | kernel development guides and status reports |
| [operations/b200-environment.md](operations/b200-environment.md) | B200 nodes: environment, GPUs, lock file, tests |
| [operations/](operations/) | cluster and launcher procedures |
| [paper/](paper/) | technical report draft |
| [graphical-abstract/](graphical-abstract/) | figures |
| [archive/](archive/) | dated checks and records: `graph-training-20260923`, `trimul-fusion`, `repository-audit-20260927`, `history`, ... |

Related checks and commands: `bash scripts/check.sh` (lint, format, type check) and
`bash scripts/b200/run_tests.sh` (the test suite on a B200 node).

## Where the team-gm code went

team-gm no longer exists as a dependency. MiniWorld and miniworld-engine are the only two repositories.

| Was (`team_gm.*`) | Is now |
|---|---|
| `core.client`, `core.batch` | `miniworld.training.client`, `miniworld.training.batch` |
| `core.callbacks` (incl. `ModelEMA`) | `miniworld.training.callbacks` |
| `diffusion` (base, edm, decoupled_xpred, augmentation) | `miniworld_engine.architecture.diffusion` |
| `modules.blocks`, `modules.layers`, `modules.primitives`, `modules.exceptions`, `modules.tensor_layout` | `miniworld_engine.architecture.{blocks,layers,primitives,exceptions,tensor_layout}` |
| `utils.se3`, `utils.align`, `utils.transform` | `miniworld_engine.architecture.utils.*` |
| `utils.script_utils` | `miniworld.utils.script_utils` |
| `typing.typecheck` | `miniworld_engine.architecture.typing` |

Parameter names are unchanged, so checkpoints written by team-gm-based versions load as before. Documents below that
link into `libs/team-gm/...` describe the code as it was when they were written.

## Configuration instead of environment variables

The kernel and integration switches are fields of `miniworld_engine.settings.Settings`; a run sets them with
`train.engine_settings={field: value}` (or `settings.configure(...)`). The run options of the training scripts are `train.*`
fields (`configs/run_options.py`). Variables that are still read from the environment are the ones an external payload or
tool reads (`CUDA_*`, `XDG_CACHE_HOME`, `OPT_CORE_DIR`, `TRIMUL_NATIVE_BUILD_DIR`, ...) and the ones that have to reach child
processes or be read when the kernel modules import (`MINIWORLD_COMPILE_WRAP`, `MINIWORLD_BUILD_*`, ...). A removed variable that is
still exported is reported once when the engine imports its settings.
