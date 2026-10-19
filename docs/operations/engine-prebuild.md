# Engine prebuild

The engine JIT-compiles its CUDA extensions (torch extensions, `load_inline` modules, raw `nvcc -cubin` kernels) on first use,
one at a time. Without a prebuild, the first model call of every new environment or engine pin waits on nvcc: on an H100 the first
FoldBench target took 840-1085 s (2000 s in one checkpoint check), and a later target that reached a new token rung paid another
~90 s. `scripts/prebuild_engine.py` builds all of it in advance.

## When

After **creating the environment** (`pixi install -e cu128`) and after **every engine pin bump**, once on a GPU node of each
architecture production runs on:

| GPU | how |
|---|---|
| H100 (sm_90) | `sbatch scripts/slurm/engine_prebuild_h100.sbatch` from the repository root (2 GPUs, 8 workers) |
| B200 (sm_100) | `python scripts/prebuild_engine.py --jobs 8` in the B200 environment ([b200-environment.md](b200-environment.md)) |
| any | `pixi run -e cu128 engine-prebuild [--jobs N]` |

`pixi run -e cu128 engine-setup` prints, per architecture, whether this engine revision has been prebuilt (`ok`, `missing`, or
`stale (<revision it was built for>)`) and a hint when not.

## What it runs

Random weights, no checkpoint, synthetic dense batches (`_build_precompile_batch`):

| mode | model / config | shapes |
|---|---|---|
| infer | distogram trunk `phase1b_distogram_medium_v200_b200` (`MiniSWAModel`, bf16-mixed, `no_grad`, 2 recycles), as the FoldBench checks run it | 11 token/atom pairs below, MSA 2048, 4 templates |
| infer | phase-2 `phase2a_diffusion_medium_v201_b200` via `Client.inference` (trunk + 2 sampler steps, one sample), as `run_miniworld_diffusion_inference.py foldbench` runs it | same 11 pairs |
| train | `phase1a_distogram_medium_v200` (L384), `phase1b_distogram_medium_v200` (L768): forward + loss + backward at every recycle depth 1-4, the graph trainer's objective and precision | crop rounded to the bucket multiples (MSA 8192) |
| train | `phase2a_diffusion_medium_v201_b200`: the phase-2 graph trainer's micro-step (`Phase2GraphTrainer._body`), every captured recycle depth | L384 / 4096 atoms, 48 augments |

Inference pairs: every token rung of the padding ladder (`_TOKEN_BUCKETS`) with the atom rung its targets pad to (~8 atoms per
token: 128/1024, 256/2048, 384/4096, 512/4096, 768/6144, 1024/8192, 1536/12288, 2048/16384, 2560/24576 -- the cells that hold
almost every FoldBench target), plus the two atom rungs no token rung reaches (2560/32768, 2560/49152). nvcc artifacts are keyed
by sources, flags and channel widths, never by the length (engine `docs/kernels/jit-cache.md`); the rungs are there because some
kernel choices are bucketed by length (TriMul H100 cells by L, SWA atom buckets), so a new rung can need a new cubin.

`--mode infer|train|all`, `--config NAME` (repeatable, replaces the default set), `--tokens 384,768 --atoms 4096,6144` (inference
pairs, zipped), `--jobs N`, `--verify-only`, `--skip-verify`.

Workers: `--jobs N` processes, GPUs round-robin, all sharing the cache; a worker that needs an extension another one is building
waits for it (the engine's lock). With more workers than GPUs the shapes that need most of a GPU (inference >= 2048 tokens or
>= 32768 atoms, training >= L768) run at the end, one process per GPU, as does anything that ran out of memory next to another
worker. Worker logs: `<jit root>/prebuild/logs/<time>/`.

## Verification

At the end (and with `--verify-only`) NEW processes run a representative set -- one shape per model and mode, the off-ladder
pair 768/4096 (72 FoldBench targets), and the largest rung (2560/49152) -- between two snapshots of every `*.so`, `*.cubin`, `*.o`,
`*.fatbin`, `*.ptx` (with mtimes) under the cache roots (the environment's engine JIT root, `TORCH_EXTENSIONS_DIR`, the engine's
`settings.engine_jit_root()` and, on engines that have it, `kernels._nvcc.cache_roots()`), and also checks the engine's own
`built_this_process()` count. Anything created, rebuilt or compiled prints `FAIL` with the list and exits non-zero; otherwise
`PASS`, and the stamp `<jit root>/prebuild/<arch>.json` (engine revision, GPU, shapes, timings) is written.

## Cache location

`src/miniworld/__init__.py` gives each environment its own root, `~/.cache/miniworld_engine_jit/py310-<sha1(sys.prefix)[:10]>`
(`$XDG_CACHE_HOME` if set, or the run's `engine_jit_root`), and -- unless the run set it -- points `TORCH_EXTENSIONS_DIR` at
`<root>/torch_extensions`. Before, the extensions loaded without a `build_directory` went to torch's shared
`~/.cache/torch_extensions/py310_cu129`, so two checkouts on different engine pins rebuilt each other's extensions.

## Measured (H100, empty cache, engine e726bf7f, 2026-10-19)

`--mode all` (25 shapes), node02, the two runs partly overlapping on the same node:

| run | prebuild | verify (7 shapes, new processes) | built |
|---|---|---|---|
| `--jobs 8`, 2 GPUs | 35.8 min | 2.4 min, PASS | 36 extensions (.so), 13 cubins (111 files) |
| `--jobs 1`, 1 GPU | 58.7 min | 4.3 min, PASS | the same 36 + 13 |

A torch extension takes ~85 s to build and the trunk's extensions are needed in a fixed order, so the workers wait on each other
for most of them; the gain from `--jobs 8` comes from the training, diffusion and length-bucketed builds running next to the
trunk chain. With this engine the 128-token rung plus the training crops already build every artifact; the larger rungs add
none (the rungs stay in the list so a later bucketed kernel is caught).

First real call after the prebuild (phase-1b distogram checkpoint, `h100_ckpt_check.py` logic, 4 recycles, new process):

| target | padded shape | before (cache empty) | after prebuild | second call |
|---|---|---|---|---|
| 7xwz-assembly1 (147 tokens) | 256 / 2048 | first target: 840-2000 s | 5.3 s | 0.18 s |
| 8bh9-assembly1 (550 tokens) | 768 / 6144 | 87.7 s (first at that rung) | 0.44 s | 0.22 s |

No build artifact was created or rebuilt. The remaining seconds of a first call (and ~40 s for a phase-2 model, which includes
building the model) are not nvcc: model construction and the in-memory FlashAttention-4 (CuTeDSL) compile; enabling FA4's disk
cache (`FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`) did not change it measurably.


## Not covered

- Token counts above 2560 (the ladder rounds them to a multiple of 8, so every such target has its own shape) and atom counts
  above 49152; `--tokens/--atoms` builds a given shape.
- `--sample-batch` inference (the diffusion module then runs with n_samples augments instead of 1) and `torch.compile` /
  CUDA-graph-capture paths: none adds an nvcc target, but they are not exercised.
- Not nvcc: FlashAttention-4 (CuTeDSL) kernels compile in every new process (in memory; its disk cache is off by default), and
  Triton / inductor kernels go to their own caches.
- Distogram-diffusion (v1.3) and confidence (phase 3) configs; pass them with `--config` once they are production (v1.3 configs
  are refused, they have their own check script).
