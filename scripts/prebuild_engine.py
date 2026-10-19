"""Build every engine CUDA extension and length-bucketed cubin the production models use, before the first real call.

The engine JIT-builds its CUDA extensions with nvcc on first use, one at a time, so without this the first model call after
creating an environment or bumping the engine pin waits 14-35 minutes on an H100. This script runs the production models (random
weights, no checkpoint) at every shape inference and training will use, so the first real call loads cached ``.so`` / ``.cubin``
files instead:

* ``infer``: the distogram trunk (``MiniSWAModel``, forward under its precision and ``no_grad``, as the FoldBench checks run it) and
  the phase-2 diffusion model (``Client.inference``: trunk + diffusion sampler steps, as ``run_miniworld_diffusion_inference.py``
  runs it) at every token rung of the inference padding ladder, each with the atom rung its targets pad to (~8 atoms per token);
  the atom rungs no token rung reaches ride on the largest token rung. MSA 2048, 4 templates (the ladder's single values).
* ``train``: forward + backward at each training config's crop (the bucket multiples), every recycle depth the graph trainer
  captures, under the trainer's precision and engine settings. Eager: CUDA-graph capture builds nothing new.

nvcc artifact keys depend on sources / flags / channel widths, not on the token length, but some kernel choices are bucketed by
length (TriMul cells by L, SWA atom buckets), which is why every rung runs. With ``--jobs N`` the shapes are dealt to N worker
processes (GPUs round-robin over the visible ones) that share the cache, so different extensions build at once; a worker that
needs an extension another one is building waits for it. Shapes that need a whole GPU run at the end, one process per GPU.

At the end a NEW process runs a representative set of inputs (one shape per model and mode, an off-ladder token/atom pair, and the
largest rung) between two snapshots of every build artifact (``*.so``, ``*.cubin``, ``*.o``, ``*.fatbin``, ``*.ptx``) under the
cache roots and fails if anything was created or rebuilt, i.e. nvcc ran. On PASS it writes ``<jit root>/prebuild/<arch>.json``
(the engine revision it built), which ``pixi run engine-setup`` checks.

    pixi run -e cu128 engine-prebuild                     # = python scripts/prebuild_engine.py --mode all
    python scripts/prebuild_engine.py --jobs 8            # 8 workers over the visible GPUs
    python scripts/prebuild_engine.py --mode infer --tokens 384,768 --atoms 4096,6144
    python scripts/prebuild_engine.py --verify-only       # only the no-rebuild check
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
CONFIG_DIR = REPO / "configs" / "miniworld"

#: The production models: the distogram trunk FoldBench is checked with, the phase-2 model of the latest run.
DEFAULT_INFER_CONFIGS = (
    "phase1b_distogram_medium_v200_b200",
    "phase2a_diffusion_medium_v201_b200",
)
#: The production training crops: phase 1a (L384) and 1b (L768) of the medium distogram trunk, phase 2a joint (L384).
DEFAULT_TRAIN_CONFIGS = (
    "phase1a_distogram_medium_v200",
    "phase1b_distogram_medium_v200",
    "phase2a_diffusion_medium_v201_b200",
)
#: The inference padding ladder of scripts/run_miniworld_diffusion_inference.py (kept equal by tests/test_prebuild_engine.py).
TOKEN_BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 2560)
ATOM_BUCKETS = (1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152)
INFER_MSA = 2048
INFER_TEMPLATES = 4
#: Atoms per token a target of a token rung pads to: the FoldBench shapes put almost every target of token rung T in the atom
#: rung that holds 8 T (128/1024, 256/2048, 384/4096, 512/4096, ..., 2560/24576).
ATOMS_PER_TOKEN = 8
INFER_RECYCLES = 2  # the recycle loop's second pass runs the recycled-input path; more passes repeat the same kernels
INFER_DIFFUSION_STEPS = 2
#: A shape this large runs alone on its GPU when several workers share one (it needs most of an 80 GB H100).
HEAVY_INFER_TOKENS = 2048
HEAVY_INFER_ATOMS = 32768
HEAVY_TRAIN_TOKENS = 768
ARTIFACT_SUFFIXES = (".so", ".cubin", ".o", ".fatbin", ".ptx")
RESULT_TAG = "[prebuild-result] "


# ----------------------------------------------------------------------------------------------------------------- shapes
def _bucket(value: int, ladder: tuple[int, ...]) -> int:
    return next((rung for rung in ladder if value <= rung), ladder[-1])


def default_infer_shapes(
    tokens: tuple[int, ...] = TOKEN_BUCKETS, atoms: tuple[int, ...] = ATOM_BUCKETS
) -> list[tuple[int, int]]:
    """Every token rung with the atom rung its targets pad to, then the atom rungs nothing reached on the largest token rung."""
    shapes = [(t, _bucket(ATOMS_PER_TOKEN * t, atoms)) for t in tokens]
    used = {a for _, a in shapes}
    shapes += [(tokens[-1], a) for a in atoms if a not in used]
    return shapes


def _ints(text: str | None) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x] if text else []


def infer_shapes_from_args(args) -> list[tuple[int, int]]:
    tokens, atoms = _ints(args.tokens), _ints(args.atoms)
    if tokens and atoms:
        if len(tokens) == len(atoms):
            return list(zip(tokens, atoms))
        return [(t, a) for t in tokens for a in atoms]
    if tokens:
        return [(t, _bucket(ATOMS_PER_TOKEN * t, ATOM_BUCKETS)) for t in tokens]
    if atoms:
        return [
            (
                next(
                    (t for t in TOKEN_BUCKETS if ATOMS_PER_TOKEN * t >= a),
                    TOKEN_BUCKETS[-1],
                ),
                a,
            )
            for a in atoms
        ]
    return default_infer_shapes()


def _ceil(value: int, multiple: int | None) -> int:
    return value if not multiple or multiple <= 0 else -(-value // multiple) * multiple


def _compose(name: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        return compose(config_name=name)


def config_kind(raw) -> str:
    """``diffusion`` (phase-2 Client) or ``distogram`` (MiniSWAModel) from a composed config."""
    if "diffuser" in raw:
        return "diffusion"
    trunk = raw.get("model", {}).get("trunk", {}) or {}
    if trunk.get("diffusion") is not None:
        raise SystemExit(
            "distogram-diffusion (v1.3) configs are not production; prebuild them with their own check script"
        )
    return "distogram"


def load_config(name: str):
    """(kind, validated config) for a config name under configs/miniworld."""
    raw = _compose(name)
    kind = config_kind(raw)
    if kind == "diffusion":
        from run_miniworld_diffusion_inference import Config
    else:
        from run_miniworld_distogram_train import Config
    return kind, Config.model_validate(raw)


def train_shape(cfg) -> tuple[int, int, int]:
    """(msa, tokens, atoms) of a training batch: the crop rounded to the bucket multiples, as the collate pads it."""
    msa_cfg = cfg.data.msa
    depth = max(
        [
            msa_cfg.max_msa_depth,
            *(getattr(msa_cfg, "max_msa_depth_by_source", None) or {}).values(),
        ]
    )
    return (
        _ceil(depth, cfg.train.bucket_msa_multiple),
        _ceil(cfg.data.crop.max_tokens, cfg.train.bucket_token_multiple),
        _ceil(cfg.data.crop.max_atoms, cfg.train.bucket_atom_multiple),
    )


def build_tasks(args) -> list[dict]:
    """The shape list, interleaved so that the first tasks of N workers touch different kernels (training vs inference, distogram
    vs diffusion, different rungs) and build different extensions at once."""
    infer, train = [], []
    if args.mode in ("infer", "all"):
        shapes = infer_shapes_from_args(args)
        for name in args.config or DEFAULT_INFER_CONFIGS:
            kind, _ = load_config(name)
            infer.append(
                [
                    dict(
                        mode="infer",
                        kind=kind,
                        config=name,
                        tokens=t,
                        atoms=a,
                        msa=INFER_MSA,
                        templates=INFER_TEMPLATES,
                    )
                    for t, a in shapes
                ]
            )
    if args.mode in ("train", "all"):
        for name in args.config or DEFAULT_TRAIN_CONFIGS:
            kind, cfg = load_config(name)
            msa, tokens, atoms = train_shape(cfg)
            train.append(
                dict(
                    mode="train",
                    kind=kind,
                    config=name,
                    tokens=tokens,
                    atoms=atoms,
                    msa=msa,
                    templates=4,
                )
            )
    # Training first (its kernels are not the inference ones), then the inference lists dealt alternately, the i-th list rotated
    # by i/n of its length, so that concurrent workers start on different models and rungs.
    order: list[dict] = list(train)
    rotated = []
    for i, lst in enumerate(infer):
        k = (i * len(lst) // len(infer)) % len(lst) if lst else 0
        rotated.append(lst[k:] + lst[:k])
    for j in range(max((len(lst) for lst in rotated), default=0)):
        order += [lst[j] for lst in rotated if j < len(lst)]
    for i, task in enumerate(order):
        task["id"] = i
        task["heavy"] = (
            task["tokens"] >= HEAVY_TRAIN_TOKENS
            if task["mode"] == "train"
            else task["tokens"] >= HEAVY_INFER_TOKENS
            or task["atoms"] >= HEAVY_INFER_ATOMS
        )
    return order


def verify_tasks(args) -> list[dict]:
    """One shape per model and mode, an off-ladder token/atom pair (FoldBench has 72 targets at 768/4096), the largest rung."""
    tasks = []
    if args.mode in ("infer", "all"):
        shapes = infer_shapes_from_args(args)
        largest = max(shapes, key=lambda s: (s[0], s[1]))
        mid = (384, 4096) if (384, 4096) in shapes else shapes[len(shapes) // 2]
        for name in args.config or DEFAULT_INFER_CONFIGS:
            kind, _ = load_config(name)
            picks = [mid, largest]
            if (
                kind == "diffusion"
                and (768, 4096) not in shapes
                and not (args.tokens or args.atoms)
            ):
                picks.append((768, 4096))
            for t, a in dict.fromkeys(picks):
                tasks.append(
                    dict(
                        mode="infer",
                        kind=kind,
                        config=name,
                        tokens=t,
                        atoms=a,
                        msa=INFER_MSA,
                        templates=INFER_TEMPLATES,
                    )
                )
    if args.mode in ("train", "all"):
        seen = set()
        for name in args.config or DEFAULT_TRAIN_CONFIGS:
            kind, cfg = load_config(name)
            if kind in seen:
                continue
            seen.add(kind)
            msa, tokens, atoms = train_shape(cfg)
            tasks.append(
                dict(
                    mode="train",
                    kind=kind,
                    config=name,
                    tokens=tokens,
                    atoms=atoms,
                    msa=msa,
                    templates=4,
                )
            )
    for i, task in enumerate(tasks):
        task["id"], task["heavy"] = i, False
    return tasks


def describe(task: dict) -> str:
    return f"{task['mode']} {task['kind']} tokens={task['tokens']} atoms={task['atoms']} msa={task['msa']} ({task['config']})"


# --------------------------------------------------------------------------------------------------------- cache snapshot
def cache_roots() -> list[Path]:
    """Where JIT builds land: this environment's engine JIT root (miniworld's per-environment logic), torch's
    TORCH_EXTENSIONS_DIR (set by ``import miniworld`` unless the run set it), and the engine's ``settings.engine_jit_root()``."""
    from miniworld_engine import settings

    import miniworld  # noqa: F401 -- sets engine_jit_root and TORCH_EXTENSIONS_DIR for this environment
    from miniworld import engine_jit_root

    roots = [engine_jit_root(), settings.engine_jit_root()]
    try:  # the engine's own list where it has one (e726bf7f and later)
        from miniworld_engine.kernels._nvcc import cache_roots as engine_cache_roots

        roots += list(engine_cache_roots())
    except ImportError:
        pass
    ext = os.environ.get("TORCH_EXTENSIONS_DIR")
    if ext:
        roots.append(Path(ext))
    else:
        from torch.utils.cpp_extension import get_default_build_root

        roots.append(Path(get_default_build_root()))
    out: list[Path] = []
    for r in roots:
        if r is not None and Path(r).resolve() not in out:
            out.append(Path(r).resolve())
    # a root nested in another is walked once
    return [r for r in out if not any(o != r and o in r.parents for o in out)]


def snapshot(roots: list[Path]) -> dict[str, tuple[int, int]]:
    files = {}
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _, names in os.walk(root):
            for name in names:
                if name.endswith(ARTIFACT_SUFFIXES):
                    path = os.path.join(dirpath, name)
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    files[path] = (st.st_mtime_ns, st.st_size)
    return files


def diff(before: dict, after: dict) -> tuple[list[str], list[str]]:
    created = sorted(p for p in after if p not in before)
    rebuilt = sorted(p for p in after if p in before and after[p] != before[p])
    return created, rebuilt


def count_kinds(paths: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in paths:
        suffix = Path(p).suffix
        out[suffix] = out.get(suffix, 0) + 1
    return out


# ------------------------------------------------------------------------------------------------------------ the worker
class _Runner:
    """One GPU process: builds each model once, then runs tasks on it."""

    def __init__(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        import torch

        import miniworld  # noqa: F401 -- per-environment JIT root + TORCH_EXTENSIONS_DIR before any engine extension loads

        self.torch = torch
        torch.set_float32_matmul_precision(
            "medium"
        )  # what the inference and training scripts set at import
        torch.cuda.set_device(0)
        self.device = torch.device("cuda", 0)
        self._patch_lock_waits()
        from miniworld_engine import settings

        self.settings = settings
        self.baseline = settings.current()
        self.models: dict[tuple[str, str], tuple] = {}

    @staticmethod
    def _patch_lock_waits() -> None:
        """Wait for another worker's build instead of giving up after the engine's 10 minutes, and do not take a live lock of a
        long build for a leftover. Harmless where a later engine already waits without a limit."""
        try:
            from miniworld_engine.kernels import _nvcc
        except ImportError:
            return
        clear = getattr(_nvcc, "clear_stale_lock", None)
        if getattr(_nvcc, "wait_for_lock", None) is None or clear is None:
            return
        if hasattr(
            _nvcc, "built_this_process"
        ):  # e726bf7f and later wait up to 2 h themselves, with a waiting line
            return

        def wait_for_lock(lock, limit=None):  # noqa: ARG001 -- the engine's bound is what this replaces
            if not lock.exists():
                return
            print(
                f"[prebuild] another worker is building this extension; waiting for {lock}",
                file=sys.stderr,
                flush=True,
            )
            deadline = time.time() + 4 * 3600.0
            while lock.exists() and time.time() < deadline:
                if clear(
                    lock
                ):  # its builder was killed: the lock aged into a leftover, take over the build
                    return
                time.sleep(2.0)

        _nvcc.wait_for_lock = wait_for_lock

    def _restore(self, state) -> None:
        self.settings.configure(
            **{f.name: getattr(state, f.name) for f in dataclasses.fields(state)}
        )

    def gpu(self) -> dict:
        torch = self.torch
        major, minor = torch.cuda.get_device_capability(0)
        return {"gpu": torch.cuda.get_device_name(0), "arch": f"sm{major}{minor}"}

    # -- models
    def _model(self, task: dict):
        key = (task["config"], task["mode"])
        if key not in self.models:
            self._restore(self.baseline)
            builder = {
                ("infer", "distogram"): self._infer_distogram,
                ("infer", "diffusion"): self._infer_diffusion,
                ("train", "distogram"): self._train_distogram,
                ("train", "diffusion"): self._train_diffusion,
            }[(task["mode"], task["kind"])]
            self.models[key] = (*builder(task["config"]), self.settings.current())
        model, run, state = self.models[key]
        self._restore(state)  # settings are process-wide: each model runs under its own
        return run

    def _num_res_class(self, cfg) -> int:
        return int(getattr(getattr(cfg.model, "shared", None), "num_res_class", 32))

    def _batch(self, script: str, cfg, task: dict):
        module = __import__(script)
        return module._build_precompile_batch(  # noqa: SLF001
            device=self.device,
            msa_depth=task["msa"],
            n_tokens=task["tokens"],
            n_atoms=task["atoms"],
            n_templates=task["templates"],
            num_res_class=self._num_res_class(cfg),
        )

    def _infer_distogram(self, name: str):
        """As the FoldBench / checkpoint checks run the trunk: MiniSWAModel in its precision, no_grad + model autocast."""
        from miniworld.models.distogram_only import MiniSWAModel
        from miniworld.training.engine_backend import configure_engine_backend
        from miniworld.training.precision import apply_precision, model_autocast

        _, cfg = load_config(name)
        configure_engine_backend(cfg.train.engine_backend)
        model = (
            apply_precision(MiniSWAModel(cfg.model), cfg.train.precision)
            .to(self.device)
            .eval()
        )

        def run(task):
            batch = self._batch("run_miniworld_distogram_train", cfg, task)
            model._forced_n_recycle = INFER_RECYCLES  # noqa: SLF001
            with self.torch.no_grad(), model_autocast(model):
                out = model(
                    batch.msa,
                    batch.reference,
                    batch.scheme,
                    batch.sequence,
                    batch.structure,
                    batch.template,
                )
            del out, batch

        return model, run

    def _infer_diffusion(self, name: str):
        """As scripts/run_miniworld_diffusion_inference.py foldbench runs it: Client.inference (trunk once, then the sampler)."""
        from phase2_graph_trainer import GraphFabric

        from miniworld.models.diffusion import Client

        _, cfg = load_config(name)
        cfg.train.param_policy.enabled = False
        client = Client(
            Client.Config(
                train=cfg.train, model=cfg.model, diffuser=cfg.diffuser, loss=cfg.loss
            )
        )
        fabric = GraphFabric()
        fabric.launch()
        client.setup(fabric=fabric)
        client.model.eval()

        def run(task):
            batch = self._batch("run_miniworld_diffusion_train", cfg, task)
            out = client.inference(
                batch,
                timesteps=INFER_DIFFUSION_STEPS,
                n_samples=1,
                n_recycle=INFER_RECYCLES,
                sample_batch=False,
            )
            del out, batch

        return client, run

    def _train_distogram(self, name: str):
        """As scripts/random_recycle_graph_trainer.py builds its graphs, eagerly: every recycle depth, its loss and precision."""
        from random_recycle_graph_trainer import (
            attach_pair_weight,
            graph_objective,
            prepare_template_graph,
        )

        from miniworld.configs import TemplateConfig
        from miniworld.models.distogram_only import MiniSWAModel
        from miniworld.training.engine_backend import (
            apply_run_options,
            configure_engine_backend,
            configure_fused_msa_train,
        )
        from miniworld.training.precision import apply_precision, model_autocast

        torch = self.torch
        _, cfg = load_config(name)
        apply_run_options(cfg.train)
        configure_engine_backend(cfg.train.engine_backend)
        configure_fused_msa_train(bool(getattr(cfg.train, "fused_msa_train", False)))
        model = (
            apply_precision(MiniSWAModel(cfg.model), cfg.train.precision)
            .to(self.device)
            .train()
        )
        loss_fn, forward_kwargs = graph_objective(cfg)
        patched = []

        def run(task):
            static = self._batch("run_miniworld_distogram_train", cfg, task)
            attach_pair_weight(static, cfg.loss)
            if not patched:
                prepare_template_graph(model, static, TemplateConfig().n_templates)
                patched.append(True)
            else:  # the template forward is already wrapped: only the static flag it reads
                static.template._graph_present = torch.tensor(
                    static.template.mask.shape[1] > 0, device=self.device
                )
            for r in range(1, model.n_recycle_max + 1):
                model._forced_n_recycle = r  # noqa: SLF001
                amp = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if getattr(cfg.train, "graph_amp", False)
                    else model_autocast(model)
                )
                with amp:
                    y = model(
                        msa=static.msa,
                        reference=static.reference,
                        scheme=static.scheme,
                        sequence=static.sequence,
                        structure=static.structure,
                        template=static.template,
                        **forward_kwargs,
                    )
                    loss = loss_fn(y, static)
                (loss / cfg.train.grad_accum_steps).backward()
                del y, loss
                model.zero_grad(set_to_none=True)
            del static

        return model, run

    def _train_diffusion(self, name: str):
        """As scripts/phase2_graph_trainer.py runs a micro-step (its ``_body``), eagerly, at every recycle depth it captures."""
        import copy

        from phase2_graph_trainer import GraphFabric, Phase2GraphTrainer
        from random_recycle_graph_trainer import prepare_template_graph

        from miniworld.configs import TemplateConfig
        from miniworld.models.diffusion import Client
        from miniworld.training.engine_backend import apply_run_options

        torch = self.torch
        _, cfg = load_config(name)
        apply_run_options(cfg.train)
        fabric = GraphFabric()
        fabric.launch()
        client = Client(
            Client.Config(
                train=cfg.train, model=cfg.model, diffuser=cfg.diffuser, loss=cfg.loss
            )
        )
        client.setup(
            fabric=fabric, gradient_accumulation_steps=cfg.train.grad_accum_steps
        )
        client.model.train()
        trainer = Phase2GraphTrainer(client, cfg)
        raw = client.model
        patched = []

        def run(task):
            first = self._batch("run_miniworld_diffusion_train", cfg, task)
            trainer.use_bond = cfg.loss.bond_loss != 0
            trainer.static = copy.deepcopy(first)
            if not patched:
                prepare_template_graph(raw, trainer.static, TemplateConfig().n_templates)
                patched.append(True)
            elif hasattr(raw, "temp_embedder") and getattr(raw, "use_template", False):
                trainer.static.template._graph_present = torch.tensor(
                    True, device=self.device
                )
            shape = trainer.static.structure.atom_pos.shape[:2]
            trainer.atom_weight = torch.empty(
                shape, device=self.device, dtype=torch.float32
            )
            trainer.atom_nuc = torch.empty(shape, device=self.device, dtype=torch.bool)
            trainer._fill_atom_weight(first)  # noqa: SLF001
            n_atom = shape[1]
            trainer.bond_i = torch.zeros(n_atom, dtype=torch.long, device=self.device)
            trainer.bond_j = torch.zeros(n_atom, dtype=torch.long, device=self.device)
            trainer.bond_valid = torch.zeros(
                n_atom, dtype=torch.bool, device=self.device
            )
            if trainer.use_bond:
                trainer._fill_bond_pairs(first)  # noqa: SLF001
            for r in trainer._recycle_depths(raw):  # noqa: SLF001
                raw._forced_n_recycle = r  # noqa: SLF001
                trainer._body()  # noqa: SLF001
                raw.zero_grad(set_to_none=True)
            raw._forced_n_recycle = None  # noqa: SLF001
            del first
            trainer.static = None

        return client, run

    # -- one task
    def run(self, task: dict) -> dict:
        torch = self.torch
        result = {"id": task["id"], **self.gpu()}
        torch.cuda.reset_peak_memory_stats()
        built_before = len(_built_this_process() or ())
        t0 = time.perf_counter()
        try:
            run = self._model(task)
            run(task)
            torch.cuda.synchronize()
            result["status"] = "ok"
        except torch.cuda.OutOfMemoryError as exc:
            result["status"] = "oom"
            result["error"] = str(exc).splitlines()[0][:300]
        except Exception as exc:  # noqa: BLE001 -- one shape must not stop the others; the parent reports it
            traceback.print_exc()
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        result["seconds"] = time.perf_counter() - t0
        result["peak_gib"] = torch.cuda.max_memory_reserved() / 2**30
        # what this process compiled during the task, by the engine's own count (None: an engine without the counter)
        built = _built_this_process()
        result["built"] = (
            None if built is None else [f"{b[0]} ({b[1]})" for b in built[built_before:]]
        )
        import gc

        gc.collect()
        return result


def _built_this_process() -> tuple | None:
    """What this process has compiled so far, by the engine's own count; None for an engine without the counter."""
    try:
        from miniworld_engine.kernels._nvcc import built_this_process
    except ImportError:
        return None
    return built_this_process()


def worker_main(result_fd: int) -> int:
    out = os.fdopen(result_fd, "w", buffering=1)
    try:
        runner = _Runner()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        out.write(
            RESULT_TAG + json.dumps({"fatal": traceback.format_exc()[-2000:]}) + "\n"
        )
        return 1
    out.write(RESULT_TAG + json.dumps({"ready": True, **runner.gpu()}) + "\n")
    while True:
        line = sys.stdin.readline()
        if not line:  # the parent went away
            break
        line = line.strip()
        if not line:
            continue
        task = json.loads(line)
        if task.get("exit"):
            break
        print(f"[worker] {describe(task)}", flush=True)
        result = runner.run(task)
        print(f"[worker] -> {result['status']} in {result['seconds']:.1f} s", flush=True)
        out.write(RESULT_TAG + json.dumps(result) + "\n")
    return 0


# ------------------------------------------------------------------------------------------------------------ the parent
def visible_gpus() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [str(i) for i, line in enumerate(out.splitlines()) if line.startswith("GPU ")]


class _Worker:
    """A worker process on one GPU; tasks go in on stdin, results come back on a pipe."""

    def __init__(self, gpu: str, log: Path) -> None:
        read_fd, write_fd = os.pipe()
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu,
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        env.setdefault("OMP_NUM_THREADS", "4")
        # the environment's libstdc++ first, as the run scripts do: a module-loaded older GCC on LD_LIBRARY_PATH otherwise
        # shadows it and extensions linked against the newer one (matplotlib, the JIT .so) fail to load
        lib = os.path.join(sys.prefix, "lib")
        paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
        if not paths or paths[0] != lib:
            env["LD_LIBRARY_PATH"] = ":".join([lib, *paths])
        self.log = log
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = log.open("a")
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--_worker", str(write_fd)],
            stdin=subprocess.PIPE,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
            pass_fds=(write_fd,),
            text=True,
            cwd=str(REPO),
        )
        os.close(write_fd)
        self._results = os.fdopen(read_fd)
        self.info = self._read()  # {"ready", "gpu", "arch"} or {"fatal"}

    def _read(self) -> dict:
        while True:
            line = self._results.readline()
            if not line:
                return {
                    "fatal": f"worker exited (code {self.proc.wait()}); see {self.log}"
                }
            if line.startswith(RESULT_TAG):
                return json.loads(line[len(RESULT_TAG) :])

    def run(self, task: dict) -> dict:
        try:
            self.proc.stdin.write(json.dumps(task) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return {
                "id": task["id"],
                "status": "error",
                "error": f"worker died; see {self.log}",
            }
        result = self._read()
        if "fatal" in result:
            return {"id": task["id"], "status": "error", "error": result["fatal"]}
        return result

    def close(self) -> None:
        with contextlib.suppress(BrokenPipeError, OSError):
            self.proc.stdin.write(json.dumps({"exit": True}) + "\n")
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._results.close()
        self._log_file.close()


class Progress:
    def __init__(self, total: int, label: str) -> None:
        self.total, self.done, self.label = total, 0, label
        self.lock = threading.Lock()

    def report(self, task: dict, result: dict, gpu: str) -> None:
        with self.lock:
            self.done += 1
            status = result.get("status", "error")
            extra = (
                ""
                if status == "ok"
                else f" [{status.upper()}: {result.get('error', '')}]"
            )
            if result.get("built") is not None:
                extra = f", {len(result['built'])} compiled here" + extra
            print(
                f"[{self.label}] {self.done}/{self.total} {task['mode']} {task['kind']} tokens={task['tokens']} "
                f"atoms={task['atoms']} msa={task['msa']}: {result.get('seconds', 0.0):.1f} s "
                f"(gpu {gpu}, peak {result.get('peak_gib', 0.0):.1f} GiB{extra})",
                flush=True,
            )


def run_pool(
    tasks: list[dict],
    gpus: list[str],
    slots: int,
    log_dir: Path,
    label: str,
    *,
    fresh: bool = False,
    progress: Progress | None = None,
) -> tuple[dict[int, dict], dict]:
    """Run ``tasks`` over ``slots`` workers (GPUs round-robin). ``fresh``: a new process for every task."""
    todo: queue.Queue = queue.Queue()
    for task in tasks:
        todo.put(task)
    results: dict[int, dict] = {}
    info: dict = {}
    progress = progress or Progress(len(tasks), label)

    def slot(index: int) -> None:
        gpu = gpus[index % len(gpus)]
        worker = None
        while True:
            try:
                task = todo.get_nowait()
            except queue.Empty:
                break
            if worker is None:
                worker = _Worker(gpu, log_dir / f"{label}-worker{index}-gpu{gpu}.log")
                if "fatal" in worker.info:
                    results[task["id"]] = {
                        "id": task["id"],
                        "status": "error",
                        "error": worker.info["fatal"][-500:],
                    }
                    progress.report(task, results[task["id"]], gpu)
                    worker.close()
                    worker = None
                    continue
                info.update(worker.info)
            result = worker.run(task)
            results[task["id"]] = result
            progress.report(task, result, gpu)
            if fresh or result.get("status") != "ok" or worker.proc.poll() is not None:
                worker.close()  # an OOM or an error leaves a process not worth reusing
                worker = None
        if worker is not None:
            worker.close()

    threads = [
        threading.Thread(target=slot, args=(i,), daemon=True)
        for i in range(max(1, min(slots, len(tasks))))
    ]
    for i, thread in enumerate(threads):
        thread.start()
        if i + 1 < len(threads):
            time.sleep(1.0)  # stagger the imports a little
    for thread in threads:
        thread.join()
    return results, info


def prebuild(
    tasks: list[dict], gpus: list[str], jobs: int, log_dir: Path
) -> tuple[dict[int, dict], dict]:
    """Light shapes over ``jobs`` workers, then the heavy ones (and anything that ran out of memory next to another worker) one
    process per GPU."""
    shared = jobs > len(
        gpus
    )  # more than one worker per GPU: the big shapes need a GPU of their own
    first = [t for t in tasks if not (shared and t["heavy"])]
    progress = Progress(len(tasks), "prebuild")
    results, info = run_pool(first, gpus, jobs, log_dir, "prebuild", progress=progress)
    again = [
        t
        for t in tasks
        if (shared and t["heavy"]) or results.get(t["id"], {}).get("status") == "oom"
    ]
    if again:
        retried = [t for t in again if t["id"] in results]
        if retried:
            print(
                f"[prebuild] {len(retried)} shape(s) ran out of memory next to another worker; retrying alone on a GPU",
                flush=True,
            )
            progress.total += len(retried)
        more, more_info = run_pool(
            again, gpus, len(gpus), log_dir, "prebuild-solo", progress=progress
        )
        results.update(more)
        info.update(more_info)
    return results, info


def verify(
    tasks: list[dict], gpus: list[str], roots: list[Path], log_dir: Path
) -> tuple[bool, list[str], dict[int, dict]]:
    """New processes run ``tasks``; PASS when no build artifact under ``roots`` was created or rebuilt and every task ran."""
    before = snapshot(roots)
    results, _ = run_pool(tasks, gpus, len(gpus), log_dir, "verify", fresh=True)
    after = snapshot(roots)
    created, rebuilt = diff(before, after)
    changed = [f"created {p}" for p in created] + [f"rebuilt {p}" for p in rebuilt]
    # the engine's own count, where it keeps one: a compile that wrote outside the snapshot roots still fails the check
    changed += [
        f"compiled {name} (verify task {i})"
        for i, r in sorted(results.items())
        for name in (r.get("built") or [])
    ]
    ran = all(results.get(t["id"], {}).get("status") == "ok" for t in tasks)
    return (not changed and ran), changed, results


def stamp_path(root: Path, arch: str) -> Path:
    return root / "prebuild" / f"{arch}.json"


def engine_identity() -> dict:
    sys.path.insert(0, str(SCRIPTS))
    from check_engine_install import installed_engine

    version, identity, _ = installed_engine()
    return {"engine_version": version, "engine": identity}


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("infer", "train", "all"), default="all")
    parser.add_argument(
        "--config",
        action="append",
        help="config name under configs/miniworld (repeatable); default: the production set "
        f"(infer: {', '.join(DEFAULT_INFER_CONFIGS)}; train: {', '.join(DEFAULT_TRAIN_CONFIGS)})",
    )
    parser.add_argument(
        "--tokens",
        help="inference token rungs, comma-separated (default: the inference ladder)",
    )
    parser.add_argument(
        "--atoms",
        help="inference atom rungs, comma-separated; zipped with --tokens when both have the same length",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="worker processes (default: one per visible GPU)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="only check that the representative shapes build nothing",
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="no verification (and no stamp)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="worker logs (default: <jit root>/prebuild/logs/<time>)",
    )
    parser.add_argument("--_worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        return worker_main(args._worker)

    sys.path.insert(0, str(SCRIPTS))
    gpus = visible_gpus()
    if not gpus:
        print(
            "[prebuild] no GPU visible: run this on a GPU node of the architecture you want to build for",
            file=sys.stderr,
        )
        return 2
    jobs = args.jobs or len(gpus)
    roots = cache_roots()
    from miniworld import engine_jit_root

    jit_root = engine_jit_root()
    log_dir = args.log_dir or jit_root / "prebuild" / "logs" / time.strftime(
        "%Y%m%d-%H%M%S"
    )
    identity = engine_identity()
    print(
        f"[prebuild] engine {identity['engine_version']} ({identity['engine']}) on {socket.gethostname()}, GPUs {','.join(gpus)}",
        flush=True,
    )
    print(f"[prebuild] cache roots: {', '.join(map(str, roots))}", flush=True)
    print(f"[prebuild] worker logs: {log_dir}", flush=True)
    exit_code = 0
    summary: dict = {}
    if not args.verify_only:
        tasks = build_tasks(args)
        print(
            f"[prebuild] {len(tasks)} shapes, {jobs} worker(s) over {len(gpus)} GPU(s)",
            flush=True,
        )
        before = snapshot(roots)
        t0 = time.perf_counter()
        results, info = prebuild(tasks, gpus, jobs, log_dir)
        seconds = time.perf_counter() - t0
        created, rebuilt = diff(before, snapshot(roots))
        built = count_kinds(created + rebuilt)
        failed = [
            t
            for t in tasks
            if results.get(t["id"], {}).get("status") not in ("ok", "oom")
        ]
        no_fit = [t for t in tasks if results.get(t["id"], {}).get("status") == "oom"]
        n_ext = built.get(".so", 0)
        print(
            f"[prebuild] done in {seconds / 60:.1f} min ({seconds:.0f} s): {n_ext} extension(s) (.so) and "
            f"{built.get('.cubin', 0)} cubin(s) built, {len(created) + len(rebuilt)} artifact files written "
            f"({', '.join(f'{k} {v}' for k, v in sorted(built.items())) or 'none'})",
            flush=True,
        )
        for t in no_fit:
            print(
                f"[prebuild] WARNING: does not fit this GPU alone, not built: {describe(t)}",
                flush=True,
            )
        for t in failed:
            print(
                f"[prebuild] FAILED: {describe(t)}: {results.get(t['id'], {}).get('error')}",
                flush=True,
            )
        if failed:
            exit_code = 1
        summary = {
            "seconds": round(seconds, 1),
            "jobs": jobs,
            "gpus": len(gpus),
            "built": built,
            "shapes": [describe(t) for t in tasks],
            "not_fitting": [describe(t) for t in no_fit],
            "failed": [describe(t) for t in failed],
            **info,
        }
    if args.skip_verify:
        return exit_code
    vtasks = verify_tasks(args)
    print(
        f"[verify] {len(vtasks)} representative shape(s) in new processes:", flush=True
    )
    for t in vtasks:
        print(f"[verify]   {describe(t)}", flush=True)
    t0 = time.perf_counter()
    ok, changed, vresults = verify(vtasks, gpus, roots, log_dir)
    vseconds = time.perf_counter() - t0
    for line in changed:
        print(f"[verify]   {line}", flush=True)
    for t in vtasks:
        r = vresults.get(t["id"], {})
        if r.get("status") != "ok":
            print(
                f"[verify]   did not run: {describe(t)}: {r.get('status')} {r.get('error', '')}",
                flush=True,
            )
    if ok:
        print(
            f"[verify] PASS: no build artifact created or rebuilt ({vseconds:.0f} s)",
            flush=True,
        )
    else:
        what = (
            f"{len(changed)} artifact(s) created or rebuilt (nvcc ran)"
            if changed
            else "a representative shape did not run"
        )
        print(f"[verify] FAIL: {what}", flush=True)
        exit_code = 1
    arch = next(
        (r.get("arch") for r in vresults.values() if r.get("arch")), summary.get("arch")
    )
    if ok and not args.verify_only and exit_code == 0 and arch:
        path = stamp_path(jit_root, arch)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = {
            **identity,
            "arch": arch,
            "gpu": next((r.get("gpu") for r in vresults.values() if r.get("gpu")), None),
            "host": socket.gethostname(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": args.mode,
            "configs": args.config
            or {"infer": DEFAULT_INFER_CONFIGS, "train": DEFAULT_TRAIN_CONFIGS},
            "verify": "PASS",
            "verify_seconds": round(vseconds, 1),
            **summary,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(stamp, indent=2) + "\n")
        tmp.replace(path)
        print(f"[prebuild] stamp: {path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
