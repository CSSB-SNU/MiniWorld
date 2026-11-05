"""Pre-flight checks for the v2.4.0 no-restype training job. CPU only, a few seconds; it reads config files and the catalog's
Arrow footer, never the data. submit/train_v240_norestype.sh runs it before torchrun; on the login node:

    source submit/env_v240.sh
    "$MW_PY" sh_scripts/preflight_v240.py --phase 1a --gpus 4 -- +model.trunk.restype_msa_only=true

Checks (exit status 1 on any problem):
  * the config composes and validates with the overrides, from THIS checkout's code; prints what the run trains with
  * model.trunk.restype_msa_only is true in the validated config
  * the item catalog matches the config (the build fingerprint in its Arrow footer), so the trainer reads it as is and does not
    start rebuilding 26M rows into a directory you cannot write
  * every LMDB / metadata path is readable and the run directory is writable
  * train_item is a multiple of GPUs x grad_accum x num_batch (the graph trainer asserts it) and the effective batch is 256
  * phase 1b: the phase-1a run exists and has written epoch=0800.pt
  * W&B: your own credentials exist, and nothing points at the repo owner's W&B server or team
"""

from __future__ import annotations

import argparse
import netrc
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/miniworld/phase1a_distogram_medium_v240.yaml"
EFFECTIVE_BATCH = 256
PHASE_1A_EPOCHS = 800
OWNER_WANDB_ENTITY = (
    "team_gm"  # the repo owner's team on forge; your own entity on forge is fine
)


def compose(config: Path, overrides: list[str]):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        return hydra_compose(config_name=config.name, overrides=overrides)


def absolute_paths(value, key=""):
    """(key, path) for every absolute-path string in a dumped config."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from absolute_paths(v, f"{key}.{k}" if key else k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from absolute_paths(v, f"{key}[{i}]")
    elif isinstance(value, str) and value.startswith("/"):
        yield key, value


def path_problem(path: Path) -> str | None:
    if not path.exists():
        return "does not exist"
    if path.is_dir():
        if path.suffix != ".lmdb":
            return (
                None
                if os.access(path, os.R_OK | os.X_OK)
                else "directory is not readable"
            )
        manifest = (
            path / "shards.json"
        )  # a shard collection, read as miniworld.data.io.load._shard_paths does
        if manifest.is_file():
            if not os.access(manifest, os.R_OK):
                return "shards.json is not readable by you (the loader reads it to find the shards): ask the owner to chmod g+r it"
            import json

            shards = [
                path / s["path"] for s in json.loads(manifest.read_text())["shards"]
            ]
            unreadable = [s for s in shards if not os.access(s / "data.mdb", os.R_OK)]
            return (
                f"{len(unreadable)} of {len(shards)} shards have no readable data.mdb"
                if unreadable
                else None
            )
        return (
            None if os.access(path / "data.mdb", os.R_OK) else "data.mdb is not readable"
        )
    return None if os.access(path, os.R_OK) else "not readable"


def writable_ancestor(path: Path) -> bool:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return os.access(probe, os.W_OK | os.X_OK)


def wandb_problems() -> list[str]:
    problems = []
    base = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    entity = os.environ.get("WANDB_ENTITY", "")
    print(f"[preflight] W&B server {base}, entity {entity or '<account default>'}")
    if entity == OWNER_WANDB_ENTITY:
        problems.append(
            f"WANDB_ENTITY={entity} is the repo owner's team: log to your own entity"
        )
    if os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        return problems
    if os.environ.get("WANDB_API_KEY"):
        return problems
    host = urlparse(base).hostname or "api.wandb.ai"
    try:
        found = netrc.netrc().authenticators(host) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        found = False
    if not found:
        problems.append(
            f"no W&B login for {host}: run `WANDB_BASE_URL={base} {sys.prefix}/bin/wandb login` on the login node once "
            "(or export WANDB_API_KEY), or submit with MW_WANDB=off"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("1a", "1b"), default="1a")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument(
        "overrides", nargs="*", help="Hydra overrides, as passed to the trainer"
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO / "scripts"))
    from run_miniworld_distogram_train import Config

    import miniworld
    from miniworld.data.dataloader.dataloader import _catalog_fingerprint

    problems: list[str] = []
    code = Path(miniworld.__file__).resolve()
    if REPO not in code.parents:
        problems.append(
            f"miniworld is imported from {code}, not from this checkout ({REPO}/src): check PYTHONPATH"
        )

    try:
        cfg = Config.model_validate(compose(args.config, list(args.overrides)))
    except Exception as exc:  # noqa: BLE001 - report any composition / validation error as a failed check
        print(f"[preflight] FAIL: the config does not compose / validate: {exc}")
        return 1

    trunk, data, train = cfg.model.trunk, cfg.data, cfg.train
    world_batch = args.gpus * train.grad_accum_steps * train.num_batch
    run_root = Path(train.run_dir).resolve()
    print(
        f"[preflight] phase {args.phase}, config {args.config.name}, code {code.parent}"
    )
    print(
        f"[preflight] model: restype_msa_only={getattr(trunk, 'restype_msa_only', '<missing>')}, "
        f"pairformer {trunk.pairformer.n_block} blocks, MSA module {trunk.msa_module.n_block} blocks, "
        f"checkpoint segments MSA {trunk.msa_module.n_checkpoint_segments} / pairformer {trunk.pairformer.n_checkpoint_segments}"
    )
    print(
        f"[preflight] crop {data.crop.max_tokens} tokens / {data.crop.max_atoms} atoms, epochs to {train.num_epoch}, "
        f"train_item {train.train_item}, {args.gpus} GPUs x grad_accum {train.grad_accum_steps} x {train.num_batch} = {world_batch}, "
        f"seed {train.seed}, lr {train.max_lr}"
    )
    print(
        f"[preflight] sampler antibody_protein {data.sampler.antibody_protein}, protein_rna {data.sampler.protein_rna}; "
        f"sources {dict(data.train_db.source_weights)}"
    )
    print(
        f"[preflight] loader {train.num_workers} workers x prefetch {train.prefetch_factor}; run root {run_root}; "
        f"W&B {'on, project ' + train.wandb_project if train.use_wandb else 'off'}"
    )

    if getattr(trunk, "restype_msa_only", None) is not True:
        problems.append(
            "model.trunk.restype_msa_only is not true (pass +model.trunk.restype_msa_only=true)"
        )
    if train.train_item % world_batch:
        problems.append(
            f"train_item {train.train_item} is not a multiple of GPUs x grad_accum x num_batch = {world_batch}"
        )
    if world_batch != EFFECTIVE_BATCH:
        problems.append(
            f"effective batch is {world_batch}, the v2.4.0 recipe uses {EFFECTIVE_BATCH} (4 GPUs x grad_accum 64)"
        )

    cache = data.train_db.catalog_cache_path
    if cache is None or not Path(cache).is_file():
        problems.append(
            f"catalog {cache} does not exist: the trainer would rebuild it (26M rows) and write it there"
        )
    else:
        import pyarrow as pa

        with pa.memory_map(str(cache), "r") as source:
            meta = pa.ipc.open_file(source).schema.metadata or {}
        expected = _catalog_fingerprint(data.train_db, data.sampler)
        found = (meta.get(b"build_fingerprint") or b"").decode()
        if meta.get(b"weight_kind") != b"raw" or found != expected:
            problems.append(
                f"catalog {cache} was built for another DB/sampler config (fingerprint {found[:12] or '<none>'}, this config "
                f"{expected[:12]}): the trainer would rebuild it and fail to write {Path(cache).parent}"
            )
        else:
            print(
                f"[preflight] catalog {cache}: fingerprint matches ({expected[:12]}), read as is"
            )

    for key, raw_path in absolute_paths(data.train_db.model_dump(mode="json")):
        if key == "catalog_cache_path":
            continue
        problem = path_problem(Path(raw_path))
        if problem:
            problems.append(f"data.train_db.{key} = {raw_path}: {problem}")
    if not writable_ancestor(run_root):
        problems.append(f"run root {run_root} is not writable")

    latest = run_root / "latest_run.txt"
    if args.phase == "1b":
        run = Path(latest.read_text().split()[0]) if latest.is_file() else None
        if run is None:
            problems.append(
                f"phase 1b continues the phase-1a run, but {latest} does not exist"
            )
        elif not (run / "checkpoints" / f"epoch={PHASE_1A_EPOCHS:04d}.pt").is_file():
            problems.append(
                f"phase 1a has not finished: {run}/checkpoints/epoch={PHASE_1A_EPOCHS:04d}.pt does not exist"
            )
    elif latest.is_file():
        run = Path(latest.read_text().split()[0])
        if (run / "checkpoints" / f"epoch={PHASE_1A_EPOCHS:04d}.pt").is_file():
            print(
                f"[preflight] note: phase 1a already reached epoch {PHASE_1A_EPOCHS} in {run}; submit MW_PHASE=1b to continue"
            )
        else:
            print(f"[preflight] resuming {run}")

    if train.use_wandb:
        problems.extend(wandb_problems())

    if problems:
        print("[preflight] FAIL:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("[preflight] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
