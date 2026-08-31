"""Training script for MiniWorld phase4 (confidence head).

Attach a confidence head (pLDDT / PAE / PDE) on top of the FROZEN phase3 structure
model (trunk + EDM diffusion) and train ONLY the confidence head (cross-entropy vs
targets from a predicted structure). The structure model is loaded from a phase3
checkpoint and frozen via the client's ``param_policy``.

The predicted structure is produced by a FULL frozen diffusion rollout inside
``phase4.client.Client.predict_structure`` — the diffusion-step seam (step count /
inline-vs-precomputed cache is an open decision).

Usage:
    torchrun --nproc_per_node=1 scripts/run_miniworld_phase4_train.py train \
        --config configs/miniworld/large_H100_phase4.yaml \
        --ckpt  /path/to/phase3_last.pt \
        --no-ckpt-strict
"""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from itertools import product
from pathlib import Path
from typing import Annotated, Union

import click
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf
from pydantic import BaseModel, Discriminator, Tag
from team_gm.core.callbacks import Callback
from team_gm.utils.script_utils import MetricsAggregator

import wandb
from miniworld.configs import (
    BioMolDBConfig,
    CropConfig,
    EDMDiffuserConfig,
    MSAConfig,
    SamplerConfig,
    TemplateConfig,
    TokenizerConfig,
)
from miniworld.data.dataloader import BioMolDBV2Config
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.features.batch import Batch
from miniworld.models.phase4 import Client, Model
from miniworld.training import trainable_parameters
from miniworld.utils import get_step_decay_scheduler_with_warmup

torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)


_V2_DB_KEYS = frozenset(
    {"pdb", "distillation_sources", "items_path", "resources_path", "source_weights"},
)


def _db_config_variant(value: object) -> str:
    """Discriminate legacy BioMolDBConfig vs multi-source BioMolDBV2Config."""
    if isinstance(value, BioMolDBV2Config):
        return "v2"
    if isinstance(value, BioMolDBConfig):
        return "v1"
    try:
        keys = set(value.keys())  # type: ignore[attr-defined]
    except (TypeError, AttributeError):
        return "v1"
    return "v2" if _V2_DB_KEYS & keys else "v1"


class DataConfig(BaseModel):
    """Configuration for data loading."""

    train_db: Annotated[
        Union[
            Annotated[BioMolDBConfig, Tag("v1")],
            Annotated[BioMolDBV2Config, Tag("v2")],
        ],
        Discriminator(_db_config_variant),
    ]
    crop: CropConfig
    msa: MSAConfig
    tokenizer: TokenizerConfig
    sampler: SamplerConfig


class Config(BaseModel):
    """Overall configuration."""

    data: DataConfig
    train: Client.TrainConfig
    model: Model.Config
    diffuser: EDMDiffuserConfig
    loss: Client.LossConfig


def _fabric_from_torchrun() -> Fabric:
    """Create Fabric with node/device counts inherited from torchrun."""
    world_size = os.environ.get("WORLD_SIZE")
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if world_size is None or local_world_size is None:
        return Fabric()

    devices = int(local_world_size)
    num_nodes = int(world_size) // devices
    return Fabric(devices=devices, num_nodes=num_nodes)


def _ceil_to_multiple(value: int, multiple: int | None) -> int:
    if multiple is None or multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _bucket_values(max_value: int, multiple: int | None) -> list[int]:
    if multiple is None or multiple <= 0:
        return [max_value]
    bucket_max = _ceil_to_multiple(max_value, multiple)
    return list(range(multiple, bucket_max + 1, multiple))


def _find_recycle_model(module: torch.nn.Module) -> Model:
    """Unwrap Fabric/compile wrappers to reach the raw phase3 model."""
    current: object = module
    visited: set[int] = set()

    while isinstance(current, torch.nn.Module):
        visited.add(id(current))
        if isinstance(current, Model):
            return current
        for attr in ("module", "_forward_module", "_orig_mod", "model"):
            child = getattr(current, attr, None)
            if isinstance(child, torch.nn.Module) and id(child) not in visited:
                current = child
                break
        else:
            break

    msg = f"Could not unwrap raw phase3 model from {type(module).__name__}."
    raise RuntimeError(msg)


def _capture_rng_state(model: Model) -> dict[str, object]:
    state: dict[str, object] = {
        "torch": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "model_rng": copy.deepcopy(model.rng.bit_generator.state),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(model: Model, state: dict[str, object]) -> None:
    torch.random.set_rng_state(state["torch"])  # pyright: ignore[reportArgumentType]
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])  # pyright: ignore[reportArgumentType]
    np.random.set_state(state["numpy"])  # pyright: ignore[reportArgumentType]
    random.setstate(state["python"])  # pyright: ignore[reportArgumentType]
    model.rng.bit_generator.state = copy.deepcopy(state["model_rng"])


def _build_precompile_batch(
    *,
    device: torch.device,
    msa_depth: int,
    n_tokens: int,
    n_atoms: int,
    n_templates: int,
    num_res_class: int,
) -> Batch:
    """Create a dense synthetic batch that exercises the normal train path."""
    batch = Batch.empty(
        n_temp=n_templates,
        msa_depth=msa_depth,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
    ).to(device=device)

    token_idx = torch.arange(n_tokens, device=device, dtype=torch.long)
    atom_idx = torch.arange(n_atoms, device=device, dtype=torch.long)
    atom_to_token = torch.div(
        atom_idx * n_tokens,
        n_atoms,
        rounding_mode="floor",
    ).clamp(max=n_tokens - 1)
    seq_token_type = token_idx.remainder(num_res_class)

    atom_pos = torch.stack(
        [
            atom_to_token.float() * 3.0,
            atom_idx.remainder(17).float() * 0.35,
            atom_idx.div(17, rounding_mode="floor").float() * 0.15,
        ],
        dim=-1,
    ).to(torch.float32)
    token_pos = torch.stack(
        [
            token_idx.float() * 3.0,
            token_idx.remainder(11).float() * 0.4,
            token_idx.div(11, rounding_mode="floor").float() * 0.2,
        ],
        dim=-1,
    ).to(torch.float32)

    batch.name = [f"precompile_m{msa_depth}_t{n_tokens}_a{n_atoms}"]
    batch.sequence.token_type.copy_(seq_token_type.unsqueeze(0))

    batch.structure.atom_pos.copy_(atom_pos.unsqueeze(0))
    batch.structure.atom_pos_mask.fill_(True)  # noqa: FBT003
    batch.structure.atom_mask.fill_(True)  # noqa: FBT003
    batch.structure.token_mask.fill_(True)  # noqa: FBT003
    if n_tokens > 1:
        batch.structure.token_bond = torch.stack(
            [
                torch.arange(n_tokens - 1, device=device, dtype=torch.long),
                torch.arange(1, n_tokens, device=device, dtype=torch.long),
            ],
            dim=-1,
        ).unsqueeze(0)

    batch.reference.pos.copy_(atom_pos.unsqueeze(0))
    batch.reference.mask.fill_(1.0)
    batch.reference.element.copy_(
        (6 + atom_idx.remainder(3)).to(torch.float32).unsqueeze(0),
    )
    batch.reference.charge.copy_(
        ((atom_idx.remainder(5).float() - 2.0) * 0.1).unsqueeze(0),
    )
    batch.reference.space_uid.zero_()

    batch.scheme.token_residue_idx.copy_(token_idx.unsqueeze(0))
    batch.scheme.token_idx.copy_(token_idx.unsqueeze(0))
    batch.scheme.token_asym_id.zero_()
    batch.scheme.token_entity_id.zero_()
    batch.scheme.token_sym_id.zero_()
    batch.scheme.atom_to_token_idx_map.copy_(atom_to_token.unsqueeze(0))
    batch.scheme.atom_to_chain_id.zero_()

    msa_row_offset = torch.arange(msa_depth, device=device, dtype=torch.long).unsqueeze(1)
    msa_sequences = (seq_token_type.unsqueeze(0) + msa_row_offset).remainder(20)
    batch.msa.aligned_sequences.copy_(msa_sequences.unsqueeze(0))
    batch.msa.mask = torch.ones_like(batch.msa.mask, dtype=torch.float32)
    batch.msa.has_deletion = torch.zeros_like(batch.msa.has_deletion, dtype=torch.int32)
    msa_deletion = torch.linspace(
        0.0,
        1.0,
        msa_depth,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1).expand(-1, n_tokens)
    batch.msa.deletion_value = msa_deletion.unsqueeze(0).to(torch.float64)
    batch.msa.profile.copy_(
        torch.nn.functional.one_hot(
            seq_token_type,
            num_classes=num_res_class,
        ).to(torch.float32).unsqueeze(0),
    )
    batch.msa.deletion_mean.copy_(msa_deletion.mean(dim=0, keepdim=True))

    template_offsets = torch.linspace(
        0.0,
        0.3,
        n_templates,
        device=device,
        dtype=torch.float32,
    )
    cb_xyz = token_pos.unsqueeze(0).repeat(n_templates, 1, 1)
    cb_xyz[:, :, 1] += template_offsets[:, None]
    bb_offsets = torch.tensor(
        [
            [-0.55, 0.10, 0.00],
            [0.00, 0.00, 0.00],
            [0.65, -0.10, 0.00],
        ],
        device=device,
        dtype=torch.float32,
    )
    bb_xyz = token_pos.unsqueeze(0).unsqueeze(-2) + bb_offsets.view(1, 1, 3, 3)
    bb_xyz = bb_xyz.repeat(n_templates, 1, 1, 1)
    bb_xyz[:, :, :, 1] += template_offsets[:, None, None]

    batch.template.mask.fill_(True)  # noqa: FBT003
    batch.template.ids.zero_()
    batch.template.res_type.copy_(
        seq_token_type.unsqueeze(0).repeat(n_templates, 1).unsqueeze(0),
    )
    batch.template.cb_xyz.copy_(cb_xyz.unsqueeze(0))
    batch.template.cb_mask.fill_(True)  # noqa: FBT003
    batch.template.bb_xyz.copy_(bb_xyz.unsqueeze(0))
    batch.template.bb_mask.fill_(True)  # noqa: FBT003

    batch.chain.entity_type.fill_(1)
    return batch


def _warmup_bucket_shapes(client: Client, cfg: Config) -> None:
    """Warm up all bucket shapes before training (single-recycle capture)."""
    raw_model = _find_recycle_model(client.model)
    rng_state = _capture_rng_state(raw_model)
    was_training = client.model.training
    param_snapshot = {
        name: p.detach().to("cpu", copy=True)
        for name, p in raw_model.named_parameters()
    }

    def _to_cpu_copy(obj):  # noqa: ANN001, ANN202
        if isinstance(obj, torch.Tensor):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: _to_cpu_copy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_cpu_copy(v) for v in obj)
        return copy.deepcopy(obj)

    optimizer_state_snapshot = _to_cpu_copy(client.optimizer.state_dict())

    msa_buckets = _bucket_values(
        cfg.data.msa.max_msa_depth,
        cfg.train.bucket_msa_multiple,
    )
    token_buckets = _bucket_values(
        cfg.data.crop.max_tokens,
        cfg.train.bucket_token_multiple,
    )
    atom_buckets = _bucket_values(
        cfg.data.crop.max_atoms,
        cfg.train.bucket_atom_multiple,
    )
    n_templates = TemplateConfig().n_templates
    warmup_n_recycle = 2
    total_bucket_shapes = len(msa_buckets) * len(token_buckets) * len(atom_buckets)
    total_variants = total_bucket_shapes

    client.fabric.barrier()
    if client.device.type == "cuda":
        torch.cuda.synchronize(client.device)
    start_time = time.perf_counter()
    if client.is_global_zero:
        client.logger.info(
            "Starting synthetic bucket warmup: %d shapes x n_recycle=%d = %d passes",
            total_bucket_shapes,
            warmup_n_recycle,
            total_variants,
        )

    try:
        client.model.train()
        client.optimizer.zero_grad(set_to_none=True)

        warmup_idx = 0
        for msa_depth, n_tokens, n_atoms in product(
            reversed(msa_buckets),
            reversed(token_buckets),
            reversed(atom_buckets),
        ):
            batch = _build_precompile_batch(
                device=client.device,
                msa_depth=msa_depth,
                n_tokens=n_tokens,
                n_atoms=n_atoms,
                n_templates=n_templates,
                num_res_class=cfg.model.shared.num_res_class,
            )
            raw_model._forced_n_recycle = warmup_n_recycle  # noqa: SLF001
            with client.fabric.no_backward_sync(
                client.model,  # pyright: ignore[reportArgumentType]
                enabled=False,
            ):
                client.training_step(batch)
            if warmup_idx == 0:
                client.optimizer.step()
            client.optimizer.zero_grad(set_to_none=True)

            warmup_idx += 1
            if client.is_global_zero and (
                warmup_idx == 1 or warmup_idx == total_variants or warmup_idx % 4 == 0
            ):
                client.logger.info(
                    "Bucket warmup %d/%d: msa=%d tokens=%d atoms=%d recycle=%d",
                    warmup_idx,
                    total_variants,
                    msa_depth,
                    n_tokens,
                    n_atoms,
                    warmup_n_recycle,
                )
    finally:
        raw_model._forced_n_recycle = None  # noqa: SLF001
        client.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, p in raw_model.named_parameters():
                p.copy_(param_snapshot[name].to(p.device, non_blocking=True))
        client.optimizer.load_state_dict(optimizer_state_snapshot)
        _restore_rng_state(raw_model, rng_state)
        client.model.train(was_training)

    client.fabric.barrier()
    if client.device.type == "cuda":
        torch.cuda.synchronize(client.device)
    if client.is_global_zero:
        client.logger.info(
            "Finished synthetic bucket warmup in %.1fs",
            time.perf_counter() - start_time,
        )


class VerboseCallback(Callback):
    """Log batch shape and memory usage per batch."""

    def on_train_batch_start(self, client, batch, batch_idx):  # noqa: ANN001
        client.logger.info(
            "rank=%d batch=%d %s | n_tokens=%d n_atoms=%d | mem=%.2fGB",
            client.fabric.global_rank,
            batch_idx,
            str(batch.name[0]),
            batch.token_length,
            batch.atom_length,
            torch.cuda.max_memory_allocated() / 1024**3,
        )


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="config file",
)
@click.option(
    "--ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="phase2 trunk checkpoint (epoch=0900.pt) to load + freeze",
)
@click.option("--job-name", type=str, help="Job name")
@click.option(
    "--ckpt-strict/--no-ckpt-strict",
    default=False,
    show_default=True,
    help="Exact model/optimizer match. Phase3 loads only trunk keys -> use --no-ckpt-strict.",
)
@click.argument("overrides", type=str, nargs=-1)
def train(  # noqa: PLR0912, PLR0915
    config: Path,
    ckpt: Path | None,
    job_name: str | None,
    ckpt_strict: bool,
    overrides: tuple[str, ...],
):
    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name, overrides=list(overrides))
    cfg = Config.model_validate(cfg)

    fabric = _fabric_from_torchrun()
    fabric.launch()
    if cfg.train.seed is not None:
        fabric.seed_everything(cfg.train.seed)

    date_dir = Path(cfg.train.run_dir) / time.strftime("%Y-%m-%d")
    run_name = time.strftime("%H%M%S")
    if not job_name:
        job_name = cfg.train.comment
    run_name += f"_{job_name}"
    run_sub_dir = date_dir / run_name
    run_sub_dir.mkdir(parents=True, exist_ok=True)

    client = Client(
        Client.Config(
            train=cfg.train,
            model=cfg.model,
            diffuser=cfg.diffuser,
            loss=cfg.loss,
        ),
    )

    formatter = logging.Formatter(
        fmt=(
            f"[%(asctime)s][rank={fabric.global_rank}]"
            "[%(name)s][%(levelname)s] %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_path = run_sub_dir / "train.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    client.logger.addHandler(file_handler)

    if cfg.train.verbose:
        client.add_callback(VerboseCallback())

    if cfg.train.compile:
        torch._dynamo.config.cache_size_limit = 128  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001
        trunk_compile_mode = cfg.train.trunk_compile_mode
        if trunk_compile_mode:
            # EXPERIMENTAL: cudagraph the FROZEN trunk conditioning path with the
            # requested inductor mode (reduce-overhead / cudagraph-trees) and
            # compile only the trainable diffusion module normally. The trunk
            # outputs are cloned in Phase3Model.forward before the grad path, so a
            # later cudagraph replay never overwrites a tensor the grad path reads.
            client.model.diffusion_module.compile(dynamic=False)
            client.model.enable_trunk_cudagraph(trunk_compile_mode)
            client.logger.info(
                "Compiled model (frozen-trunk cudagraph mode=%s, "
                "diffusion_module dynamic=False)",
                trunk_compile_mode,
            )
        else:
            client.model.compile(dynamic=False)
            client.logger.info("Compiled model")

    config_dict = cfg.model_dump(mode="json")
    msg = f"config:\n{OmegaConf.to_yaml(OmegaConf.create(config_dict))}"
    client.logger.debug(msg)
    if fabric.is_global_zero:
        OmegaConf.save(OmegaConf.create(config_dict), run_sub_dir / "config.yaml")
        if cfg.train.use_wandb:
            # Phase3 is a NEW run: keep a stable id in THIS run_dir so preempt/
            # resume appends to the same phase3 run (never reuse a phase2 id).
            wandb_id_file = Path(cfg.train.run_dir) / "wandb_run_id.txt"
            if wandb_id_file.exists():
                wandb_id = wandb_id_file.read_text().strip()
            else:
                wandb_id = wandb.util.generate_id()
                wandb_id_file.parent.mkdir(parents=True, exist_ok=True)
                wandb_id_file.write_text(wandb_id)
            wandb.init(
                project=cfg.train.wandb_project,
                name=job_name,
                id=wandb_id,
                resume="allow",
                config=config_dict,
            )

    # Load the checkpoint BEFORE building the optimizer so param_policy can
    # load + freeze the trunk and the optimizer only ever sees the trainable
    # subset (to_token_single_trunk + diffusion_module).
    state_dict = torch.load(ckpt, map_location="cpu") if ckpt else None
    policy_summary = client.maybe_apply_param_policy(state_dict)

    optim_params = (
        trainable_parameters(client.model)
        if policy_summary is not None
        else client.model.parameters()
    )

    if cfg.train.optimizer is None or cfg.train.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(optim_params, cfg.train.max_lr)
    elif cfg.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(optim_params, cfg.train.max_lr, betas=(0.9, 0.95))
    else:
        msg = f"Unsupported optimizer: {cfg.train.optimizer}"
        raise ValueError(msg)
    scheduler = get_step_decay_scheduler_with_warmup(
        optimizer=optimizer,
        warmup_steps=cfg.train.warmup_steps,
        decay_steps=cfg.train.decay_steps,
        decay_factor=cfg.train.decay_factor,
    )

    client.setup(
        fabric=fabric,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        gradient_clip_norm=cfg.train.grad_clip_max_norm,
    )

    if state_dict is not None:
        if policy_summary is None:
            # policy disabled -> standard full load (model + optimizer + scheduler + epoch).
            client.load_state_dict(state_dict, strict=ckpt_strict)
        else:
            # policy ON: maybe_apply_param_policy already restored model weights + epoch/step.
            # ALSO resume the trainable (confidence-head) optimizer + LR scheduler so Adam
            # momentum and the warmup/decay schedule CONTINUE across requeue instead of
            # resetting every restart. Skipped on the phase3 SEED, whose optimizer is over
            # the diffusion params and will not match the confidence-head optimizer.
            opt_sd = state_dict.get("optimizer_state_dict")
            sch_sd = state_dict.get("scheduler_state_dict")
            if opt_sd is not None:
                try:
                    client.optimizer.load_state_dict(opt_sd)
                    if sch_sd is not None and client.scheduler is not None:
                        client.scheduler.load_state_dict(sch_sd)
                    client.logger.info(
                        "[resume] restored confidence optimizer + LR scheduler (continuing schedule)",
                    )
                except (ValueError, KeyError, RuntimeError) as e:
                    client.logger.info(
                        "[resume] seed ckpt (phase3 optimizer) -> fresh confidence optimizer/scheduler (%s)",
                        type(e).__name__,
                    )

    # Memory-PROFILE mode (guarded, off by default). MW_MEM_PROFILE=1 hooks the diffusion
    # sub-modules (+ trunk) and logs each one's retained-activation delta AS IT RUNS, so the
    # per-module 768 x num_augment footprint is visible even when a later module OOMs.
    if os.getenv("MW_MEM_PROFILE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        _raw = _find_recycle_model(client.model)
        _GB = 1024 ** 3

        def _mk_memhook(nm, mod):  # noqa: ANN001, ANN202
            def _pre(m, inp):  # noqa: ANN001, ANN202
                m._memprof_before = torch.cuda.memory_allocated()

            def _post(m, inp, out):  # noqa: ANN001, ANN202
                after = torch.cuda.memory_allocated()
                client.logger.info(
                    "[memprof] %-30s delta=%+.2fGB exit_alloc=%.2fGB peak=%.2fGB",
                    nm,
                    (after - getattr(m, "_memprof_before", after)) / _GB,
                    after / _GB,
                    torch.cuda.max_memory_allocated() / _GB,
                )

            mod.register_forward_pre_hook(_pre)
            mod.register_forward_hook(_post)

        for _nm, _mod in _raw.diffusion_module.named_children():
            _mk_memhook("diffusion." + _nm, _mod)
        for _nm in ("input_feature_embedder", "msa_module", "pairformer_blocks", "temp_embedder"):
            _m = getattr(_raw, _nm, None)
            if _m is not None:
                _mk_memhook("trunk." + _nm, _m)
        client.logger.info("[memprof] hooks installed (diffusion children + trunk)")

    # Autotune cache-BUILD mode (guarded, off by default). MW_CAPTURE_CACHE=1 (+ set
    # MINIWORLD_RUN_AUTOTUNE=1 to unlock the full grid) installs the capture hook so the warmup
    # fwd/bwd records the winning triton config per (op,dtype,bucket) — with the fork+SIGKILL
    # compile-timeout guard so register-spill "monster" configs (5-20 min ptxas at 768) get
    # killed at 60s instead of stalling the run. Then flush the cache and exit WITHOUT training.
    _capture_cache = os.getenv("MW_CAPTURE_CACHE", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    # MW_COMPILE_TIMEOUT=1 (without capture-build): install ONLY the fork+SIGKILL compile
    # guard so 768 warmup autotune bounds each monster config to 60s and TRAINING CONTINUES
    # (no flush/exit). Lets a run get past the uncached-768 monster-compile stall without a
    # prebuilt cache. (capture.install patches _bench too — harmless overhead, never flushed.)
    _compile_timeout = os.getenv("MW_COMPILE_TIMEOUT", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if _capture_cache or _compile_timeout:
        from miniworld_engine.autotune import capture

        capture.install()
        client.logger.info(
            "[capture] compile-timeout guard installed (%s)",
            "build mode" if _capture_cache else "timeout-only",
        )

    # Phase4 warmup: the synthetic-bucket warmup calls client.training_step, which for
    # phase4 runs a FULL frozen diffusion rollout (predict_structure) — far too heavy to
    # do per bucket shape, and its cost/shape depends on the still-open diffusion-step
    # decision. Skip by default; the 768 trunk/diffusion autotune then happens on the
    # first real training step (bounded by MW_COMPILE_TIMEOUT). Set MW_WARMUP=1 to force it.
    if os.getenv("MW_WARMUP", "0").strip().lower() in {"1", "true", "yes", "on"}:
        _warmup_bucket_shapes(client, cfg)
    else:
        client.logger.info("[warmup] skipped (phase4 default; set MW_WARMUP=1 to enable)")

    if _capture_cache:
        from miniworld_engine.autotune import capture

        capture.flush(top_k=5)
        client.logger.info("[capture] flushed autotune cache; exiting (cache-build mode)")
        return

    world_size = fabric.world_size
    train_num_item = cfg.train.train_item // world_size

    train_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler,
        tokenizer_config=cfg.data.tokenizer,
    )
    train_dataset = BioMolData(train_data_config)
    train_dataloader = train_dataset.create_ddp_dataloader(
        world_size=world_size,
        rank=fabric.global_rank,
        seed=cfg.train.seed,
        drop_last=True,
        batch_size=cfg.train.num_batch,
        num_workers=cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
        num_samples_per_rank=train_num_item,
        persistent_workers=cfg.train.num_workers > 0,
        shuffle=True,
        bucket_msa_multiple=cfg.train.bucket_msa_multiple,
        bucket_token_multiple=cfg.train.bucket_token_multiple,
        bucket_atom_multiple=cfg.train.bucket_atom_multiple,
        bucket_template_multiple=TemplateConfig().n_templates,
    )

    train_aggregator = MetricsAggregator(client, "train", use_wandb=cfg.train.use_wandb)
    checkpoint_dir = run_sub_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    client.logger.info("Start training")
    while client.epoch < cfg.train.num_epoch:
        client.logger.info("Training Epoch %d", client.epoch)
        train_dataloader.sampler.set_epoch(client.epoch)  # pyright: ignore[reportAttributeAccessIssue]
        train_dataset.set_epoch(client.epoch)

        for step, result in enumerate(client.training_epoch(train_dataloader)):
            train_aggregator.log_step(result)
            if step == train_num_item - 1:
                break
        train_aggregator.log_epoch()

        checkpoint_path = checkpoint_dir / "last.pt"
        client.save_checkpoint(checkpoint_path)
        if client.epoch % cfg.train.save_freq == 0:
            checkpoint_path = checkpoint_dir / f"epoch={client.epoch:04d}.pt"
            client.save_checkpoint(checkpoint_path)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
