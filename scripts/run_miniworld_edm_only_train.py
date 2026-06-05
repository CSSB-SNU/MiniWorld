"""Training script for MiniWorld (plain AF3-like EDM diffusion).

Same harness as ``run_miniworld_only_train.py`` (param-policy, bucket warmup),
but drives the ``miniworld_edm`` client + ``EuclideanDiffuser`` / ``AF3Solver``
instead of the decoupled VE x-prediction diffuser.

Usage:
    torchrun --nproc_per_node=1 scripts/run_miniworld_edm_only_train.py train \
        --config configs/miniworld/config_edm.yaml \
        data=overfitting train=overfitting model=small diffuser=edm
"""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from itertools import product
from pathlib import Path

import click
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf
from pydantic import BaseModel
from team_gm.core.callbacks import Callback
from team_gm.utils.script_utils import MetricsAggregator

import wandb
from miniworld.configs import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TemplateConfig,
    TokenizerConfig,
    EDMDiffuserConfig,
)
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.features.batch import Batch
from miniworld.models.miniworld_edm import Client, Model
from miniworld.training import trainable_parameters
from miniworld.utils import get_step_decay_scheduler_with_warmup

torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)


class DataConfig(BaseModel):
    """Configuration for data loading."""

    train_db: BioMolDBConfig
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
    """Round a value up to the nearest bucket multiple."""
    if multiple is None or multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _bucket_values(max_value: int, multiple: int | None) -> list[int]:
    """Enumerate every padded bucket size used by training."""
    if multiple is None or multiple <= 0:
        return [max_value]
    bucket_max = _ceil_to_multiple(max_value, multiple)
    return list(range(multiple, bucket_max + 1, multiple))


def _find_recycle_model(module: torch.nn.Module) -> Model:
    """Unwrap Fabric/compile wrappers to reach the raw MiniWorld model."""
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

    msg = f"Could not unwrap raw MiniWorld model from {type(module).__name__}."
    raise RuntimeError(msg)


def _capture_rng_state(model: Model) -> dict[str, object]:
    """Capture RNG state so warmup does not perturb the real training run."""
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
    """Restore RNG state after the synthetic warmup pass."""
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
    batch.structure.atom_pos_mask.fill_(True)
    batch.structure.atom_mask.fill_(True)
    batch.structure.token_mask.fill_(True)
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
    batch.msa.mask.fill_(True)
    batch.msa.has_deletion.zero_()
    msa_deletion = torch.linspace(
        0.0,
        1.0,
        msa_depth,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1).expand(-1, n_tokens)
    batch.msa.deletion_value.copy_(msa_deletion.unsqueeze(0))
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

    batch.template.mask.fill_(True)
    batch.template.ids.zero_()
    batch.template.res_type.copy_(
        seq_token_type.unsqueeze(0).repeat(n_templates, 1).unsqueeze(0),
    )
    batch.template.cb_xyz.copy_(cb_xyz.unsqueeze(0))
    batch.template.cb_mask.fill_(True)
    batch.template.bb_xyz.copy_(bb_xyz.unsqueeze(0))
    batch.template.bb_mask.fill_(True)

    batch.chain.entity_type.fill_(1)
    return batch


def _warmup_bucket_shapes(client: Client, cfg: Config) -> None:
    """Exhaustively warm up all bucket/recycle combinations before training."""
    raw_model = _find_recycle_model(client.model)
    rng_state = _capture_rng_state(raw_model)
    was_training = client.model.training
    param_snapshot = {
        name: p.detach().to("cpu", copy=True)
        for name, p in raw_model.named_parameters()
    }

    def _to_cpu_copy(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: _to_cpu_copy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_cpu_copy(v) for v in obj)
        return copy.deepcopy(obj)

    # Snapshot so the warmup's synthetic optimizer.step() doesn't pollute resumed Adam moments.
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
            (
                "Starting synthetic bucket warmup: %d shapes x n_recycle=%d "
                "= %d forward/backward passes"
            ),
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
            raw_model._forced_n_recycle = warmup_n_recycle
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
                warmup_idx == 1
                or warmup_idx == total_variants
                or warmup_idx % 4 == 0
            ):
                client.logger.info(
                    (
                        "Bucket warmup %d/%d: msa=%d tokens=%d atoms=%d "
                        "recycle=%d"
                    ),
                    warmup_idx,
                    total_variants,
                    msa_depth,
                    n_tokens,
                    n_atoms,
                    warmup_n_recycle,
                )
    finally:
        raw_model._forced_n_recycle = None
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
            (
                "rank=%d batch=%d %s | n_tokens=%d n_atoms=%d "
                "n_msa_valid=%d n_msa_bucket=%d "
                "n_template_valid=%d n_template_bucket=%d | mem=%.2fGB"
            ),
            client.fabric.global_rank,
            batch_idx,
            str(batch.name[0]),
            batch.token_length,
            batch.atom_length,
            batch.msa_count,
            batch.msa_depth,
            batch.template_count,
            batch.template_number,
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
    help="checkpoint file",
)
@click.option(
    "--job-name",
    type=str,
    help="Job name",
)
@click.option(
    "--ckpt-strict/--no-ckpt-strict",
    default=True,
    show_default=True,
    help="Whether checkpoint loading requires an exact model/optimizer match.",
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
        client.model.compile(dynamic=False)
        client.logger.info("Compiled model")

    config_dict = cfg.model_dump(mode="json")
    msg = f"config:\n{OmegaConf.to_yaml(OmegaConf.create(config_dict))}"
    client.logger.debug(msg)
    if fabric.is_global_zero:
        OmegaConf.save(OmegaConf.create(config_dict), run_sub_dir / "config.yaml")
        if cfg.train.use_wandb:
            wandb.init(
                project=cfg.train.wandb_project,
                name=job_name,
                config=config_dict,
            )

    # Load the checkpoint (if any) BEFORE building the optimizer so that
    # param_policy can selectively reinit / freeze layers and the optimizer
    # only ever sees the trainable subset. When the policy is disabled this
    # collapses to the original flow (load_state_dict after setup()).
    state_dict = torch.load(ckpt, map_location="cpu") if ckpt else None
    policy_summary = client.maybe_apply_param_policy(state_dict)

    optim_params = (
        trainable_parameters(client.model)
        if policy_summary is not None
        else client.model.parameters()
    )

    if cfg.train.optimizer is None or cfg.train.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            optim_params,
            cfg.train.max_lr,
        )
    elif cfg.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            optim_params,
            cfg.train.max_lr,
            betas=(0.9, 0.95),
        )
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

    if state_dict is not None and policy_summary is None:
        client.load_state_dict(state_dict, strict=ckpt_strict)

    _warmup_bucket_shapes(client, cfg)

    train_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler,
        tokenizer_config=cfg.data.tokenizer,
    )
    train_dataset = BioMolData(train_data_config)
    train_dataloader = train_dataset.create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.global_rank,
        seed=cfg.train.seed,
        drop_last=True,
        batch_size=cfg.train.num_batch,
        num_workers=cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
        shuffle=True,
        bucket_msa_multiple=cfg.train.bucket_msa_multiple,
        bucket_token_multiple=cfg.train.bucket_token_multiple,
        bucket_atom_multiple=cfg.train.bucket_atom_multiple,
    )
    world_size = fabric.world_size
    train_num_item = cfg.train.train_item // world_size

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
    # set mp start method
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
