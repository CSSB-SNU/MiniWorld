from __future__ import annotations

import logging
import os
import time
from pathlib import Path
import warnings

import click
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf
from pydantic import BaseModel
from team_gm.core.callbacks import Callback
from team_gm.utils.script_utils import MetricsAggregator

import wandb
from miniworld.configs.data_explicit import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TokenizerConfig,
)
from miniworld.configs import EDMDiffuserConfig

from miniworld.data.dataloader.dataloader_explicit import BioMolData
from miniworld.models import ExplicitClient as Client
from miniworld.models.af3_like_explicit import Model
from miniworld.utils import get_step_decay_scheduler_with_warmup

torch.set_float32_matmul_precision("medium")
# anomaly detection
torch.autograd.set_detect_anomaly(False)


class DataConfig(BaseModel):
    """Configuration for data loading."""

    train_db: BioMolDBConfig
    valid_db: BioMolDBConfig
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
    help="checkpoint file",
)
@click.option(
    "--job-name",
    type=str,
    help="Job name",
)
@click.argument("overrides", type=str, nargs=-1)
def train(  # noqa: PLR0912, PLR0915
    config: Path,
    ckpt: Path | None,
    job_name: str | None,
    overrides: tuple[str, ...],
):
    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name, overrides=list(overrides))
    cfg = Config.model_validate(cfg)
    
    if torch.cuda.is_available():
        visible_gpu_count = torch.cuda.device_count()
        world_size_env = os.environ.get("WORLD_SIZE")
        local_world_size_env = os.environ.get("LOCAL_WORLD_SIZE")

        if world_size_env is None and local_world_size_env is None:
            # When the script is launched directly under SLURM without torchrun/srun,
            # all requested GPUs can be visible while Fabric still defaults to a
            # single process. In that case, use all visible GPUs on this node.
            world_size = visible_gpu_count
            local_world_size = visible_gpu_count
        else:
            world_size = int(world_size_env or "0")
            local_world_size = int(local_world_size_env or "0")
            if local_world_size == 0:
                local_world_size = visible_gpu_count
            if world_size == 0:
                world_size = local_world_size

        # Bind the process to its rank-local device before Fabric/DDP touches CUDA.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        fabric = Fabric(
            accelerator="cuda",
            devices=local_world_size,
            num_nodes=max(1, world_size // local_world_size),
            strategy="ddp" if world_size > 1 else "auto",
        )
    else:
        fabric = Fabric(accelerator="cpu", devices=1)

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

    if fabric.is_global_zero:
        formatter = logging.Formatter(
            fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        log_path = run_sub_dir / "train.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        client.logger.addHandler(file_handler)

    if cfg.train.verbose:
        client.add_callback(VerboseCallback())

    if cfg.train.optimizer is None or cfg.train.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            client.model.parameters(),
            cfg.train.max_lr,
        )
    elif cfg.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            client.model.parameters(),
            cfg.train.max_lr,
            betas=(0.9, 0.95),
        )
    else:
        msg = f"Unsupported optimizer: {cfg.train.optimizer}"
        raise ValueError(msg)
    
    if cfg.train.compile:
        torch._dynamo.config.cache_size_limit = 128  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001
        # Compile the raw model before Fabric wraps it so Lightning can reapply
        # compile safely around DDP instead of compiling the Fabric wrapper itself.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.jit\.script_method` is deprecated\..*",
                category=DeprecationWarning,
                module=r"torch\.jit\._script",
            )
            client.model.compile(dynamic=False)
        client.logger.info("Compiled model")

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

    if ckpt:
        state_dict = torch.load(ckpt, map_location="cpu")
        client.load_state_dict(state_dict)

    train_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler,
        tokenizer_config=cfg.data.tokenizer,
    )
    valid_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.valid_db,
        sampler_config=cfg.data.sampler,
        tokenizer_config=cfg.data.tokenizer,
    )

    train_dataloader = BioMolData(train_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=True,
        batch_size=cfg.train.num_batch,
        num_workers=cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
        shuffle=True,
        bucket_msa_multiple=cfg.train.bucket_msa_multiple,
        bucket_token_multiple=cfg.train.bucket_token_multiple,
        bucket_atom_multiple=cfg.train.bucket_atom_multiple,
    )

    valid_dataloader = BioMolData(valid_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=False,
        batch_size=cfg.train.num_batch,  # or 1
        num_workers=0,
    )
    world_size = fabric.world_size
    train_num_item = cfg.train.train_item // world_size
    valid_num_item = cfg.train.valid_item // world_size

    train_aggregator = MetricsAggregator(client, "train", use_wandb=cfg.train.use_wandb)
    valid_aggregator = MetricsAggregator(client, "valid", use_wandb=cfg.train.use_wandb)
    checkpoint_dir = run_sub_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    client.logger.info("Start training")
    while client.epoch < cfg.train.num_epoch:
        client.logger.info("Training Epoch %d", client.epoch)
        train_dataloader.sampler.set_epoch(client.epoch)  # pyright: ignore[reportAttributeAccessIssue]

        for step, result in enumerate(client.training_epoch(train_dataloader)):
            train_aggregator.log_step(result)
            if step >= train_num_item:
                break
        train_aggregator.log_epoch()

        checkpoint_path = checkpoint_dir / "last.pt"
        client.save_checkpoint(checkpoint_path)
        if client.epoch % cfg.train.save_freq == 0:
            checkpoint_path = checkpoint_dir / f"epoch={client.epoch:04d}.pt"
            client.save_checkpoint(checkpoint_path, model_only=True)

        if (client.epoch - 1) % cfg.train.eval_freq == 0:
            valid_dataloader.sampler.set_epoch(client.epoch)  # pyright: ignore[reportAttributeAccessIssue]
            client.logger.info("Validation Epoch %d", client.epoch)
            for n_item, result in enumerate(client.validation_epoch(valid_dataloader)):
                valid_aggregator.log_step(result, ignore_step=True)
                if n_item + 1 >= valid_num_item:
                    client.call_callbacks("on_validation_epoch_end")
                    break

            valid_aggregator.log_epoch()


if __name__ == "__main__":
    # set mp start method
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
