import datetime
import json
import logging
from pathlib import Path

import click
import torch
from lightning import Fabric
from omegaconf import OmegaConf
from pydantic import BaseModel
from team_gm.utils.script_utils import MetricsAggregator

import wandb
from miniworld.data.dataloader.dataloader_edge_backprop import (
    BioMolData,
    BioMolDBConfig,
    CropConfig,
    EdgeWeightConfig,
    MSAConfig,
)
from miniworld.data.to_cif import batch_to_cif
from miniworld.models.af3_natten import AF3NattenClient
from miniworld.utils import get_step_decay_scheduler_with_warmup, set_seed

# torch.set_float32_matmul_precision("high")  # noqa: ERA001
# anomaly detection
torch.autograd.set_detect_anomaly(False)


def setup_logger(client: AF3NattenClient) -> None:
    if not client.is_global_zero:
        return

    client.logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    client.logger.addHandler(handler)

    now = datetime.datetime.now(datetime.timezone.utc)
    file_handler = logging.FileHandler(
        f"logs/af3_natten/af3_natten_{now:%Y%m%d_%H%M%S}.log",
    )
    file_handler.setFormatter(formatter)
    client.logger.addHandler(file_handler)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="config file",
)
@click.option(
    "--resume-from-ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="checkpoint file",
)
@click.option(
    "-w",
    is_flag=True,
    help="Use wandb for logging",
)
@click.option(
    "--ckpt-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default="checkpoints/",
    help="dir for save checkpoint",
)
@click.option(
    "--seed",
    type=int,
    help="random seed",
)
def train(  # noqa: PLR0912, PLR0915
    config: Path | None,
    resume_from_ckpt: Path | None,
    w: bool,
    ckpt_dir: Path,
    seed: int | None,
):
    if config and not resume_from_ckpt:
        cfg = OmegaConf.load(config)
        cfg = AF3NattenClient.Config.model_validate(cfg)
        client = AF3NattenClient(cfg)
    else:
        msg = "You must provide either a config file or a checkpoint file, but not both."
        raise ValueError(msg)

    if cfg.experiment.compile:
        client.model.compile()
        client.logger.info("Compiled model")

    fabric = Fabric()
    fabric.launch()

    optimizer = torch.optim.AdamW(
        client.model.parameters(),
        client.config.experiment.max_lr,
    )
    scheduler = get_step_decay_scheduler_with_warmup(
        optimizer=optimizer,
        warmup_steps=client.config.experiment.warmup_steps,
        decay_steps=client.config.experiment.decay_steps,
        decay_factor=client.config.experiment.decay_factor,
    )

    client.setup(
        fabric=fabric,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation_steps=client.config.experiment.grad_accum_steps,
        gradient_clip_norm=client.config.experiment.grad_clip_max_norm,
    )
    setup_logger(client)

    if resume_from_ckpt is not None:
        state_dict = torch.load(resume_from_ckpt, map_location="cpu")
        client.load_state_dict(state_dict)
        client.logger.info(
            "Load pretrain weight: %s (%d epoch)",
            resume_from_ckpt.name,
            client.epoch,
        )

    if w and client.is_global_zero:
        wandb.init(name=client.config.experiment.comment)
        wandb.config.update(client.config.model_dump())
    msg = f"Config:\n{json.dumps(client.config.model_dump(), indent=4, default=str)}"
    client.logger.info(msg)

    if seed is not None:
        set_seed(seed)
        client.logger.info("Set random seed: %d", seed)

    train_data_config = BioMolData.BioMolConfig(
        crop_config=client.config.data.crop,
        msa_config=client.config.data.msa,
        DB_config=client.config.data.train_db,
        edge_weight_config=client.config.data.edge_weight,
    )
    valid_data_config = BioMolData.BioMolConfig(
        crop_config=client.config.data.crop,
        msa_config=client.config.data.msa,
        DB_config=client.config.data.valid_db,
        edge_weight_config=client.config.data.edge_weight,
    )

    prefetch_factor = (
        None
        if client.config.experiment.prefetch_factor == 0
        else int(client.config.experiment.prefetch_factor)
    )

    train_loader = BioMolData(train_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=True,
        use_adaptive_sampler=True,
        batch_size=client.config.experiment.num_batch,
        num_workers=client.config.experiment.num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,
    )
    valid_loader = BioMolData(valid_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=False,
        use_adaptive_sampler=False,
        batch_size=client.config.experiment.num_batch,  # or 1
        num_workers=0,
    )

    client.logger.info("-" * 70)
    client.logger.info("")
    client.logger.info("Start training".center(70))
    client.logger.info("")
    client.logger.info("-" * 70)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_aggregator = MetricsAggregator(client, "train", use_wandb=w)
    valid_aggregator = MetricsAggregator(client, "valid", use_wandb=w)

    world_size = fabric.world_size
    train_num_item = client.config.experiment.train_item // world_size
    valid_num_item = client.config.experiment.valid_item // world_size
    min_train_loss = float("inf")
    comment = client.config.experiment.comment

    for epoch in range(client.epoch, client.config.experiment.num_epoch):
        client.logger.info("Training Epoch %d", client.epoch)
        train_loader.sampler.set_epoch(epoch)  # pyright: ignore[reportAttributeAccessIssue]
        for n_item, result in enumerate(client.training_epoch(train_loader)):
            train_aggregator.log_step(result)
            if n_item + 1 >= train_num_item:
                client._epoch += 1  # noqa: SLF001
                client.call_callbacks("on_train_epoch_end")
                break

        means = train_aggregator.log_epoch()

        if client.is_global_zero:
            train_loss = means["total_loss"]
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                checkpoint_path = ckpt_dir / f"miniworld_{comment}_best.pt"
                client.save_checkpoint(checkpoint_path)
                client.logger.info(
                    "Save best checkpoint: %s (train loss: %.4g)",
                    checkpoint_path.name,
                    train_loss,
                )

        if (client.epoch - 1) % client.config.experiment.eval_freq == 0:
            valid_loader.sampler.set_epoch(epoch)  # pyright: ignore[reportAttributeAccessIssue]
            client.logger.info("Validation Epoch %d", client.epoch)
            for n_item, result in enumerate(client.validation_epoch(valid_loader)):
                valid_aggregator.log_step(result, ignore_step=True)
                if n_item + 1 >= valid_num_item:
                    client.call_callbacks("on_validation_epoch_end")
                    break

            valid_aggregator.log_epoch()

            checkpoint_path = ckpt_dir / f"miniworld_{comment}_{epoch}.pt"
            client.save_checkpoint(checkpoint_path)


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="config file for validation data",
)
@click.option(
    "--ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="checkpoint file",
)
@click.option(
    "--seed",
    type=int,
    help="random seed",
)
def validate(
    config: Path,
    ckpt: Path | None,
    seed: int | None,
):
    class ValidateDBConfig(BaseModel):
        """Configuration for validation data loading."""

        valid_db: BioMolDBConfig
        crop: CropConfig
        msa: MSAConfig
        edge_weight: EdgeWeightConfig

    class ValidateExperimentConfig(BaseModel):
        """Configuration for validation experiments."""

        num_batch: int = 1
        compile: bool = False
        valid_item: int = 640

        # Validation arguments
        timesteps: int = 100
        num_workers: int = 4
        prefetch_factor: int = 8
        use_ema: bool = True

    class ValidateConfig(BaseModel):
        """Overall configuration for validation."""

        data: ValidateDBConfig
        experiment: ValidateExperimentConfig

    cfg = OmegaConf.load(config)
    cfg = ValidateConfig.model_validate(cfg)
    if not ckpt:
        msg = "You must provide a checkpoint file."
        raise ValueError(msg)
    client = AF3NattenClient.from_checkpoint(ckpt)
    client.model.to("cuda")
    if cfg.experiment.compile:
        client.model.compile()

    setup_logger(client)

    fabric = Fabric()
    fabric.launch()

    world_size = fabric.world_size
    local_rank = fabric.local_rank

    client.logger.info(
        "Load pretrain weight: %s (%d epoch)",
        ckpt.name,
        client.epoch,
    )

    msg = f"Config:\n{json.dumps(client.config.model_dump(), indent=4, default=str)}"
    client.logger.info(msg)

    if seed is not None:
        set_seed(seed)
        client.logger.info("Set random seed: %d", seed)

    valid_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.valid_db,
        edge_weight_config=cfg.data.edge_weight,
    )

    prefetch_factor = (
        None
        if cfg.experiment.prefetch_factor == 0
        else int(cfg.experiment.prefetch_factor)
    )
    valid_loader = BioMolData(valid_data_config).create_ddp_dataloader(
        world_size=world_size,
        rank=local_rank,
        drop_last=False,
        batch_size=cfg.experiment.num_batch,  # or 1
        num_workers=cfg.experiment.num_workers,
        prefetch_factor=prefetch_factor,
    )

    client.logger.info("-" * 70)
    client.logger.info("")
    client.logger.info("Start validation".center(70))
    client.logger.info("")
    client.logger.info("-" * 70)

    valid_num_item = cfg.experiment.valid_item // world_size

    for ii, _batch in enumerate(valid_loader):
        batch = fabric.to_device(_batch)
        click.echo(
            f"residue length : {batch.scheme.residue_idx.shape[1]} \
                   atom length : {batch.structure.atom_pos.shape[1]}",
        )

        batch = batch.duplicate(client.config.experiment.eval_sample_num)
        output = client.inference(
            batch,
            timesteps=client.config.experiment.eval_timesteps,
        )

        result_dict = client.test_inference_quality(
            batch,
            output,
        )
        batch_to_cif(
            batch,
            output.atom_pos_pred,
            Path(f"outputs/miniworld_v1.0.0/{batch.name[0]}_pred.cif"),
        )
        click.echo(
            f"result for {batch.name[0]}: "
            + ", ".join([f"{k}: {v:.4g}" for k, v in result_dict.items()]),
        )
        if ii >= valid_num_item:
            client.call_callbacks("on_validation_epoch_end")
            break


if __name__ == "__main__":
    # set mp start method
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
