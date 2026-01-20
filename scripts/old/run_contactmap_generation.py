import datetime
import json
import logging
from pathlib import Path

import click
import torch
from lightning import Fabric
from omegaconf import OmegaConf
from team_gm.utils.script_utils import MetricsAggregator, set_seed

import wandb
from miniworld.data.dataloader.dataloader_multistate import (
    BioMolMonomerData,
)
from miniworld.models.contact_map_genreation import ContactMapGenerationClient

torch.autograd.set_detect_anomaly(False)


def setup_logger(client: ContactMapGenerationClient) -> None:
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
        f"logs/contactmap_generation/contactmap_generation_{now:%Y%m%d_%H%M%S}.log",
    )
    file_handler.setFormatter(formatter)
    client.logger.addHandler(file_handler)


def get_step_decay_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 1e3,
    decay_steps: int = 5e4,
    decay_factor: float = 0.95,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Step decay scheduler with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / float(warmup_steps)
        num_decays = (step - warmup_steps) // decay_steps
        return decay_factor**num_decays

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
        cfg = ContactMapGenerationClient.Config.model_validate(cfg)
        client = ContactMapGenerationClient(cfg)
    elif not config and resume_from_ckpt:
        client = ContactMapGenerationClient.from_checkpoint(resume_from_ckpt)
    else:
        msg = (
            "You must provide either a config file or a checkpoint file, but not both."
        )
        raise ValueError(msg)

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
        client.load_optimizer_state(resume_from_ckpt)
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

    train_data_config = BioMolMonomerData.BioMolConfig(
        crop_config=client.config.data.crop.model_dump(),
        msa_config=client.config.data.msa.model_dump(),
        kmer_fast_align_config=client.config.data.kmer_fast_align.model_dump(),
        multistate_config=client.config.data.multistate.model_dump(),
        preprocess_config=client.config.data.train_preprocessing.model_dump(),
    )
    valid_data_config = BioMolMonomerData.BioMolConfig(
        crop_config=client.config.data.crop.model_dump(),
        msa_config=client.config.data.msa.model_dump(),
        kmer_fast_align_config=client.config.data.kmer_fast_align.model_dump(),
        multistate_config=client.config.data.multistate.model_dump(),
        preprocess_config=client.config.data.valid_preprocessing.model_dump(),
    )

    prefetch_factor = (
        None
        if client.config.experiment.prefetch_factor == 0
        else int(client.config.experiment.prefetch_factor)
    )
    train_loader = BioMolMonomerData(train_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=True,
        batch_size=client.config.experiment.num_batch,
        num_workers=client.config.experiment.num_workers,
        prefetch_factor=prefetch_factor,
    )
    valid_loader = BioMolMonomerData(valid_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=False,
        batch_size=client.config.experiment.num_batch,
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

    for epoch in range(client.epoch, client.config.experiment.num_epoch):
        client.logger.info("Training Epoch %d", client.epoch)
        train_loader.sampler.set_epoch(epoch)
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
                checkpoint_path = ckpt_dir / "contactmap_generation_best.pt"
                # client.save_checkpoint(checkpoint_path)
                client.logger.info(
                    "Save best checkpoint: %s (train loss: %.4g)",
                    checkpoint_path.name,
                    train_loss,
                )

        if (client.epoch - 1) % client.config.experiment.eval_freq == 0:
            valid_loader.sampler.set_epoch(epoch)
            client.logger.info("Validation Epoch %d", client.epoch)
            for n_item, result in enumerate(client.validation_epoch(valid_loader)):
                valid_aggregator.log_step(result, ignore_step=True)
                if n_item + 1 >= valid_num_item:
                    client.call_callbacks("on_validation_epoch_end")
                    break
            valid_aggregator.log_epoch()
            checkpoint_path = ckpt_dir / f"contactmap_generation_{epoch}.pt"
            # client.save_checkpoint(checkpoint_path)


@cli.command()
@click.option(
    "--ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="checkpoint file",
)
@click.option(
    "-w",
    is_flag=True,
    help="Use wandb for logging",
)
@click.option(
    "--seed",
    type=int,
    help="random seed",
)
@click.option(
    "--save-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="directory to save generated contact maps",
)
@torch.inference_mode()
def sample(
    ckpt: Path | None,
    w: bool,
    seed: int | None,
    save_dir: Path | None,
):
    if not ckpt:
        msg = "You must provide a checkpoint file."
        raise ValueError(msg)
    client = ContactMapGenerationClient.from_checkpoint(ckpt)

    fabric = Fabric()
    fabric.launch()

    client.setup(
        fabric=fabric,
        optimizer=torch.optim.AdamW(
            client.model.parameters(),
            client.config.experiment.max_lr,
        ),
        gradient_accumulation_steps=client.config.experiment.grad_accum_steps,
        gradient_clip_norm=client.config.experiment.grad_clip_max_norm,
    )
    setup_logger(client)

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

    preprocess_config = client.config.data.valid_preprocessing.model_dump()
    msa_config = client.config.data.msa.model_dump()
    preprocess_config["a3m_db_path"] = (
        "/home/psk6950/data/BioMolDBv2_2024Oct21/slim_a3m.lmdb"
    )
    msa_config["max_msa_depth"] = 512

    valid_data_config = BioMolMonomerData.BioMolConfig(
        crop_config=client.config.data.crop.model_dump(),
        msa_config=msa_config,
        kmer_fast_align_config=client.config.data.kmer_fast_align.model_dump(),
        multistate_config=client.config.data.multistate.model_dump(),
        preprocess_config=preprocess_config,
    )

    prefetch_factor = (
        None
        if client.config.experiment.prefetch_factor == 0
        else int(client.config.experiment.prefetch_factor)
    )
    dataset = BioMolMonomerData(valid_data_config)
    valid_loader = dataset.create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=False,
        batch_size=client.config.experiment.num_batch,
        num_workers=client.config.experiment.num_workers,
        prefetch_factor=prefetch_factor,
    )

    client.logger.info("-" * 70)
    client.logger.info("")
    client.logger.info("Start sampling".center(70))
    client.logger.info("")
    client.logger.info("-" * 70)

    client.model.eval()
    for batch in valid_loader:
        client.generate_contact_map(
            batch,
            num_steps=client.config.experiment.eval_timesteps,
            save_dir=Path(save_dir) if save_dir else None,
        )

    if w and client.is_global_zero:
        wandb.finish()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
