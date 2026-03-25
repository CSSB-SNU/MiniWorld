import logging
import time
from pathlib import Path

import click
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf
from pydantic import BaseModel
from team_gm.core.callbacks import Callback
from team_gm.utils.script_utils import MetricsAggregator

import wandb
from miniworld.data.dataloader.dataloader_atom_token import (
    BioMolData,
    BioMolDBConfig,
    CropConfig,
    EdgeWeightConfig,
    MSAConfig,
)
from miniworld.diffusion.configs import EDMDiffuserConfig
from miniworld.models.af3_atom_token_fingerprint import Client, Model
from miniworld.utils import get_step_decay_scheduler_with_warmup
from miniworld.data.to_cif import batch_to_cif

torch.set_float32_matmul_precision("high")  # noqa: ERA001
# anomaly detection
torch.autograd.set_detect_anomaly(False)

class DataConfig(BaseModel):
    """Configuration for data loading."""

    train_db: BioMolDBConfig
    valid_db: BioMolDBConfig
    edge_weight: EdgeWeightConfig
    crop: CropConfig
    msa: MSAConfig


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
    fabric = Fabric()
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

    optimizer = torch.optim.AdamW(
        client.model.parameters(),
        cfg.train.max_lr,
    )
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
    
    if ckpt:
        state_dict = torch.load(ckpt, map_location="cpu")
        client.load_state_dict(state_dict)
    
    train_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db,
        edge_weight_config=cfg.data.edge_weight,
    )
    valid_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.valid_db,
        edge_weight_config=cfg.data.edge_weight,
    )

    train_dataloader = BioMolData(train_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=True,
        # use_adaptive_sampler=True,
        use_adaptive_sampler=False,
        batch_size=cfg.train.num_batch,
        num_workers=cfg.train.num_workers,
        prefetch_factor=cfg.train.prefetch_factor,
        shuffle=False,
    )
    sampler = cast("AdaptiveEdgeSampler", train_dataloader.sampler)
    if ckpt:
        sampler_state_path = cfg.data.edge_weight.state_load_path
        if sampler_state_path is None:
            msg = f"No sampler state load path provided in config, but checkpoint {ckpt} is given. Skipping sampler state loading."
            client.logger.warning(msg)
        else:
            sampler.load_sampler_state(sampler_state_path)
            msg = f"Loaded sampler state from {sampler_state_path}"
            client.logger.info(msg)

    valid_dataloader = BioMolData(valid_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=False,
        use_adaptive_sampler=False,
        batch_size=cfg.train.num_batch,  # or 1
        num_workers=0,
    )

    world_size = fabric.world_size
    train_num_item = cfg.train.train_item // world_size
    valid_num_item = cfg.train.valid_item // world_size
    min_train_loss = float("inf")

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
        means = train_aggregator.log_epoch()

        checkpoint_path = checkpoint_dir / "last.pt"
        client.save_checkpoint(checkpoint_path)
        if client.epoch % cfg.train.save_freq == 0:
            checkpoint_path = checkpoint_dir / f"epoch={client.epoch:04d}.pt"
            client.save_checkpoint(checkpoint_path)

        if client.is_global_zero:
            train_loss = means["total_loss"]
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                best_checkpoint_path = checkpoint_dir / f"best.pt"
                client.save_checkpoint(best_checkpoint_path)
                client.logger.info(
                    "Save best checkpoint: %s (train loss: %.4g)",
                    best_checkpoint_path.name,
                    train_loss,
                )
        
        if (client.epoch - 1) % cfg.train.eval_freq == 0:
            valid_dataloader.sampler.set_epoch(client.epoch)  # pyright: ignore[reportAttributeAccessIssue]
            client.logger.info("Validation Epoch %d", client.epoch)
            for n_item, result in enumerate(client.validation_epoch(valid_dataloader)):
                valid_aggregator.log_step(result, ignore_step=True)
                if n_item + 1 >= valid_num_item:
                    client.call_callbacks("on_validation_epoch_end")
                    break

            valid_aggregator.log_epoch()


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

    raw_cfg = OmegaConf.load(config)
    cfg = ValidateConfig.model_validate(raw_cfg)
    if not ckpt:
        msg = "You must provide a checkpoint file."
        raise ValueError(msg)
    state_dict = torch.load(ckpt, map_location="cpu")
    ckpt_cfg = Client.Config.model_validate(state_dict["config"])
    # Validation can override the token embedding path from the runtime config.
    override_embedding_path = OmegaConf.select(
        raw_cfg,
        "model.token_embedding.embedding_path",
    )
    if override_embedding_path:
        ckpt_cfg.model.token_embedding.embedding_path = Path(override_embedding_path)
    client = Client(ckpt_cfg)
    client.load_state_dict(state_dict, model_only=True)
    client.model.to("cuda")
    if cfg.train.compile:
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
        if cfg.train.prefetch_factor == 0
        else int(cfg.train.prefetch_factor)
    )
    valid_loader = BioMolData(valid_data_config).create_ddp_dataloader(
        world_size=world_size,
        rank=local_rank,
        drop_last=False,
        batch_size=cfg.train.num_batch,  # or 1
        num_workers=cfg.train.num_workers,
        prefetch_factor=prefetch_factor,
    )

    client.logger.info("-" * 70)
    client.logger.info("")
    client.logger.info("Start validation".center(70))
    client.logger.info("")
    client.logger.info("-" * 70)

    valid_num_item = cfg.train.valid_item // world_size

    for ii, _batch in enumerate(valid_loader):
        batch = fabric.to_device(_batch)
        click.echo(
            f"name : {batch.name[0]} \
            residue length : {batch.scheme.token_idx.shape[1]} \
            atom length : {batch.structure.atom_pos.shape[1]}",
        )

        batch = batch.duplicate(client.config.train.eval_sample_num)
        output = client.inference(
            batch,
            timesteps=client.config.train.eval_timesteps,
        )

        result_dict = client.test_inference_quality(
            batch,
            output,
        )
        batch_to_cif(
            batch,
            output.atom_pos_pred,

            Path(f"outputs/atom_token_fingerprint_config2/{batch.name[0]}_pred.cif"),
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
