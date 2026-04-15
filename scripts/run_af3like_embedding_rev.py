from __future__ import annotations

import logging
import os
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message=".*torch.jit.script_method.*", category=DeprecationWarning) 
import click
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
    EDMDiffuserConfig,
    MSAConfig,
    SamplerConfig,
    TemplateConfig,
    TokenizerConfig,
)
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.models import EmbeddingClient_rev as Client
from miniworld.models.af3_like_embedding import Model_rev as Model
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
    template: TemplateConfig = TemplateConfig()


class Config(BaseModel):
    """Overall configuration."""

    data: DataConfig
    train: Client.TrainConfig
    model: Model.Config
    diffuser: EDMDiffuserConfig
    loss: Client.LossConfig


def _fabric_from_torchrun(**fabric_kwargs) -> Fabric:
    """Create Fabric with node/device counts inherited from torchrun."""
    world_size = os.environ.get("WORLD_SIZE")
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if world_size is None or local_world_size is None:
        return Fabric(**fabric_kwargs)

    devices = int(local_world_size)
    num_nodes = int(world_size) // devices
    return Fabric(devices=devices, num_nodes=num_nodes, **fabric_kwargs)


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
    torch._dynamo.reset()  # clear stale compile cache (prevents bf16/fp32 mismatch on resume)

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
        sampler_config=None,
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

    valid_dataset = BioMolData(valid_data_config)
    valid_dataloader = valid_dataset.create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.global_rank,
        seed=cfg.train.seed,
        drop_last=True,
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
        train_dataset.set_epoch(client.epoch)

        for step, result in enumerate(client.training_epoch(train_dataloader)):
            train_aggregator.log_step(result)
            if step >= train_num_item:
                break
        train_aggregator.log_epoch()

        checkpoint_path = checkpoint_dir / "last.pt"
        client.save_checkpoint(checkpoint_path)
        if client.epoch % cfg.train.save_freq == 0:
            checkpoint_path = checkpoint_dir / f"epoch={client.epoch:04d}.pt"
            client.save_checkpoint(checkpoint_path)

        if client.epoch % cfg.train.eval_freq == 0:
            valid_dataloader.sampler.set_epoch(client.epoch)  # pyright: ignore[reportAttributeAccessIssue]
            valid_dataset.set_epoch(client.epoch)
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
    required=True,
    help="config file",
)
@click.option(
    "--ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="checkpoint file",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="directory to write predicted and reference CIF files",
)
@click.option("--max-tokens", type=int, default=None, help="override max_tokens in crop config")
@click.option("--max-atoms", type=int, default=None, help="override max_atoms in crop config")
@click.option("--timesteps", type=int, default=100, help="number of diffusion timesteps")
@click.option("--num-samples", type=int, default=5, help="number of samples per structure")
@click.option("--num-items", type=int, default=None, help="max number of items to process")
@click.option("--seed", type=int, default=0, help="random seed")
@click.argument("overrides", type=str, nargs=-1)
def infer(
    config: Path,
    ckpt: Path,
    output_dir: Path,
    max_tokens: int | None,
    max_atoms: int | None,
    timesteps: int,
    num_samples: int,
    num_items: int | None,
    seed: int,
    overrides: tuple[str, ...],
):
    from miniworld.data.io.to_cif import batch_to_cif

    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name, overrides=list(overrides))
    cfg = Config.model_validate(cfg)

    if max_tokens is not None:
        cfg.data.crop.max_tokens = max_tokens
    if max_atoms is not None:
        cfg.data.crop.max_atoms = max_atoms

    fabric = Fabric(devices=1)
    fabric.launch()
    fabric.seed_everything(seed)

    client = Client(
        Client.Config(
            train=cfg.train,
            model=cfg.model,
            diffuser=cfg.diffuser,
            loss=cfg.loss,
        ),
    )
    client.setup(fabric=fabric)

    state_dict = torch.load(ckpt, map_location="cpu")
    client.load_state_dict(state_dict, model_only=True)

    infer_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.valid_db,
        sampler_config=None,
        tokenizer_config=cfg.data.tokenizer,
    )
    infer_dataset = BioMolData(infer_data_config)
    infer_dataloader = infer_dataset.create_ddp_dataloader(
        world_size=1,
        rank=0,
        seed=seed,
        drop_last=False,
        batch_size=1,
        num_workers=0,
    )

    pred_dir = output_dir / "predicted"
    ref_dir = output_dir / "reference"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    client.model.eval()
    total = num_items if num_items is not None else len(infer_dataset)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logger = logging.getLogger("infer")
    logger.info(
        "Starting inference: max_tokens=%s, max_atoms=%s, timesteps=%d, num_samples=%d",
        cfg.data.crop.max_tokens, cfg.data.crop.max_atoms, timesteps, num_samples,
    )

    n_success, n_fail = 0, 0
    for idx, _batch in enumerate(infer_dataloader):
        if num_items is not None and idx >= num_items:
            break
        batch = _batch
        batch = batch.to(device=client.device)
        name = batch.name[0]
        name = name.replace('[','').replace("'","").replace(']','')
        logger.info("[%d/%d] Processing %s (tokens=%d, atoms=%d)",
                    idx + 1, total, name, batch.token_length, batch.atom_length)

        try:
            # Run inference with num_samples duplicates
            infer_batch = batch.duplicate(num_samples)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = client.inference(infer_batch, timesteps=timesteps)

            # Write reference CIF only on success
            ref_path = ref_dir / f"{name}.cif"
            batch_to_cif(batch, atom_pos_pred=None, save_path=ref_path)

            # Write predicted CIF for each sample
            for s in range(num_samples):
                pred_path = pred_dir / f"{name}_sample{s}.cif"
                atom_pos_s = output.atom_pos_pred[s : s + 1]
                batch_to_cif(batch, atom_pos_pred=atom_pos_s, save_path=pred_path)
            n_success += 1
        except torch.cuda.OutOfMemoryError:
            logger.warning("[%d] OOM on %s (tokens=%d, atoms=%d), skipping",
                           idx + 1, name, batch.token_length, batch.atom_length)
            torch.cuda.empty_cache()
            n_fail += 1
        except Exception:
            logger.exception("[%d] Failed on %s, skipping", idx + 1, name)
            n_fail += 1

    logger.info("Inference complete: %d succeeded, %d failed. Results saved to %s",
                n_success, n_fail, output_dir)


if __name__ == "__main__":
    # set mp start method
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
