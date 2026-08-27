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
from team_gm.core.callbacks import Callback, ModelEMA
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

from miniworld.data.dataloader.dataloader_explicit import BioMolData
from miniworld.data.dataloader.dataloader_infer import (
    NoModifiedResidueError,
    apply_modified_focus,
)
from miniworld.models import ExplicitClient as Client
from miniworld.models.af3_like_explicit import Model as Model
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


def _strip_compile_prefix(sd: dict) -> dict:
    """Remove leading '_orig_mod.' prefix left by torch.compile in state_dict keys."""
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def _strip_wrapper_prefix(name: str) -> str:
    """Strip Lightning Fabric (`_forward_module.`) and DDP (`module.`) prefixes
    that ``named_parameters`` adds after ``fabric.setup_module``. State_dicts
    saved by Lightning are *unwrapped* (no prefix), so we must align the names
    from ``named_parameters`` to the same flat naming used in the checkpoint.
    Strip iteratively because Fabric+DDP nests ``_forward_module.module.``."""
    while True:
        new = name.removeprefix("_forward_module.").removeprefix("module.")
        if new == name:
            return name
        name = new


def _reconcile_state_dict_with_model(client, state_dict: dict) -> bool:
    """In-place: align checkpoint's model/EMA dicts with the current model.

    Handles architecture changes between save and load (e.g. enabling
    ``use_qk_norm`` adds RMSNorm γ params; disabling something removes keys).
    Missing keys are filled from the current model's init; extra keys are
    dropped. Compile prefixes are stripped on both model_state_dict and
    ema_state_dict. When the architecture differs, optimizer state is also
    remapped so Adam moments survive for unchanged params.

    Returns True iff the checkpoint's architecture differed from the current
    model. The caller can use this for diagnostic logging; loading itself is
    safe regardless.
    """
    msd = _strip_compile_prefix(state_dict["model_state_dict"])
    if state_dict.get("ema_state_dict") is not None:
        state_dict["ema_state_dict"] = _strip_compile_prefix(state_dict["ema_state_dict"])

    # Snapshot the OLD model_state_dict key order before any mutation. We need
    # this (not the post-fill order) to recover the old param positions during
    # optimizer-state migration.
    old_msd_keys_in_order = list(msd.keys())

    cur = client.model.state_dict()
    missing = [k for k in cur if k not in msd]
    extra = [k for k in msd if k not in cur]
    if missing:
        for k in missing:
            msd[k] = cur[k].detach().clone()
        preview = missing[:8] + (["..."] if len(missing) > 8 else [])
        client.logger.info(
            "Filled %d missing key(s) from current model init: %s",
            len(missing), preview,
        )
    if extra:
        for k in extra:
            del msd[k]
        preview = extra[:8] + (["..."] if len(extra) > 8 else [])
        client.logger.info(
            "Dropped %d extra key(s) not in current model: %s",
            len(extra), preview,
        )
    state_dict["model_state_dict"] = msd
    arch_changed = bool(missing) or bool(extra)
    if arch_changed:
        _migrate_optimizer_state(client, state_dict, old_msd_keys_in_order)
    return arch_changed


def _migrate_optimizer_state(
    client, state_dict: dict, old_msd_keys_in_order: list[str],
) -> None:
    """Remap optimizer state across an architecture change so old Adam moments
    are preserved for unchanged params; new params default-init at first step.

    Mutates ``state_dict["optimizer_state_dict"]`` in place. On any
    inconsistency, drops the optimizer state entirely so the caller can fall
    back to ``model_only=True`` semantics safely.
    """
    opt_state = state_dict.get("optimizer_state_dict")
    if opt_state is None:
        return  # model-only checkpoint — nothing to migrate

    # Use *unwrapped* names so they match the saved state_dict keys (Fabric
    # serializes flat keys, but named_parameters yields '_forward_module.module.X'
    # after fabric.setup_module). Position in the list is preserved, which is
    # what the optimizer needs for index-based loading.
    new_named = [
        (_strip_wrapper_prefix(n), p)
        for n, p in client.model.named_parameters() if p.requires_grad
    ]
    n_new = len(new_named)
    n_saved = sum(len(g["params"]) for g in opt_state["param_groups"])

    new_name_to_idx = {n: i for i, (n, _) in enumerate(new_named)}
    new_param_names = set(new_name_to_idx)

    # Recover old param order from the *pre-fill* model_state_dict insertion
    # order (matches old model.named_parameters() order, modulo buffers).
    # Filtering by new_param_names drops buffers and any "extra" keys (e.g.
    # params removed by an arch change).
    old_param_order = [k for k in old_msd_keys_in_order if k in new_param_names]

    # Name-based remap: saved state[old_idx] → new_state[new_idx]. Always run
    # this when called (arch_changed is True): even when n_new == n_saved
    # (one-out-one-in swap), positions may have shifted and a positional load
    # would corrupt state silently. New params (no entry at their new idx)
    # default-init at the first optimizer.step().
    new_state = {}
    for old_idx, name in enumerate(old_param_order):
        if old_idx in opt_state["state"]:
            new_state[new_name_to_idx[name]] = opt_state["state"][old_idx]

    # Keep the saved param-group hyperparams (lr / initial_lr / betas / ...).
    # The saved scheduler state will be restored alongside, so saved optimizer
    # and saved scheduler stay mutually consistent at the saved point in time.
    sg = opt_state["param_groups"][0]
    new_pg = {k: v for k, v in sg.items() if k != "params"}
    new_pg["params"] = list(range(n_new))

    state_dict["optimizer_state_dict"] = {
        "state": new_state,
        "param_groups": [new_pg],
    }
    n_default_init = n_new - len(new_state)
    client.logger.info(
        "Migrated optimizer state: %d Adam moments preserved (from %d saved), "
        "%d param(s) will default-init at first step. Saved hyperparams: "
        "lr=%.2e, initial_lr=%.2e.",
        len(new_state), n_saved, n_default_init,
        new_pg.get("lr", float("nan")),
        new_pg.get("initial_lr", float("nan")),
    )


def _log_lr_consistency(client) -> None:
    """Log optimizer initial_lr vs scheduler base_lrs after a checkpoint load.

    They should be equal: both come from the same checkpoint (or both default
    to ``cfg.train.max_lr`` on a fresh start). If they disagree, the scheduler
    will compute LR from its base_lrs while the optimizer thinks initial_lr is
    something else — silent training-curve drift.
    """
    if client._optimizer is None or client.scheduler is None:
        return
    pg = client._optimizer.param_groups[0]
    base_lrs = getattr(client.scheduler, "base_lrs", None)
    initial_lr = pg.get("initial_lr", float("nan"))
    cur_lr = pg.get("lr", float("nan"))
    client.logger.info(
        "LR consistency check after load: optimizer initial_lr=%.2e, lr=%.2e; "
        "scheduler base_lrs=%s, last_epoch=%s.",
        initial_lr, cur_lr,
        base_lrs, getattr(client.scheduler, "last_epoch", "?"),
    )
    if base_lrs is not None and any(abs(b - initial_lr) > 1e-12 for b in base_lrs):
        client.logger.warning(
            "LR consistency: scheduler.base_lrs %s differs from optimizer "
            "initial_lr %.2e — scheduler-computed LR will not match.",
            base_lrs, initial_lr,
        )


def _extend_ema_with_new_params(client) -> None:
    """Add any model parameters not yet tracked by EMA (e.g. RMSNorm γ from a
    newly-enabled use_qk_norm) using the model's current value as the EMA seed.
    Must be called *after* ``client.load_state_dict``.
    """
    from team_gm.core.callbacks import ModelEMA
    ema_cb = next((cb for cb in client._callbacks if isinstance(cb, ModelEMA)), None)
    if ema_cb is None or ema_cb._ema_params is None:
        return
    added = []
    with torch.no_grad():
        for name, param in client.model.named_parameters():
            if param.requires_grad and name not in ema_cb._ema_params:
                ema_cb._ema_params[name] = param.detach().clone()
                added.append(name)
    if added:
        preview = added[:8] + (["..."] if len(added) > 8 else [])
        client.logger.info(
            "EMA dict extended with %d new param(s): %s", len(added), preview,
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
        _reconcile_state_dict_with_model(client, state_dict)
        client.load_state_dict(state_dict)
        _extend_ema_with_new_params(client)
        _log_lr_consistency(client)

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
            # Validate on non-EMA weights; next epoch start is a no-op restore.
            ema_cb = next(
                (cb for cb in client._callbacks if isinstance(cb, ModelEMA)),
                None,
            )
            if ema_cb is not None:
                ema_cb._restore_original_params(client)  # noqa: SLF001

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

    # Infer on non-EMA weights: load_state_dict swaps in EMA params via the
    # ModelEMA callback, so restore the original (non-EMA) weights here.
    ema_cb = next(
        (cb for cb in client._callbacks if isinstance(cb, ModelEMA)),
        None,
    )
    if ema_cb is not None:
        ema_cb._restore_original_params(client)  # noqa: SLF001

    infer_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.valid_db,
        sampler_config=None,
        tokenizer_config=cfg.data.tokenizer,
    )
    infer_dataset = BioMolData(infer_data_config)
    # apply_modified_focus(infer_dataset)
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

    n_success, n_fail, n_skipped = 0, 0, 0
    infer_iter = iter(infer_dataloader)
    idx = 0
    while True:
        if num_items is not None and idx >= num_items:
            break
        try:
            _batch = next(infer_iter)
        except StopIteration:
            break
        except NoModifiedResidueError as e:
            logger.info("[%d/%d] No modified residue, skipping: %s",
                        idx + 1, total, e)
            n_skipped += 1
            idx += 1
            continue
        except Exception:
            logger.exception("[%d/%d] Failed to load item, skipping", idx + 1, total)
            n_skipped += 1
            idx += 1
            continue

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

            # idx prefix: distinct edges from same parent PDB collapse to one batch.name
            ref_path = ref_dir / f"{idx:04d}_{name}.cif"
            batch_to_cif(batch, atom_pos_pred=None, save_path=ref_path)

            for s in range(num_samples):
                pred_path = pred_dir / f"{idx:04d}_{name}_sample{s}.cif"
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

        idx += 1

    logger.info(
        "Inference complete: %d succeeded, %d failed, %d skipped (no modified residue). "
        "Results saved to %s",
        n_success, n_fail, n_skipped, output_dir,
    )


if __name__ == "__main__":
    # set mp start method
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
