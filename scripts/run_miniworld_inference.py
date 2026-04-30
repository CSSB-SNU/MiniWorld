"""Validation/inference script for MiniWorld (VE x-prediction, decoupled R/T).

Usage:
    python scripts/run_miniworld_inference.py validate \
        --config configs/miniworld/config_validation.yaml \
        --ckpt path/to/checkpoint.pt
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Literal

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf
from pydantic import BaseModel

from miniworld.configs import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TokenizerConfig,
    XPredDecoupledDiffuserConfig,
)
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.features.batch import Batch
from miniworld.data.io.to_cif import batch_to_cif
from miniworld.diffusion.decoupled_xpred import DecoupledXPredScheduler
from miniworld.loss import metrics
from miniworld.models.miniworld import Client, Model
from miniworld.models.miniworld.model import InferenceOutput

torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _training_sigma_y_hist(
    scheduler: DecoupledXPredScheduler,
    sigma_range: tuple[float, float],
    n_samples: int = 200_000,
    n_bins: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    sigma_y_samples, _, _ = scheduler.sample_noise(n_samples)
    log_bins = np.linspace(
        np.log10(sigma_range[0]),
        np.log10(sigma_range[1]),
        n_bins,
    )
    hist, bin_edges = np.histogram(
        np.log10(sigma_y_samples.numpy()),
        bins=log_bins,
        density=True,
    )
    bin_centers = 10 ** (0.5 * (bin_edges[:-1] + bin_edges[1:]))
    return bin_centers, hist


def plot_trajectory_rmsd(
    output: InferenceOutput,
    batch: Batch,
    sigmas_y: np.ndarray,
    scheduler: DecoupledXPredScheduler,
    save_path: Path,
) -> None:
    x0 = batch.structure.atom_pos[0].cpu()
    atom_mask = batch.structure.atom_mask[0].cpu()
    model_traj = output.model_traj[0]
    n_steps = model_traj.shape[0]
    rmsds = [
        metrics.cal_aligned_rmsd(torch.from_numpy(model_traj[t]), x0, atom_mask)
        for t in range(n_steps)
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas_y[:n_steps], rmsds, marker="o", markersize=2, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("sigma_y")
    ax.set_ylabel("RMSD (x0_hat vs x0)")
    ax.set_title("Per-step denoised prediction RMSD (x-prediction)")
    ax.grid(visible=True, alpha=0.3)
    ax2 = ax.twinx()
    bc, h = _training_sigma_y_hist(scheduler, (sigmas_y.min(), sigmas_y.max()))
    ax2.fill_between(bc, 0, h, alpha=0.15, color="gray", label="train p(sigma_y)")
    ax2.plot(bc, h, color="gray", alpha=0.4, linewidth=0.8)
    ax2.set_ylabel("sigma_y density", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def sigma_sweep_with_loss(
    client: Client,
    batch: Batch,
    n_sigmas: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[torch.Tensor], list[torch.Tensor]]:
    """Single-step denoising sweep using x-prediction.

    Model predicts x0/sigma_data. We recover x0_hat = F * sigma_data,
    then compute loss via client.diffuser.cal_loss.

    Returns sigmas, losses, and per-sigma x0_hat / x_noisy tensors (CPU).
    """
    from miniworld.models.miniworld.model import ModelWrapper
    from miniworld.utils.structure.se3 import apply_chain_rt, sample_rigid

    raw_model = getattr(client.model, "module", client.model)
    model_wrapper = ModelWrapper(raw_model)  # pyright: ignore[reportArgumentType]
    batch = batch.to(device=client.device)
    model_wrapper.prepare_condition(
        msa=batch.msa,
        template=batch.template,
        reference=batch.reference,
        scheme=batch.scheme,
        sequence=batch.sequence,
        structure=batch.structure,
    )

    x0 = batch.structure.atom_pos
    atom_mask = batch.structure.atom_pos_mask
    atom_chain_break = batch.scheme.atom_to_chain_id
    chain_num = int(atom_chain_break.max().item()) + 1

    scheduler = client.diffusion_scheduler
    sigma_data = scheduler.config.sigma_data
    sigma_min = scheduler.config.sigma_y_min * sigma_data
    sigma_max = scheduler.config.sigma_y_max * sigma_data
    sigmas = torch.logspace(
        float(np.log10(sigma_min)),
        float(np.log10(sigma_max)),
        n_sigmas,
    )

    losses = []
    x0_hats: list[torch.Tensor] = []
    x_noisies: list[torch.Tensor] = []
    for sigma_y_cpu in sigmas:
        sigma_y = sigma_y_cpu.to(client.device)
        sigma_R, sigma_T = scheduler.convert_to_sigma_rt(sigma_y.unsqueeze(0))

        y = x0 + sigma_y * torch.randn_like(x0)
        R, T = sample_rigid(
            sigma_R,
            sigma_T,
            C=chain_num,
            device=client.device,
            dtype=y.dtype,
        )
        x_noisy = apply_chain_rt(y, R, T, atom_chain_break)

        # x-prediction: c_in only, no c_skip/c_out
        t_emb = scheduler.noise_condition(sigma_y)
        c_in = scheduler.input_scale(sigma_y, sigma_T.squeeze())
        x_input = x_noisy * c_in
        x_update = model_wrapper(x_input, t_emb)

        # x0_hat via get_x0_hat (F * sigma_data)
        x0_hat = client.diffuser.get_x0_hat(
            x0=x0.unsqueeze(0),
            x_input=x_noisy.unsqueeze(0),
            x_update=x_update.unsqueeze(0).float(),
            sigma_y=sigma_y.unsqueeze(0),
            rotation_matrix=R,
            translation_vector=T,
            atom_to_combine=atom_chain_break,
            mask=atom_mask.unsqueeze(0),
        )

        loss = client.diffuser.cal_loss(
            x0=x0.unsqueeze(0),
            x_pred=x0_hat,
            sigma_y=sigma_y.unsqueeze(0),
            mask=atom_mask.unsqueeze(0),
        )
        losses.append(loss.item())
        # Normalize to (1, L, 3) for batch_to_cif: x0_hat is (1, 1, L, 3), x_noisy is (1, L, 3)
        x0_hats.append(x0_hat.detach().reshape(1, -1, 3).cpu())
        x_noisies.append(x_noisy.detach().reshape(1, -1, 3).cpu())

    return sigmas.numpy(), np.array(losses), x0_hats, x_noisies


def plot_sigma_sweep_loss(
    sigmas: np.ndarray,
    losses: np.ndarray,
    scheduler: DecoupledXPredScheduler,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, losses, marker="o", markersize=3, label="diffusion loss (x-pred)")
    ax.set_xscale("log")
    ax.set_xlabel("sigma_y")
    ax.set_ylabel("Diffusion loss")
    ax.set_title("Single-step denoising loss vs sigma_y (x-prediction)")
    ax.legend(loc="upper left")
    ax.grid(visible=True, alpha=0.3)
    ax2 = ax.twinx()
    bc, h = _training_sigma_y_hist(scheduler, (sigmas.min(), sigmas.max()))
    ax2.fill_between(bc, 0, h, alpha=0.15, color="gray", label="train p(sigma_y)")
    ax2.plot(bc, h, color="gray", alpha=0.4, linewidth=0.8)
    ax2.set_ylabel("sigma_y density", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    validate_db: BioMolDBConfig
    crop: CropConfig
    msa: MSAConfig
    tokenizer: TokenizerConfig


class ValidationConfig(BaseModel):
    seed: int = 0
    num_samples: int = 5
    timesteps: int = 100
    num_workers: int = 0
    compile: bool = True
    no_rt: bool = False
    combine_all: bool = False
    update_rule: Literal["ode", "ode_aligned", "x0_centered"] = "x0_centered"
    use_ema: bool = True


class Config(BaseModel):
    data: DataConfig
    validation: ValidationConfig
    model: Model.Config
    diffuser: XPredDecoupledDiffuserConfig
    loss: Client.LossConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fabric_from_torchrun() -> Fabric:
    world_size = os.environ.get("WORLD_SIZE")
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if world_size is None or local_world_size is None:
        return Fabric()
    devices = int(local_world_size)
    num_nodes = int(world_size) // devices
    return Fabric(devices=devices, num_nodes=num_nodes)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/miniworld_validation"),
    show_default=True,
    help="Root directory for validation outputs; results land in "
         "<output_dir>/<YYYY-MM-DD>/<HHMMSS>[_<job_name>]/.",
)
@click.option("--job-name", type=str)
@click.argument("overrides", type=str, nargs=-1)
def validate(  # noqa: PLR0915
    config: Path,
    ckpt: Path,
    output_dir: Path,
    job_name: str | None,
    overrides: tuple[str, ...],
) -> None:
    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name, overrides=list(overrides))
    cfg = Config.model_validate(cfg)
    fabric = _fabric_from_torchrun()
    fabric.launch()
    fabric.seed_everything(cfg.validation.seed)

    date_dir = output_dir / time.strftime("%Y-%m-%d")
    run_name = time.strftime("%H%M%S")
    if job_name:
        run_name += f"_{job_name}"
    run_sub_dir = date_dir / run_name
    run_sub_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = Client.TrainConfig(seed=cfg.validation.seed, use_ema=cfg.validation.use_ema)
    client = Client(
        Client.Config(
            train=train_cfg,
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
        log_path = run_sub_dir / "validation.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        client.logger.addHandler(file_handler)

    if cfg.validation.compile:
        torch._dynamo.config.cache_size_limit = 128  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001
        client.model.compile(dynamic=False)
        client.logger.info("Compiled model")

    config_dict = cfg.model_dump(mode="json")
    if fabric.is_global_zero:
        OmegaConf.save(OmegaConf.create(config_dict), run_sub_dir / "config.yaml")

    client.setup(fabric=fabric)

    state_dict = torch.load(ckpt, map_location="cpu")
    client.load_state_dict(state_dict, model_only=True)

    validate_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.validate_db,
        sampler_config=SamplerConfig(),
        tokenizer_config=cfg.data.tokenizer,
    )
    validate_dataset = BioMolData(validate_data_config)
    validate_dataloader = validate_dataset.create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.global_rank,
        seed=cfg.validation.seed,
        drop_last=False,
        batch_size=1,
        num_workers=cfg.validation.num_workers,
    )

    cif_dir = run_sub_dir / "structures"
    cif_dir.mkdir(parents=True, exist_ok=True)

    client.logger.info("Start x-prediction validation")
    client.model.eval()

    for batch_idx, raw_batch in enumerate(validate_dataloader):
        batch = raw_batch.to(device=client.device)
        name = str(batch.name[0])
        client.logger.info(
            (
                "rank=%d batch=%d %s | n_tokens=%d n_atoms=%d "
                "n_msa_valid=%d n_msa_bucket=%d "
                "n_template_valid=%d n_template_bucket=%d | mem=%.2fGB"
            ),
            fabric.global_rank,
            batch_idx,
            name,
            batch.token_length,
            batch.atom_length,
            batch.msa_count,
            batch.msa_depth,
            batch.template_count,
            batch.template_number,
            torch.cuda.max_memory_allocated() / 1024**3,
        )

        output = client.inference(
            batch,
            timesteps=cfg.validation.timesteps,
            no_rt=cfg.validation.no_rt,
            update_rule=cfg.validation.update_rule,
            combine_all=cfg.validation.combine_all,
        )

        quality = client.test_inference_quality(batch, output)
        client.logger.info(
            "batch=%d %s | rmsd=%.4f lddt=%.4f distogram_loss=%.4f",
            batch_idx,
            name,
            quality["best_rmsd"],
            quality["best_lddt"],
            quality["vald_distogram_loss"],
        )

        # Save structures
        batch_to_cif(batch, output.atom_pos_pred, cif_dir / f"{name}_pred.cif")
        batch_to_cif(batch, None, cif_dir / f"{name}_gt.cif")

        # Save trajectory CIFs: x0_hat (model prediction) and xt (noisy state) per step
        traj_dir = cif_dir / f"{name}_traj"
        traj_x0hat_dir = traj_dir / "x0hat"
        traj_xt_dir = traj_dir / "xt"
        traj_input_dir = traj_dir / "x_with_noise"
        traj_x0hat_dir.mkdir(parents=True, exist_ok=True)
        traj_xt_dir.mkdir(parents=True, exist_ok=True)
        traj_input_dir.mkdir(parents=True, exist_ok=True)
        model_traj = output.model_traj[0]
        inter_traj = output.inter_traj[0]
        input_traj = output.input_traj[0]
        scheduler = client.diffusion_scheduler
        time_steps = scheduler.sampling_time_steps(cfg.validation.timesteps)
        sigmas_y = time_steps[:-1].numpy()
        device = batch.structure.atom_pos.device
        for t in range(model_traj.shape[0]):
            tag = f"step{t:03d}_sigma{sigmas_y[t]:.4f}"
            x0_hat_t = torch.from_numpy(model_traj[t]).unsqueeze(0).to(device)
            xt = torch.from_numpy(inter_traj[t]).unsqueeze(0).to(device)
            x_with_noise_t = torch.from_numpy(input_traj[t]).unsqueeze(0).to(device)
            batch_to_cif(batch, x0_hat_t, traj_x0hat_dir / f"{tag}.cif")
            batch_to_cif(batch, xt, traj_xt_dir / f"{tag}.cif")
            batch_to_cif(batch, x_with_noise_t, traj_input_dir / f"{tag}.cif")

        # Trajectory RMSD plot
        plot_trajectory_rmsd(
            output,
            batch,
            sigmas_y,
            scheduler,
            run_sub_dir / f"{name}_trajectory_rmsd.png",
        )

        # Sigma sweep loss plot
        sweep_sigmas, sweep_losses, sweep_x0_hats, sweep_x_noisies = (
            sigma_sweep_with_loss(client, batch, n_sigmas=50)
        )
        plot_sigma_sweep_loss(
            sweep_sigmas,
            sweep_losses,
            scheduler,
            run_sub_dir / f"{name}_sigma_sweep_loss.png",
        )

        # Save per-sigma single-step denoised structures (x0_hat) and noisy inputs
        sweep_dir = cif_dir / f"{name}_sweep"
        sweep_x0hat_dir = sweep_dir / "x0hat"
        sweep_noisy_dir = sweep_dir / "noisy"
        sweep_x0hat_dir.mkdir(parents=True, exist_ok=True)
        sweep_noisy_dir.mkdir(parents=True, exist_ok=True)
        for i, sigma_y_val in enumerate(sweep_sigmas):
            tag = f"step{i:03d}_sigma{float(sigma_y_val):.4f}"
            batch_to_cif(
                batch,
                sweep_x0_hats[i].to(device),
                sweep_x0hat_dir / f"{tag}.cif",
            )
            batch_to_cif(
                batch,
                sweep_x_noisies[i].to(device),
                sweep_noisy_dir / f"{tag}.cif",
            )

        client.logger.info("Saved results for %s", name)

    client.logger.info("Validation complete. Results saved to %s", run_sub_dir)


# ---------------------------------------------------------------------------
# Inference (real inference: fasta + a3m + (optional) template + contacts JSON)
# ---------------------------------------------------------------------------


class InferConfig(BaseModel):
    seed: int = 0
    timesteps: int = 100
    compile: bool = True
    no_rt: bool = False
    combine_all: bool = False
    update_rule: Literal["ode", "ode_aligned", "x0_centered"] = "x0_centered"
    use_ema: bool = True
    max_msa_depth: int = 256
    missing_policy: Literal["query", "gap"] = "query"


class InferenceConfig(BaseModel):
    infer: InferConfig
    model: Model.Config
    diffuser: XPredDecoupledDiffuserConfig
    loss: Client.LossConfig


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--spec",
    "spec_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON spec describing fasta / a3m / template / contacts inputs.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/miniworld_inference"),
    show_default=True,
    help="Root directory for inference outputs; results land in "
         "<output_dir>/<YYYY-MM-DD>/<HHMMSS>/<job_name|spec.name>/.",
)
@click.option("--job-name", type=str)
@click.argument("overrides", type=str, nargs=-1)
def inference(  # noqa: PLR0915
    config: Path,
    ckpt: Path,
    spec_path: Path,
    output_dir: Path,
    job_name: str | None,
    overrides: tuple[str, ...],
) -> None:
    from miniworld.data.inference import InferenceSpec, build_inference_batch

    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name, overrides=list(overrides))
    cfg = InferenceConfig.model_validate(cfg)
    fabric = _fabric_from_torchrun()
    fabric.launch()
    fabric.seed_everything(cfg.infer.seed)

    spec = InferenceSpec.from_json(spec_path)

    time_dir = (
        output_dir
        / time.strftime("%Y-%m-%d")
        / time.strftime("%H%M%S")
    )
    run_id = job_name if job_name else spec.name
    run_sub_dir = time_dir / run_id
    run_sub_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = Client.TrainConfig(seed=cfg.infer.seed, use_ema=cfg.infer.use_ema)
    client = Client(
        Client.Config(
            train=train_cfg,
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
        log_path = run_sub_dir / "inference.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        client.logger.addHandler(file_handler)

    if cfg.infer.compile:
        torch._dynamo.config.cache_size_limit = 128  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001
        client.model.compile(dynamic=False)
        client.logger.info("Compiled model")

    config_dict = cfg.model_dump(mode="json")
    if fabric.is_global_zero:
        OmegaConf.save(OmegaConf.create(config_dict), run_sub_dir / "config.yaml")
        (run_sub_dir / "spec.json").write_text(spec_path.read_text())

    client.setup(fabric=fabric)

    state_dict = torch.load(ckpt, map_location="cpu")
    client.load_state_dict(state_dict, model_only=True)

    cif_dir = run_sub_dir / "structures"
    cif_dir.mkdir(parents=True, exist_ok=True)

    client.model.eval()

    # ``combine_all=True`` zeroes out atom_to_combine in the solver
    # (decoupled_xpred/solver.py:232). When set together with
    # ``spec.combine_groups``, the explicit grouping is silently ignored —
    # but empirically this preserves the trained model's behaviour, while
    # forcing combine_all=False has been observed to collapse the backbone
    # (CA–CA ≈ 3.0 Å instead of 3.8 Å). Warn the user about the conflict
    # but do **not** auto-override here.
    if cfg.infer.combine_all and spec.combine_groups:
        client.logger.warning(
            "spec.combine_groups is set but cfg.infer.combine_all=true: "
            "the solver will zero atom_to_combine, so the explicit groups "
            "have no effect on per-frame SE(3). Set infer.combine_all=false "
            "to honor them, but be aware the current checkpoint may not "
            "denoise multi-frame structures cleanly.",
        )

    n_trunk = spec.n_trunk_samples
    n_diffusion = spec.n_diffusion_samples
    chunk_size = min(spec.diffusion_batch_size, n_diffusion)
    save_traj = spec.save_trajectory
    client.logger.info(
        "Inference scaling: n_trunk=%d n_diffusion=%d (chunk=%d) "
        "save_trajectory=%s -> %d total structures",
        n_trunk, n_diffusion, chunk_size, save_traj, n_trunk * n_diffusion,
    )

    scheduler = client.diffusion_scheduler
    time_steps = scheduler.sampling_time_steps(cfg.infer.timesteps)
    sigmas_y = time_steps[:-1].numpy()

    for i_trunk in range(n_trunk):
        # Each trunk index uses a fresh seed -> different MSA sample +
        # (when ``rng`` paths consume it) a different SE(3) reference.
        trunk_seed = cfg.infer.seed + i_trunk
        client.logger.info(
            "Building Batch from spec %s (trunk %d/%d, seed=%d)",
            spec_path, i_trunk + 1, n_trunk, trunk_seed,
        )
        batch = build_inference_batch(
            spec,
            max_msa_depth=cfg.infer.max_msa_depth,
            missing_policy=cfg.infer.missing_policy,
            seed=trunk_seed,
        )
        name = batch.name[0]
        if i_trunk == 0:
            client.logger.info(
                "Built batch %s | n_tokens=%d n_atoms=%d n_msa=%d",
                name, batch.token_length, batch.atom_length, batch.msa_count,
            )

        wrapper, batch = client.prepare(batch)
        device = batch.structure.atom_pos.device

        for chunk_start in range(0, n_diffusion, chunk_size):
            cur_chunk = min(chunk_size, n_diffusion - chunk_start)
            # Reproducible per (trunk, chunk_start): same spec + cfg.infer.seed
            # always reproduces the same N1*N2 structures.
            torch.manual_seed(trunk_seed * 100003 + chunk_start * 1009)
            output = client.sample(
                wrapper, batch,
                n_samples=cur_chunk,
                timesteps=cfg.infer.timesteps,
                no_rt=cfg.infer.no_rt,
                update_rule=cfg.infer.update_rule,
                combine_all=cfg.infer.combine_all,
            )

            for k in range(cur_chunk):
                j_diff = chunk_start + k
                tag_ij = f"t{i_trunk:02d}_d{j_diff:02d}"
                pred_k = output.atom_pos_pred[k:k + 1]
                batch_to_cif(batch, pred_k, cif_dir / f"{name}_{tag_ij}_pred.cif")

                if not save_traj:
                    continue
                traj_dir = cif_dir / f"{name}_{tag_ij}_traj"
                traj_x0hat_dir = traj_dir / "x0hat"
                traj_xt_dir = traj_dir / "xt"
                traj_input_dir = traj_dir / "x_with_noise"
                traj_x0hat_dir.mkdir(parents=True, exist_ok=True)
                traj_xt_dir.mkdir(parents=True, exist_ok=True)
                traj_input_dir.mkdir(parents=True, exist_ok=True)
                model_traj_k = output.model_traj[k]
                inter_traj_k = output.inter_traj[k]
                input_traj_k = output.input_traj[k]
                for t in range(model_traj_k.shape[0]):
                    tag = f"step{t:03d}_sigma{sigmas_y[t]:.4f}"
                    x0_hat_t = torch.from_numpy(model_traj_k[t]).unsqueeze(0).to(device)
                    xt = torch.from_numpy(inter_traj_k[t]).unsqueeze(0).to(device)
                    x_with_noise_t = torch.from_numpy(input_traj_k[t]).unsqueeze(0).to(device)
                    batch_to_cif(batch, x0_hat_t, traj_x0hat_dir / f"{tag}.cif")
                    batch_to_cif(batch, xt, traj_xt_dir / f"{tag}.cif")
                    batch_to_cif(batch, x_with_noise_t, traj_input_dir / f"{tag}.cif")

            client.logger.info(
                "Trunk %d/%d, diffusion samples %d-%d/%d done",
                i_trunk + 1, n_trunk,
                chunk_start + 1, chunk_start + cur_chunk, n_diffusion,
            )

    client.logger.info("Inference complete. Results saved to %s", run_sub_dir)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
