"""Sampling / inference script for the MiniWorld EDM variant (AF3-like diffusion).

Generates 3D structures from the trained EDM checkpoint (e.g. ``epoch=0440.pt``,
the 3/24/3 diffusion module trained on top of the frozen distogram trunk) by
sampling targets out of the training/validation LMDB and running the AF3 ODE
solver. There is no EDM equivalent of ``run_miniworld_inference.py`` (which is
hardwired to the decoupled-xpred ``miniworld`` model), hence this script.

The model / diffuser / loss config is read straight from the checkpoint, so the
architecture always matches the weights. EMA weights are applied automatically
on load (the ``ModelEMA`` callback swaps the trained-param EMA shadow into the
model). Per target we save the predicted + ground-truth CIF and log aligned
RMSD / atom lDDT / distogram loss.

Usage (single GPU):
    pixi run python scripts/run_miniworld_edm_inference.py sample \
        --config configs/miniworld/config_exp_msa3_24_3_edm.yaml \
        --ckpt epoch=0440.pt \
        --num-targets 4 --n-samples 2 --timesteps 100 \
        data=local_sample_edm
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import click
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
)
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.io.to_cif import batch_to_cif
from miniworld.loss import metrics
from miniworld.models.miniworld_edm import Client

torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)


class DataConfig(BaseModel):
    """Data-loading sub-config (mirrors the EDM training script)."""

    train_db: BioMolDBConfig
    crop: CropConfig
    msa: MSAConfig
    tokenizer: TokenizerConfig
    sampler: SamplerConfig


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Hydra config (used only for the `data` subtree).",
)
@click.option(
    "--ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="EDM checkpoint; model/diffuser/loss config is read from it.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/miniworld_edm_sample"),
    show_default=True,
)
@click.option("--num-targets", type=int, default=4, show_default=True,
              help="Number of targets to sample from the DB.")
@click.option("--n-samples", type=int, default=2, show_default=True,
              help="Diffusion samples per target (augmentation axis).")
@click.option("--timesteps", type=int, default=100, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--compile/--no-compile", "do_compile", default=False, show_default=True)
@click.option("--ema/--no-ema", "use_ema", default=True, show_default=True,
              help="Use EMA weights (default) or the raw trained weights.")
@click.option("--job-name", type=str, default=None)
@click.argument("overrides", type=str, nargs=-1)
def sample(  # noqa: PLR0915
    config: Path,
    ckpt: Path,
    output_dir: Path,
    num_targets: int,
    n_samples: int,
    timesteps: int,
    seed: int,
    do_compile: bool,
    use_ema: bool,
    job_name: str | None,
    overrides: tuple[str, ...],
) -> None:
    # --- compose data config -------------------------------------------------
    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name, overrides=list(overrides))
    data_cfg = DataConfig.model_validate(OmegaConf.to_container(cfg.data, resolve=True))

    fabric = Fabric(devices=1, num_nodes=1)
    fabric.launch()
    fabric.seed_everything(seed)

    date_dir = output_dir / time.strftime("%Y-%m-%d")
    run_name = time.strftime("%H%M%S")
    if job_name:
        run_name += f"_{job_name}"
    run_sub_dir = date_dir / run_name
    run_sub_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = run_sub_dir / "structures"
    cif_dir.mkdir(parents=True, exist_ok=True)

    # --- build client from the checkpoint's own config -----------------------
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
    client_config = Client.Config.model_validate(state_dict["config"])
    # use_ema controls whether the ModelEMA callback is registered: when True the
    # checkpoint's EMA shadow is swapped into the (trained) params on load; when
    # False the raw trained weights from model_state_dict are used as-is.
    client_config.train.use_ema = use_ema
    client = Client(client_config)

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(run_sub_dir / "sample.log")
    fh.setFormatter(formatter)
    client.logger.addHandler(fh)
    client.logger.info(
        "ckpt=%s epoch=%s step=%s | num_targets=%d n_samples=%d timesteps=%d ema=%s",
        ckpt, state_dict.get("epoch"), state_dict.get("global_step"),
        num_targets, n_samples, timesteps, use_ema,
    )

    if do_compile:
        torch._dynamo.config.cache_size_limit = 128  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001
        client.model.compile(dynamic=False)
        client.logger.info("Compiled model")

    OmegaConf.save(
        OmegaConf.create({"data": OmegaConf.to_container(cfg.data, resolve=True),
                          "model": client_config.model.model_dump(mode="json"),
                          "diffuser": client_config.diffuser.model_dump(mode="json")}),
        run_sub_dir / "config.yaml",
    )

    client.setup(fabric=fabric)
    # Applies EMA shadow to the trained (diffusion) params via ModelEMA callback.
    client.load_state_dict(state_dict, model_only=True)
    client.model.eval()

    # --- dataloader over the training DB -------------------------------------
    bio_cfg = BioMolData.BioMolConfig(
        crop_config=data_cfg.crop,
        msa_config=data_cfg.msa,
        DB_config=data_cfg.train_db,
        sampler_config=data_cfg.sampler,
        tokenizer_config=data_cfg.tokenizer,
    )
    dataset = BioMolData(bio_cfg)
    dataset.set_epoch(0)
    dataloader = dataset.create_ddp_dataloader(
        world_size=1,
        rank=0,
        seed=seed,
        drop_last=False,
        batch_size=1,
        num_workers=0,
        shuffle=True,
    )

    client.logger.info("Start EDM sampling")
    done = 0
    for raw_batch in dataloader:
        if done >= num_targets:
            break
        batch = raw_batch.to(device=client.device)
        name = str(batch.name[0])
        client.logger.info(
            "target %d/%d %s | n_tokens=%d n_atoms=%d n_msa=%d",
            done + 1, num_targets, name,
            batch.token_length, batch.atom_length, batch.msa_count,
        )

        # Run trunk once, then sample n_samples diffusion trajectories.
        wrapper, batch = client.prepare(batch)
        torch.manual_seed(seed * 100003 + done * 1009)
        output = client.sample(
            wrapper, batch, n_samples=n_samples, timesteps=timesteps,
        )

        # Ground truth once.
        batch_to_cif(batch, None, cif_dir / f"{name}_gt.cif")

        gt_pos = batch.structure.atom_pos[0]
        gt_mask = batch.structure.atom_mask[0]
        best_rmsd, best_lddt, best_k = float("inf"), 0.0, 0
        for k in range(n_samples):
            pred_k = output.atom_pos_pred[k:k + 1]
            batch_to_cif(batch, pred_k, cif_dir / f"{name}_pred_{k}.cif")
            rmsd = float(metrics.cal_aligned_rmsd(output.atom_pos_pred[k], gt_pos, gt_mask))
            lddt = float(metrics.cal_atom_lddt(output.atom_pos_pred[k], gt_pos, gt_mask))
            client.logger.info("  sample %d: rmsd=%.3f lddt=%.4f", k, rmsd, lddt)
            if rmsd < best_rmsd:
                best_rmsd, best_lddt, best_k = rmsd, lddt, k

        client.logger.info(
            "target %s DONE | best(sample=%d) rmsd=%.3f lddt=%.4f",
            name, best_k, best_rmsd, best_lddt,
        )
        done += 1

    client.logger.info("Sampling complete. %d targets -> %s", done, run_sub_dir)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
