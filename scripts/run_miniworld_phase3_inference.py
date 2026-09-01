"""Inference / validation for phase3 (frozen mini-SWA trunk + EDM diffusion head).

Runs the phase3 model's EDM diffusion solver (:meth:`phase3.Client.inference`)
over a dataset and, for each item, samples ``--num-samples`` structures, scores
best-of-N RMSD / lDDT against the ground truth, and writes predicted + GT CIFs.

This is the EDM/phase3 analogue of ``run_miniworld_inference.py validate`` (which
targets the XPred-decoupled *miniworld* model); phase3 uses the AF3 EDM solver, so
it needs its own entrypoint. It parses the SAME top-level phase3 config used for
training (e.g. ``configs/miniworld/large_H100_phase3.yaml``) — the evaluation set
is ``data.train_db``; point it at a held-out DB via a hydra override, e.g.::

    torchrun --standalone --nproc_per_node=1 \
        scripts/run_miniworld_phase3_inference.py validate \
        --config configs/miniworld/large_H100_phase3.yaml \
        --ckpt   logs/phase3/large_H100_diffusion_hlr/.../checkpoints/last.pt \
        --num-items 50 --num-samples 5 --timesteps 200

EMA weights are used by default (``--use-ema``): the phase3 checkpoint stores the
diffusion-head EMA; loading swaps it in automatically (frozen trunk stays as saved).
"""

from __future__ import annotations

import logging
import dataclasses
import os
import time
from pathlib import Path
from typing import Annotated, Union

import click
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf
from pydantic import BaseModel, Discriminator, Tag

from miniworld.configs import (
    BioMolDBConfig,
    CropConfig,
    EDMDiffuserConfig,
    MSAConfig,
    SamplerConfig,
    TokenizerConfig,
)
from miniworld.data.dataloader import BioMolDBV2Config
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.features.batch import Batch
from miniworld.data.dataloader.collate import _ceil_to_multiple
from miniworld.data.features import Batch as _Batch
from miniworld.data.inference import InferenceSpec, build_inference_batch
from miniworld.data.io.to_cif import batch_to_cif
from miniworld.loss import metrics
from miniworld.models.phase3 import Client, Model

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
    """Data config — mirrors the phase3 training script so the same YAML parses."""

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
    """Top-level config — identical shape to the phase3 training config."""

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


def _best_of_n(
    output_atom_pos: torch.Tensor,  # (N_str, L, 3)
    gt_atom_pos: torch.Tensor,      # (L, 3)
    atom_mask: torch.Tensor,        # (L,)
) -> tuple[float, float, int, int, list[float], list[float]]:
    """Best (min RMSD / max lDDT) over the N sampled structures.

    Returns (best_rmsd, best_lddt, argmin_rmsd, argmax_lddt, all_rmsd, all_lddt).
    """
    rmsds = [
        float(metrics.cal_aligned_rmsd(output_atom_pos[i], gt_atom_pos, atom_mask))
        for i in range(output_atom_pos.shape[0])
    ]
    lddts = [
        float(metrics.cal_atom_lddt(output_atom_pos[i], gt_atom_pos, atom_mask))
        for i in range(output_atom_pos.shape[0])
    ]
    i_rmsd = int(np.argmin(rmsds))
    i_lddt = int(np.argmax(lddts))
    return rmsds[i_rmsd], lddts[i_lddt], i_rmsd, i_lddt, rmsds, lddts


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
    default=Path("outputs/phase3_validation"),
    show_default=True,
    help="Root dir; results land in <output_dir>/<YYYY-MM-DD>/<HHMMSS>[_<job>]/.",
)
@click.option("--job-name", type=str)
@click.option("--num-items", type=int, default=50, show_default=True,
              help="Number of dataset items to evaluate (per rank).")
@click.option("--num-samples", type=int, default=5, show_default=True,
              help="Diffusion samples per item (best-of-N scoring).")
@click.option("--timesteps", type=int, default=200, show_default=True,
              help="EDM solver reverse steps.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--use-ema/--no-ema", default=True, show_default=True,
              help="Use the diffusion-head EMA weights from the checkpoint.")
@click.option("--compile/--no-compile", "do_compile", default=False,
              show_default=True, help="torch.compile the model (dynamic=False).")
@click.option("--save-all/--save-best", default=False, show_default=True,
              help="Save every sample's CIF, or only the best-RMSD one.")
@click.argument("overrides", type=str, nargs=-1)
def validate(  # noqa: PLR0913, PLR0915
    config: Path,
    ckpt: Path,
    output_dir: Path,
    job_name: str | None,
    num_items: int,
    num_samples: int,
    timesteps: int,
    seed: int,
    use_ema: bool,
    do_compile: bool,
    save_all: bool,
    overrides: tuple[str, ...],
) -> None:
    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name, overrides=list(overrides))
    cfg = Config.model_validate(cfg)

    fabric = _fabric_from_torchrun()
    fabric.launch()
    fabric.seed_everything(seed)

    date_dir = output_dir / time.strftime("%Y-%m-%d")
    run_name = time.strftime("%H%M%S")
    if job_name:
        run_name += f"_{job_name}"
    run_sub_dir = date_dir / run_name
    run_sub_dir.mkdir(parents=True, exist_ok=True)

    # Build the client with the training config, but force eval knobs.
    cfg.train.use_ema = use_ema
    cfg.train.seed = seed
    # param_policy is a training-time load/freeze mechanism; for inference we load
    # the full phase3 state directly (trunk + diffusion), so disable it.
    cfg.train.param_policy.enabled = False
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
        fh = logging.FileHandler(run_sub_dir / "validation.log")
        fh.setFormatter(formatter)
        client.logger.addHandler(fh)

    if do_compile:
        torch._dynamo.config.cache_size_limit = 128  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001
        client.model.compile(dynamic=False)
        client.logger.info("Compiled model (dynamic=False)")

    if fabric.is_global_zero:
        OmegaConf.save(
            OmegaConf.create(cfg.model_dump(mode="json")),
            run_sub_dir / "config.yaml",
        )

    client.setup(fabric=fabric)

    # Load the phase3 checkpoint (trunk + diffusion). strict=False tolerates the
    # unused distogram_head / any harmless key drift; use_ema swaps in the EMA
    # diffusion weights via the ModelEMA callback's on_load_state_dict.
    state_dict = torch.load(ckpt, map_location="cpu")
    client.load_state_dict(state_dict, model_only=True, strict=False)
    client.logger.info(
        "Loaded phase3 ckpt %s (epoch=%d, step=%d) use_ema=%s",
        ckpt, client.epoch, client.global_step, use_ema,
    )

    data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler,
        tokenizer_config=cfg.data.tokenizer,
    )
    dataset = BioMolData(data_config)
    dataloader = dataset.create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.global_rank,
        seed=seed,
        drop_last=False,
        batch_size=1,
        num_workers=0,
        num_samples_per_rank=num_items,
        shuffle=False,
        bucket_msa_multiple=cfg.train.bucket_msa_multiple,
        bucket_token_multiple=cfg.train.bucket_token_multiple,
        bucket_atom_multiple=cfg.train.bucket_atom_multiple,
    )
    dataset.set_epoch(0)

    cif_dir = run_sub_dir / "structures"
    cif_dir.mkdir(parents=True, exist_ok=True)

    client.model.eval()
    client.logger.info(
        "Start phase3 EDM inference: num_items=%d num_samples=%d timesteps=%d",
        num_items, num_samples, timesteps,
    )

    all_best_rmsd: list[float] = []
    all_best_lddt: list[float] = []
    for batch_idx, raw_batch in enumerate(dataloader):
        if batch_idx >= num_items:
            break
        batch = raw_batch.to(device=client.device)
        name = str(batch.name[0])

        # best-of-N: replicate the single item into num_samples independent draws.
        sample_batch = batch.duplicate(num_samples)
        torch.manual_seed(seed * 100003 + batch_idx * 1009)
        output = client.inference(sample_batch, timesteps=timesteps)

        gt = batch.structure.atom_pos[0]
        atom_mask = batch.structure.atom_mask[0]
        best_rmsd, best_lddt, i_rmsd, i_lddt, rmsds, lddts = _best_of_n(
            output.atom_pos_pred, gt, atom_mask,
        )
        all_best_rmsd.append(best_rmsd)
        all_best_lddt.append(best_lddt)
        client.logger.info(
            "rank=%d item=%d %s | n_tokens=%d n_atoms=%d | "
            "best_rmsd=%.4f (s%d) best_lddt=%.4f (s%d) | "
            "rmsd=%s lddt=%s | mem=%.2fGB",
            fabric.global_rank, batch_idx, name,
            batch.token_length, batch.atom_length,
            best_rmsd, i_rmsd, best_lddt, i_lddt,
            ",".join(f"{r:.2f}" for r in rmsds),
            ",".join(f"{l:.3f}" for l in lddts),
            torch.cuda.max_memory_allocated() / 1024**3,
        )

        # Save GT + predicted structures.
        batch_to_cif(batch, None, cif_dir / f"{name}_gt.cif")
        if save_all:
            for s in range(num_samples):
                pred_s = output.atom_pos_pred[s : s + 1]
                batch_to_cif(batch, pred_s, cif_dir / f"{name}_s{s:02d}_pred.cif")
        else:
            best_pred = output.atom_pos_pred[i_rmsd : i_rmsd + 1]
            batch_to_cif(batch, best_pred, cif_dir / f"{name}_best_pred.cif")

    if all_best_rmsd:
        client.logger.info(
            "DONE rank=%d | items=%d | mean_best_rmsd=%.4f median_best_rmsd=%.4f | "
            "mean_best_lddt=%.4f median_best_lddt=%.4f",
            fabric.global_rank, len(all_best_rmsd),
            float(np.mean(all_best_rmsd)), float(np.median(all_best_rmsd)),
            float(np.mean(all_best_lddt)), float(np.median(all_best_lddt)),
        )
    client.logger.info("Validation complete. Results saved to %s", run_sub_dir)


# ---------------------------------------------------------------------------
# FoldBench inference: run the phase3 model over FoldBench target specs.
# ---------------------------------------------------------------------------
def _nullify_like(real: object, dummy: object) -> None:
    """Set ``dummy.<field> = None`` wherever ``real.<field>`` is None (recursive).

    ``Batch.empty`` populates every optional field, but an inference batch leaves
    some as None (e.g. ``atom_is_rep`` — no GT structure). Collating the two then
    fails on the None-vs-Tensor mismatch; matching the dummy's None-ness first lets
    the pad-collate go through.
    """
    for f in dataclasses.fields(real):
        rv = getattr(real, f.name, None)
        dv = getattr(dummy, f.name, None)
        if rv is None and dv is not None:
            object.__setattr__(dummy, f.name, None)
        elif dataclasses.is_dataclass(rv) and dataclasses.is_dataclass(dv):
            _nullify_like(rv, dv)


def _pad_inference_batch(batch: Batch, mult: int = 8) -> Batch:
    """Pad msa/token/atom dims up to a multiple of ``mult``.

    The engine's fused GEMM kernels (quack ``gemm_act``) require the sequence stride
    divisible by 8; FoldBench targets are arbitrary sizes (training crops were bucket-
    aligned). Pads via collating with a bucket-sized ``Batch.empty`` dummy (masks mark
    the padding); the caller slices outputs back to the real atom count.
    """
    bt = _ceil_to_multiple(int(batch.token_length), mult)
    ba = _ceil_to_multiple(int(batch.atom_length), mult)
    bm = _ceil_to_multiple(int(batch.msa_depth), mult)
    if bt == int(batch.token_length) and ba == int(batch.atom_length) and bm == int(batch.msa_depth):
        return batch
    dummy = _Batch.empty(
        n_temp=batch.template_number, msa_depth=bm, n_tokens=bt, n_atoms=ba,
    )
    _nullify_like(batch, dummy)
    return _Batch.collate_fn([batch, dummy])[0 : batch.batch_size]


def _phase3_client_from_config(
    config: Path,
    ckpt: Path,
    seed: int,
    use_ema: bool,
    do_compile: bool,
    fabric: Fabric,
    overrides: list[str],
) -> Client:
    """Build a phase3 Client, load the checkpoint (model-only, EMA-aware)."""
    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name, overrides=overrides)
    cfg = Config.model_validate(cfg)
    cfg.train.use_ema = use_ema
    cfg.train.seed = seed
    cfg.train.param_policy.enabled = False
    client = Client(
        Client.Config(
            train=cfg.train, model=cfg.model, diffuser=cfg.diffuser, loss=cfg.loss,
        ),
    )
    if do_compile:
        torch._dynamo.config.cache_size_limit = 128  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001
        client.model.compile(dynamic=False)
    client.setup(fabric=fabric)
    state_dict = torch.load(ckpt, map_location="cpu")
    client.load_state_dict(state_dict, model_only=True, strict=False)
    client.model.eval()
    return client


@cli.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--data", "data_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Single FoldBench target data.yaml (InferenceSpec).")
@click.option("--index", "index_file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Index file of target ids (one per line) for batch mode.")
@click.option("--inputs-root", type=click.Path(path_type=Path),
              default=Path("/home/psk6950/data/foldbench/inputs"), show_default=True,
              help="Root holding <target>/data.yaml (index mode).")
@click.option("--output-dir", type=click.Path(path_type=Path),
              default=Path("runs/foldbench/phase3"), show_default=True,
              help="Prediction output dir (default: repo-relative runs/foldbench/phase3, gitignored).")
@click.option("--timesteps", type=int, default=200, show_default=True)
@click.option("--n-samples", type=int, default=5, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--use-ema/--no-ema", default=True, show_default=True)
@click.option("--compile/--no-compile", "do_compile", default=False, show_default=True)
@click.option("--max-msa-depth", type=int, default=256, show_default=True)
@click.option("--missing-policy", type=click.Choice(["query", "gap"]), default="query", show_default=True)
@click.argument("overrides", type=str, nargs=-1)
def foldbench(  # noqa: PLR0913
    config: Path,
    ckpt: Path,
    data_path: Path | None,
    index_file: Path | None,
    inputs_root: Path,
    output_dir: Path,
    timesteps: int,
    n_samples: int,
    seed: int,
    use_ema: bool,
    do_compile: bool,
    max_msa_depth: int,
    missing_policy: str,
    overrides: tuple[str, ...],
) -> None:
    """Run the phase3 model over FoldBench targets -> predicted CIFs.

    One GPU worker walks its shard (env ``SHARD``/``N_SHARDS``, stride) of the
    index, holding the model in memory. Finished targets (an existing CIF) are
    skipped, so resubmitting the same shard resumes. Predictions land in
    ``<output_dir>/<target>/<target>_s{k}_pred.cif`` — feed the dir to
    ``cal_foldbench.py`` then FoldBench ``evaluate.py``.
    """
    fabric = _fabric_from_torchrun()
    fabric.launch()
    fabric.seed_everything(seed)

    # Resolve the target list.
    if data_path is not None:
        targets = [(data_path.parent.name, data_path)]
    elif index_file is not None:
        ids = [ln.strip() for ln in index_file.read_text().splitlines() if ln.strip()]
        shard = int(os.environ.get("SHARD", "0"))
        n_shards = int(os.environ.get("N_SHARDS", "1"))
        ids = ids[shard::n_shards]  # stride shard
        targets = [(tid, inputs_root / tid / "data.yaml") for tid in ids]
    else:
        msg = "Pass --data <data.yaml> or --index <index_file>."
        raise click.UsageError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    client = _phase3_client_from_config(
        config, ckpt, seed, use_ema, do_compile, fabric, list(overrides),
    )
    client.logger.info(
        "FoldBench phase3 inference: %d targets | n_samples=%d timesteps=%d ema=%s",
        len(targets), n_samples, timesteps, use_ema,
    )

    for i, (tid, dpath) in enumerate(targets):
        if not dpath.exists():
            client.logger.warning("skip %s: no data.yaml at %s", tid, dpath)
            continue
        out_sub = output_dir / tid
        done = out_sub / f"{tid}_s{n_samples - 1}_pred.cif"
        if done.exists():
            client.logger.info("[%d/%d] %s: already done, skip", i + 1, len(targets), tid)
            continue
        try:
            spec = InferenceSpec.from_yaml(dpath)
            batch = build_inference_batch(
                spec, max_msa_depth=max_msa_depth, missing_policy=missing_policy, seed=seed,
            )
            name = str(batch.name[0])
            orig_atom = int(batch.atom_length)
            # Pad msa/token/atom to a multiple of 8: the engine's fused GEMM kernels
            # (quack gemm_act) require the sequence stride divisible by 8. FoldBench
            # targets are arbitrary sizes (training crops were bucket-aligned). Padding
            # positions are masked; slice the output back to the real atom count for the
            # CIF so no padding atoms are written.
            padded = _pad_inference_batch(batch, 8)
            sample_batch = padded.duplicate(n_samples)
            torch.manual_seed(seed * 100003 + i * 1009)
            output = client.inference(sample_batch, timesteps=timesteps)
            out_sub.mkdir(parents=True, exist_ok=True)
            for k in range(n_samples):
                pred = output.atom_pos_pred[k : k + 1, :orig_atom]
                batch_to_cif(batch, pred, out_sub / f"{name}_s{k}_pred.cif")
            client.logger.info(
                "[%d/%d] %s: OK tok=%d atom=%d (pad->%d) -> %d CIFs",
                i + 1, len(targets), tid, int(batch.token_length), orig_atom,
                int(padded.atom_length), n_samples,
            )
        except Exception as e:  # noqa: BLE001 — one bad target must not kill the shard
            client.logger.exception("[%d/%d] %s: FAILED (%s)", i + 1, len(targets), tid, type(e).__name__)

    client.logger.info("FoldBench inference complete. Results in %s", output_dir)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
