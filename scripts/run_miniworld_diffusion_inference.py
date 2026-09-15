"""Inference / validation for phase 2 (frozen mini-SWA trunk + EDM diffusion head).

Runs the phase 1b model's EDM diffusion solver (:meth:`diffusion.Client.inference`)
over a dataset and, for each item, samples ``--num-samples`` structures, scores
best-of-N RMSD / lDDT against the ground truth, and writes predicted + GT CIFs.

This is the EDM/phase 2 analogue of ``run_miniworld_inference.py validate`` (which
targets the XPred-decoupled *miniworld* model); phase 2 uses the AF3 EDM solver, so
it needs its own entrypoint. It parses the SAME top-level phase 2 config used for
training (e.g. ``configs/miniworld/phase2b_diffusion.yaml``) — the evaluation set
is ``data.train_db``; point it at a held-out DB via a hydra override, e.g.::

    torchrun --standalone --nproc_per_node=1 \
        scripts/run_miniworld_diffusion_inference.py validate \
        --config configs/miniworld/phase2b_diffusion.yaml \
        --ckpt   runs/v1.0.0/phase2b/large_diffusion_L768/.../checkpoints/last.pt \
        --num-items 50 --num-samples 5 --timesteps 200

EMA weights are used by default (``--use-ema``): the phase 2 checkpoint stores the
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
from miniworld.models.diffusion import Client, Model

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
    """Data config — mirrors the phase 2 training script so the same YAML parses."""

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
    """Top-level config — identical shape to the phase 2 training config."""

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
    default=Path("outputs/phase2_validation"),
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
    # the full phase 2 state directly (trunk + diffusion), so disable it.
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

    # Load the phase 2 checkpoint (trunk + diffusion). strict=False tolerates the
    # unused distogram_head / any harmless key drift; use_ema swaps in the EMA
    # diffusion weights via the ModelEMA callback's on_load_state_dict.
    state_dict = torch.load(ckpt, map_location="cpu")
    client.load_state_dict(state_dict, model_only=True, strict=False)
    client.logger.info(
        "Loaded phase 2 ckpt %s (epoch=%d, step=%d) use_ema=%s",
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
        "Start phase 2 EDM inference: num_items=%d num_samples=%d timesteps=%d",
        num_items, num_samples, timesteps,
    )

    all_best_rmsd: list[float] = []
    all_best_lddt: list[float] = []
    for batch_idx, raw_batch in enumerate(dataloader):
        if batch_idx >= num_items:
            break
        batch = raw_batch.to(device=client.device)
        name = str(batch.name[0])

        # best-of-N: the client draws num_samples independent diffusion samples
        # from one B=1 trunk pass (the trimul cute kernel is B=1 only).
        torch.manual_seed(seed * 100003 + batch_idx * 1009)
        output = client.inference(batch, timesteps=timesteps, n_samples=num_samples)

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
# FoldBench inference: run the phase 2 model over FoldBench target specs.
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


# Shape ladders for inference padding. Every distinct (token, atom) shape pays a
# fresh Triton JIT + autotune grid search in each worker process -- measured at ~7
# minutes, which on a 1,522-target sweep is 72% of the total wall time (a 28-token
# target took 10.5 min, a 30-token one that happened to reuse a shape took 2.8).
# Rounding up to a ladder collapses ~740 distinct shapes to ~23, so each worker pays
# the compile cost ~23 times instead of once per target. The wasted compute from
# over-padding is real but small next to that: the worst case is a target just above
# a rung, and the rungs are tight where targets are dense.
# Ladder chosen against the measured FoldBench shape distribution
# (submits/phase2b/foldbench_shapes.csv, produced by scripts/foldbench_scan_shapes.py).
# A finer ladder is not better: it cuts wasted compute only marginally while adding a
# compile. Over the 1,522 targets, finer (14x14 rungs) gives 46 buckets at 1.64x mean
# token^2 padding; this one gives 29 buckets at 1.74x -- 17 fewer compiles, ~4 min each
# per worker, for ~6% more arithmetic.
_TOKEN_BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 2560)
_ATOM_BUCKETS = (1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152)
# MSA depth and template count are shape dimensions too, and they vary a lot more than
# you would guess: a DNA-only target has msa_depth 1, an RNA one 192, a protein one
# 2048, and template_number is 1 or 4. Leaving them unbucketed kept re-triggering the
# autotune even for targets that shared a (token, atom) bucket -- measured: with a
# multi-rung MSA ladder, targets at depth 1 / 192 / 2048 still cost 4.1 min each,
# while same-depth ones cost 0.12. So collapse both to ONE value. Padding a depth-1
# MSA up to 2048 is nearly free in absolute terms (those targets are tiny, and every
# target big enough for the MSA module to matter already has the full 2048 rows).
_MSA_BUCKETS = (2048,)
_N_TEMPLATE = 4


def _bucket(value: int, ladder: tuple[int, ...], mult: int = 8) -> int:
    """Round ``value`` up to the next rung, or to a multiple of ``mult`` beyond it."""
    for rung in ladder:
        if value <= rung:
            return rung
    return _ceil_to_multiple(value, mult)


def _pad_inference_batch(batch: Batch, mult: int = 8, *, bucket: bool = True) -> Batch:
    """Pad msa/token/atom dims, to a shape ladder by default.

    The engine's fused GEMM kernels (quack ``gemm_act``) require the sequence stride
    divisible by 8; FoldBench targets are arbitrary sizes (training crops were bucket-
    aligned). Pads via collating with a bucket-sized ``Batch.empty`` dummy (masks mark
    the padding); the caller slices outputs back to the real atom count.

    ``bucket=False`` falls back to the old behaviour (multiple of ``mult``), which
    wastes no compute but gives almost every target its own shape.
    """
    if bucket:
        bt = _bucket(int(batch.token_length), _TOKEN_BUCKETS, mult)
        ba = _bucket(int(batch.atom_length), _ATOM_BUCKETS, mult)
    else:
        bt = _ceil_to_multiple(int(batch.token_length), mult)
        ba = _ceil_to_multiple(int(batch.atom_length), mult)
    if bucket:
        bm = _bucket(int(batch.msa_depth), _MSA_BUCKETS, mult)
        nt = max(_N_TEMPLATE, int(batch.template_number))
    else:
        bm = _ceil_to_multiple(int(batch.msa_depth), mult)
        nt = int(batch.template_number)
    if (bt == int(batch.token_length) and ba == int(batch.atom_length)
            and bm == int(batch.msa_depth) and nt == int(batch.template_number)):
        return batch
    dummy = _Batch.empty(
        n_temp=nt, msa_depth=bm, n_tokens=bt, n_atoms=ba,
    )
    _nullify_like(batch, dummy)
    return _Batch.collate_fn([batch, dummy])[0 : batch.batch_size]


def _diffusion_client_from_config(
    config: Path,
    ckpt: Path,
    seed: int,
    use_ema: bool,
    do_compile: bool,
    fabric: Fabric,
    overrides: list[str],
) -> Client:
    """Build a phase 2 Client, load the checkpoint (model-only, EMA-aware)."""
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
              default=Path("runs/foldbench/phase2"), show_default=True,
              help="Prediction output dir (default: repo-relative runs/foldbench/phase2, gitignored).")
@click.option("--timesteps", default="200", show_default=True,
              help="Denoising steps. Accepts a comma-separated list (e.g. '20,50,100,200') to "
                   "sweep in ONE process: the shape-dependent kernel compile is paid once "
                   "instead of once per value, and each value writes to "
                   "<output-dir>/steps<N>/<target>/. A single value writes to "
                   "<output-dir>/<target>/ as usual.")
@click.option("--n-samples", type=int, default=5, show_default=True,
              help="Diffusion samples drawn per seed (AF3/FoldBench: 5).")
@click.option("--n-seeds", type=int, default=5, show_default=True,
              help="Independent model seeds per target (AF3/FoldBench: 5). Each seed rebuilds\n                   the input features (reference-conformer SE3 randomisation, MSA subsample,\n                   template sampling) and runs its own trunk pass, then draws --n-samples\n                   diffusion samples from it. Total predictions = n_seeds * n_samples.")
@click.option("--num-recycles", type=int, default=10, show_default=True,
              help="Trunk recycle depth at inference (overrides model n_recycle_max). 10 is what "
                   "FoldBench runs the published methods at (Protenix N_cycle=10, AF3 "
                   "--num_recycles=10). Training samples n_recycle from 1..4, so anything here "
                   "is an extrapolation; 20 is a bigger one and was never validated.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--use-ema/--no-ema", default=True, show_default=True)
@click.option("--compile/--no-compile", "do_compile", default=False, show_default=True)
@click.option("--max-msa-depth", type=int, default=2048, show_default=True,
              help="Max MSA depth (match training: H100_default uses 2048).")
@click.option("--sample-batch/--no-sample-batch", default=False, show_default=True,
              help="Draw all --n-samples diffusion trajectories in one batched solver run "
                   "rather than looping one at a time. Off by default until measured.")
@click.option("--msa-subsample/--no-msa-subsample", default=True, show_default=True,
              help="Per seed, draw WHICH max-msa-depth rows to keep at random (query always "
                   "kept) instead of always taking the top rows. Depth is unchanged. Without "
                   "it every seed gets the identical alignment, so a 5-seed ensemble only "
                   "varies the reference conformers and the diffusion noise.")
@click.option("--no-pairing-msa/--pairing-msa", "no_pairing_msa", default=True,
              show_default=True,
              help="Force no_pairing MSA to match training (pairing_mode: no_pairing). "
                   "The InferenceSpec default is 'mixed' (species pairing on).")
@click.option("--missing-policy", type=click.Choice(["query", "gap"]), default="query", show_default=True)
@click.argument("overrides", type=str, nargs=-1)
def foldbench(  # noqa: PLR0913
    config: Path,
    ckpt: Path,
    data_path: Path | None,
    index_file: Path | None,
    inputs_root: Path,
    output_dir: Path,
    timesteps: str,
    n_samples: int,
    n_seeds: int,
    num_recycles: int,
    seed: int,
    use_ema: bool,
    do_compile: bool,
    max_msa_depth: int,
    sample_batch: bool,
    msa_subsample: bool,
    no_pairing_msa: bool,
    missing_policy: str,
    overrides: tuple[str, ...],
) -> None:
    """Run the phase 2 model over FoldBench targets -> predicted CIFs.

    One GPU worker walks its shard (env ``SHARD``/``N_SHARDS``, stride) of the
    index, holding the model in memory. Finished targets (an existing CIF) are
    skipped, so resubmitting the same shard resumes. Predictions land in
    ``<output_dir>/<target>/<target>_seed{s}_sample{k}_pred.cif`` — feed the dir to
    ``cal_foldbench.py`` then FoldBench ``evaluate.py``.
    """
    fabric = _fabric_from_torchrun()
    fabric.launch()
    fabric.seed_everything(seed)

    step_list = [int(x) for x in str(timesteps).split(",") if x.strip()]
    if not step_list:
        msg = "--timesteps must name at least one value"
        raise click.UsageError(msg)
    # One value keeps the flat layout; a sweep gets a steps<N>/ level so the runs stay
    # separable and the "already done" check below still works per step count.
    step_dirs = {n: (output_dir if len(step_list) == 1 else output_dir / f"steps{n}")
                 for n in step_list}

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
    client = _diffusion_client_from_config(
        config, ckpt, seed, use_ema, do_compile, fabric, list(overrides),
    )
    # MW_COMPILE_TIMEOUT=1: install the fork+SIGKILL compile guard so uncached-shape
    # autotune bounds each "monster" config (e.g. transition_b2b full grid) to 60s
    # instead of hanging indefinitely. FoldBench targets are arbitrary sizes with no
    # prebuilt cache, so without this the first uncached op stalls the run.
    if os.getenv("MW_COMPILE_TIMEOUT", "0").strip().lower() in {"1", "true", "yes", "on"}:
        from miniworld_engine.autotune import capture

        capture.install()
        client.logger.info("[capture] compile-timeout guard installed (bounds monster autotune)")
    client.logger.info(
        "FoldBench phase 2 inference: %d targets | %d seeds x %d samples = %d preds/target "
        "| recycles=%d timesteps=%s msa=%d(%s) ema=%s",
        len(targets), n_seeds, n_samples, n_seeds * n_samples, num_recycles,
        ",".join(str(n) for n in step_list),
        max_msa_depth, "subsampled" if msa_subsample else "top-N", use_ema,
    )

    for i, (tid, dpath) in enumerate(targets):
        if not dpath.exists():
            client.logger.warning("skip %s: no data.yaml at %s", tid, dpath)
            continue
        last = f"{tid}_seed{n_seeds - 1}_sample{n_samples - 1}_pred.cif"
        if all((step_dirs[n] / tid / last).exists() for n in step_list):
            client.logger.info("[%d/%d] %s: already done, skip", i + 1, len(targets), tid)
            continue
        try:
            spec = InferenceSpec.from_yaml(dpath)
            if no_pairing_msa and not spec.no_pairing_msa:
                # Match training (pairing_mode: no_pairing). The InferenceSpec
                # default resolves to 'mixed' (species pairing on) — override so
                # inference stacks all homologs block-diagonally like training.
                spec = spec.model_copy(update={"no_pairing_msa": True})
            for n in step_list:
                (step_dirs[n] / tid).mkdir(parents=True, exist_ok=True)
            # AF3 / FoldBench protocol: n_seeds INDEPENDENT model runs, each drawing
            # n_samples diffusion samples -> n_seeds * n_samples predictions per target.
            # A seed is not just a diffusion noise seed: build_inference_batch's rng
            # drives the per-residue SE3 randomisation of the reference conformers and
            # the template sampling, so every seed feeds the trunk a different input and
            # gets its own recycled representation. Drawing 25 samples from ONE trunk
            # pass instead only varies the diffusion noise.
            # The MSA joins that list only with --msa-subsample: in no_pairing mode
            # ComplexMSA.sample otherwise takes the first max_msa_depth rows
            # deterministically and every seed sees the identical alignment.
            n_done = 0
            for s in range(n_seeds):
                seed_s = seed + s
                batch = build_inference_batch(
                    spec, max_msa_depth=max_msa_depth, missing_policy=missing_policy,
                    seed=seed_s, msa_subsample=msa_subsample,
                )
                name = str(batch.name[0])
                orig_atom = int(batch.atom_length)
                # Pad msa/token/atom to a multiple of 8: the engine's fused GEMM kernels
                # (quack gemm_act) require the sequence stride divisible by 8. FoldBench
                # targets are arbitrary sizes (training crops were bucket-aligned). Padding
                # positions are masked; slice the output back to the real atom count for the
                # CIF so no padding atoms are written.
                padded = _pad_inference_batch(batch, 8)
                # Keep the batch at B=1 (the trunk trimul cute kernel is B=1 only); the
                # client runs the trunk once and draws n_samples independent diffusion
                # samples internally.
                # Seed BOTH generators. The per-step CentreRandomAugmentation draws
                # its rotation with scipy's Rotation.random, which reads numpy's
                # global RNG, not torch's -- seeding only torch leaves the rotations
                # running off whatever numpy state the process happens to be in, so
                # a rerun (or a different shard split) would not reproduce.
                for n_steps in step_list:
                    # Re-seed per (seed, step count) so every step count starts from the
                    # same noise for that seed -- the sweep is then a paired comparison
                    # of step count, not of luck.
                    torch.manual_seed(seed_s * 100003 + i * 1009)
                    np.random.seed((seed_s * 100003 + i * 1009) % (2**32 - 1))  # noqa: NPY002
                    try:
                        output = client.inference(
                            padded, timesteps=n_steps, n_samples=n_samples,
                            n_recycle=num_recycles, sample_batch=sample_batch,
                        )
                    except torch.cuda.OutOfMemoryError:
                        # Batched sampling multiplies the atom-DiT activations by
                        # n_samples, so the biggest buckets can fall over where the
                        # sequential loop fits. Drop to the loop for this target rather
                        # than losing it; the result is identical, only slower.
                        if not sample_batch:
                            raise
                        client.logger.warning(
                            "%s seed %d steps %d: OOM with sample_batch, "
                            "retrying one sample at a time", tid, s, n_steps,
                        )
                        # Deliberately no torch.cuda.empty_cache() here, the usual
                        # remedy: it corrupts the engine's cute kernels, which would
                        # cost the whole shard instead of this one target. See
                        # docs/known-issues.md. The headroom comes from the retry
                        # itself -- sequential sampling needs 1/n_samples of the
                        # activations -- and from expandable_segments in the launcher.
                        torch.manual_seed(seed_s * 100003 + i * 1009)
                        np.random.seed((seed_s * 100003 + i * 1009) % (2**32 - 1))  # noqa: NPY002
                        output = client.inference(
                            padded, timesteps=n_steps, n_samples=n_samples,
                            n_recycle=num_recycles, sample_batch=False,
                        )
                    for k in range(n_samples):
                        pred = output.atom_pos_pred[k : k + 1, :orig_atom]
                        batch_to_cif(
                            batch, pred,
                            step_dirs[n_steps] / tid / f"{name}_seed{s}_sample{k}_pred.cif",
                            chain_names=spec.chain_names,
                        )
                        n_done += 1
                    del output
                shape = (int(batch.token_length), orig_atom, int(padded.atom_length))
                del padded, batch
            client.logger.info(
                "[%d/%d] %s: OK tok=%d atom=%d (pad->%d) -> %d CIFs (%dx%d, steps=%s)",
                i + 1, len(targets), tid, shape[0], shape[1], shape[2],
                n_done, n_seeds, n_samples, ",".join(str(n) for n in step_list),
            )
        except Exception as e:  # noqa: BLE001 — one bad target must not kill the shard
            client.logger.exception("[%d/%d] %s: FAILED (%s)", i + 1, len(targets), tid, type(e).__name__)

    client.logger.info("FoldBench inference complete. Results in %s", output_dir)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
