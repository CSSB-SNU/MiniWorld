"""Validation-only sweep without EMA parameters for atom_token_fingerprint variant.

Based on scripts/run_af3like_rev_no_ema_valid.py (the atom_token version) and
scripts/run_af3like_embedding.py (EmbeddingClient + af3_like_embedding.Model).
For each checkpoint in --ckpt-dir(s), loads only the non-EMA model weights
(cfg.train.use_ema=False so the ModelEMA callback is not attached and the
saved EMA parameters are ignored), runs one validation pass, and appends a
JSON record (epoch -> metrics) to --output.

Note on use_qk_norm: fingerprint training switched use_qk_norm: false -> true
mid-run (resume570). Invoke this script once per (config, ckpt-dirs) pair so
the model arch matches the checkpoint, then append both runs to the same
--output (epoch dedup handles overlap).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore", message=".*torch.jit.script_method.*", category=DeprecationWarning,
)

import click
import torch
from lightning import Fabric
from omegaconf import OmegaConf
from pydantic import BaseModel
from team_gm.utils.script_utils import MetricsAggregator

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
from miniworld.models import EmbeddingClient as Client
from miniworld.models.af3_like_embedding import Model as Model

torch.set_float32_matmul_precision("medium")
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
    world_size = os.environ.get("WORLD_SIZE")
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if world_size is None or local_world_size is None:
        return Fabric(devices=1, **fabric_kwargs)
    devices = int(local_world_size)
    num_nodes = int(world_size) // devices
    return Fabric(devices=devices, num_nodes=num_nodes, **fabric_kwargs)


CKPT_RE = re.compile(r"epoch=(\d{4})\.pt$")


def _strip_wrapper_prefixes(sd: dict) -> dict:
    out: dict = {}
    for k, v in sd.items():
        while True:
            if k.startswith("_forward_module."):
                k = k[len("_forward_module."):]
            elif k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
            elif k.startswith("module."):
                k = k[len("module."):]
            else:
                break
        out[k] = v
    return out


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="saved config.yaml from training run dir",
)
@click.option(
    "--ckpt-dir",
    "ckpt_dirs",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    multiple=True,
    required=True,
    help="checkpoint directory; may be repeated to concatenate runs",
)
@click.option("--epoch-min", type=int, default=0, help="inclusive lower bound")
@click.option("--epoch-max", type=int, default=10_000, help="inclusive upper bound")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file appended with one record per checkpoint",
)
@click.option(
    "--valid-item",
    type=int,
    default=None,
    help="override cfg.train.valid_item (e.g. --valid-item 64 for a quick sweep)",
)
@click.argument("overrides", type=str, nargs=-1)
def main(
    config: Path,
    ckpt_dirs: tuple[Path, ...],
    epoch_min: int,
    epoch_max: int,
    output: Path,
    valid_item: int | None,
    overrides: tuple[str, ...],
) -> None:
    raw = OmegaConf.load(config)
    if overrides:
        raw = OmegaConf.merge(raw, OmegaConf.from_dotlist(list(overrides)))
    cfg = Config.model_validate(OmegaConf.to_container(raw, resolve=True))
    cfg.train.use_ema = False
    cfg.train.compile = False
    cfg.train.use_wandb = False
    cfg.train.verbose = False
    if valid_item is not None:
        cfg.train.valid_item = valid_item

    fabric = _fabric_from_torchrun()
    fabric.launch()
    if cfg.train.seed is not None:
        fabric.seed_everything(cfg.train.seed)

    client = Client(
        Client.Config(
            train=cfg.train,
            model=cfg.model,
            diffuser=cfg.diffuser,
            loss=cfg.loss,
        ),
    )
    client.setup(fabric=fabric)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ),
    )
    if not any(isinstance(h, logging.StreamHandler) for h in client.logger.handlers):
        client.logger.addHandler(handler)
    client.logger.setLevel(logging.INFO)

    valid_data_config = BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.valid_db,
        sampler_config=None,
        tokenizer_config=cfg.data.tokenizer,
    )
    valid_dataset = BioMolData(valid_data_config)
    valid_dataloader = valid_dataset.create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.global_rank,
        seed=cfg.train.seed,
        drop_last=True,
        batch_size=cfg.train.num_batch,
        num_workers=0,
    )

    world_size = fabric.world_size
    valid_num_item = cfg.train.valid_item // world_size

    if fabric.is_global_zero:
        output.parent.mkdir(parents=True, exist_ok=True)

    ckpts: list[tuple[int, Path]] = []
    for d in ckpt_dirs:
        for path in sorted(Path(d).glob("epoch=*.pt")):
            m = CKPT_RE.search(path.name)
            if not m:
                continue
            ep = int(m.group(1))
            if epoch_min <= ep <= epoch_max:
                ckpts.append((ep, path))

    seen: set[int] = set()
    unique_ckpts: list[tuple[int, Path]] = []
    for ep, p in sorted(ckpts, key=lambda t: t[0]):
        if ep in seen:
            continue
        seen.add(ep)
        unique_ckpts.append((ep, p))

    already: set[int] = set()
    if output.exists():
        for line in output.read_text().splitlines():
            try:
                already.add(int(json.loads(line).get("epoch", -1)))
            except Exception:  # noqa: BLE001
                pass

    client.logger.info(
        "sweep plan: %d checkpoints (range [%d,%d]); %d already recorded",
        len(unique_ckpts), epoch_min, epoch_max, len(already),
    )

    for ep, ckpt_path in unique_ckpts:
        if ep in already:
            client.logger.info("epoch %d already in %s, skipping", ep, output)
            continue

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_sd = _strip_wrapper_prefixes(state_dict["model_state_dict"])
        result = client.model.load_state_dict(model_sd, strict=False)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        client.logger.info(
            "loaded %s (missing=%d, unexpected=%d)",
            ckpt_path.name, len(missing), len(unexpected),
        )
        client._epoch = int(state_dict.get("epoch", ep))  # noqa: SLF001
        client._global_step = int(state_dict.get("global_step", 0))  # noqa: SLF001

        valid_dataloader.sampler.set_epoch(ep)  # pyright: ignore[reportAttributeAccessIssue]
        valid_dataset.set_epoch(ep)

        aggregator = MetricsAggregator(client, "valid", use_wandb=False)
        t0 = time.perf_counter()
        for n_item, step_result in enumerate(client.validation_epoch(valid_dataloader)):
            aggregator.log_step(step_result, ignore_step=True)
            if n_item + 1 >= valid_num_item:
                client.call_callbacks("on_validation_epoch_end")
                break
        means = aggregator.log_epoch()
        wall = time.perf_counter() - t0

        if fabric.is_global_zero:
            record = {
                "epoch": ep,
                "wall_time": wall,
                **{k: float(v) for k, v in means.items()},
            }
            with output.open("a") as f:
                f.write(json.dumps(record) + "\n")
            client.logger.info("epoch %d done: %s", ep, record)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
