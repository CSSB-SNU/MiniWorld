"""Capture per-module activation scale during the trunk forward (client.prepare).

Usage:
    python scripts/dbg_activation_scale.py --ckpt <ckpt.pt> --out <stats.json>
Reuses the no-single EDM inference machinery; runs the deterministic trunk on
the first DB target (seed=0) with forward hooks recording abs-max / rms / nan.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf

from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.models.miniworld_no_single_at_trunk import Client
from run_miniworld_no_single_edm_inference import DataConfig  # type: ignore

torch.set_float32_matmul_precision("medium")


def _stat(t: torch.Tensor) -> dict:
    f = t.detach().float()
    return {
        "dtype": str(t.dtype).replace("torch.", ""),
        "shape": list(t.shape),
        "absmax": float(f.abs().max()) if f.numel() else 0.0,
        "rms": float(f.pow(2).mean().sqrt()) if f.numel() else 0.0,
        "nan": int(torch.isnan(f).any()),
        "inf": int(torch.isinf(f).any()),
    }


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--seed", type=int, default=0)
def main(config: Path, ckpt: Path, out: Path, seed: int) -> None:
    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name)
    data_cfg = DataConfig.model_validate(OmegaConf.to_container(cfg.data, resolve=True))

    fabric = Fabric(devices=1, num_nodes=1)
    fabric.launch()
    fabric.seed_everything(seed)

    state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
    client_config = Client.Config.model_validate(state_dict["config"])
    client_config.train.use_ema = True
    client = Client(client_config)
    client.setup(fabric=fabric)
    client.load_state_dict(state_dict, model_only=True)
    client.model.eval()

    bio_cfg = BioMolData.BioMolConfig(
        crop_config=data_cfg.crop, msa_config=data_cfg.msa,
        DB_config=data_cfg.train_db, sampler_config=data_cfg.sampler,
        tokenizer_config=data_cfg.tokenizer,
    )
    dataset = BioMolData(bio_cfg)
    dataset.set_epoch(0)
    dl = dataset.create_ddp_dataloader(
        world_size=1, rank=0, seed=seed, drop_last=False,
        batch_size=1, num_workers=0, shuffle=True,
    )

    stats: dict[str, dict] = {}

    def mk_hook(name):
        def hook(_m, _inp, outp):
            tensors = []
            if isinstance(outp, torch.Tensor):
                tensors = [outp]
            elif isinstance(outp, (tuple, list)):
                tensors = [o for o in outp if isinstance(o, torch.Tensor)]
            best = None
            for t in tensors:
                if not t.is_floating_point():
                    continue
                s = _stat(t)
                if best is None or s["absmax"] > best["absmax"]:
                    best = s
            if best is not None:
                # keep running MAX across all (timestep) invocations
                prev = stats.get(name)
                if prev is None or best["absmax"] >= prev["absmax"]:
                    stats[name] = best
        return hook

    handles = []
    for name, mod in client.model.named_modules():
        if name and len(list(mod.children())) == 0:  # leaf modules only
            handles.append(mod.register_forward_hook(mk_hook(name)))

    batch = next(iter(dl))
    batch = batch.to(device=client.device)
    target = str(batch.name[0])
    with torch.no_grad():
        wrapper, batch = client.prepare(batch)
        torch.manual_seed(seed)
        client.sample(wrapper, batch, n_samples=1, timesteps=20)

    for h in handles:
        h.remove()

    out.write_text(json.dumps({"ckpt": str(ckpt), "target": target,
                               "epoch": state_dict.get("epoch"),
                               "stats": stats}, indent=0))
    print(f"target={target} epoch={state_dict.get('epoch')} modules={len(stats)} -> {out}")
    # quick top-10 by absmax
    top = sorted(stats.items(), key=lambda kv: kv[1]["absmax"], reverse=True)[:10]
    for n, s in top:
        print(f"  absmax={s['absmax']:.3e} rms={s['rms']:.3e} nan={s['nan']} {s['dtype']} {n}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
