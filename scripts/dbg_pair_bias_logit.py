"""Per-layer pair-bias magnitude in attention logits + bf16 sensitivity.

Hooks every `attention_pair_bias.to_bias` (the additive bias fed to SDPA),
runs the diffusion sampler in fp32, and per layer reports:
  - absmax / rms of the bias entering the attention logits
  - per-query logit spread (std over keys) and softmax saturation (mean max-prob)
  - bf16 sensitivity: softmax(bias_fp32) vs softmax(round_bf16(bias_fp32)),
    max prob diff and mean total-variation. This isolates whether bf16 rounding
    at these magnitudes actually perturbs the attention distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf

from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.models.miniworld_no_single_at_trunk import Client
from run_miniworld_no_single_edm_inference import DataConfig  # type: ignore

torch.set_float32_matmul_precision("medium")


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

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cc = Client.Config.model_validate(sd["config"])
    cc.train.use_ema = True
    client = Client(cc)
    client.setup(fabric=fabric)
    client.load_state_dict(sd, model_only=True)
    client.model.eval()

    # force fp32 diffusion so to_bias outputs are the true (unrounded) values
    fwd = client.model._forward_module  # noqa: SLF001
    fwd.autocast_bf16 = False
    _orig = fwd.diffusion_forward

    def _fp32_fwd(*a, **kw):
        a = [x.float() if torch.is_tensor(x) and x.dtype == torch.bfloat16 else x for x in a]
        kw = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.bfloat16 else v)
              for k, v in kw.items()}
        return _orig(*a, **kw)

    fwd.diffusion_forward = _fp32_fwd

    bio_cfg = BioMolData.BioMolConfig(
        crop_config=data_cfg.crop, msa_config=data_cfg.msa, DB_config=data_cfg.train_db,
        sampler_config=data_cfg.sampler, tokenizer_config=data_cfg.tokenizer)
    dataset = BioMolData(bio_cfg)
    dataset.set_epoch(0)
    dl = dataset.create_ddp_dataloader(world_size=1, rank=0, seed=seed,
                                       drop_last=False, batch_size=1, num_workers=0, shuffle=True)

    stats: dict[str, dict] = {}

    def analyse(t: torch.Tensor) -> dict:
        # t: (..., keys, n_head) as output by to_bias (B L L2 H); keys at dim=-2
        f = t.detach().float()
        p32 = F.softmax(f, dim=-2)
        p16 = F.softmax(f.bfloat16().float(), dim=-2)
        return {
            "absmax": float(f.abs().max()),
            "rms": float(f.pow(2).mean().sqrt()),
            "key_std": float(f.std(dim=-2).mean()),          # spread driving softmax
            "maxprob": float(p32.max(dim=-2).values.mean()),  # saturation (1.0=hardmax)
            "bf16_maxprobdiff": float((p32 - p16).abs().max()),
            "bf16_meanTV": float(0.5 * (p32 - p16).abs().sum(dim=-2).mean()),
        }

    def mk_hook(name):
        def hook(_m, _i, o):
            if not isinstance(o, torch.Tensor) or not o.is_floating_point():
                return
            s = analyse(o)
            prev = stats.get(name)
            if prev is None or s["absmax"] >= prev["absmax"]:
                stats[name] = s
        return hook

    handles = []
    for name, mod in client.model.named_modules():
        if name.endswith("attention_pair_bias.to_bias"):
            handles.append(mod.register_forward_hook(mk_hook(name)))

    batch = next(iter(dl)).to(device=client.device)
    target = str(batch.name[0])
    with torch.no_grad():
        wrapper, batch = client.prepare(batch)
        torch.manual_seed(seed)
        client.sample(wrapper, batch, n_samples=1, timesteps=20)
    for h in handles:
        h.remove()

    out.write_text(json.dumps({"ckpt": str(ckpt), "epoch": sd.get("epoch"),
                               "target": target, "stats": stats}, indent=0))
    rows = sorted(stats.items(), key=lambda kv: kv[1]["absmax"], reverse=True)
    print(f"target={target} epoch={sd.get('epoch')}  layers={len(stats)}")
    print(f"{'absmax':>9} {'rms':>7} {'key_std':>7} {'maxprob':>7} {'bf16dP':>7} {'bf16TV':>7}  layer")
    for n, s in rows[:20]:
        print(f"{s['absmax']:9.1f} {s['rms']:7.1f} {s['key_std']:7.1f} {s['maxprob']:7.3f} "
              f"{s['bf16_maxprobdiff']:7.3f} {s['bf16_meanTV']:7.3f}  {n.split('diffusion_module.')[-1]}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
