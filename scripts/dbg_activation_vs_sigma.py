"""Per-sampling-step (per-sigma) activation magnitude during EDM sampling.

Tags each denoiser forward with the sampling step (-> sigma) and records, per
step, the global max activation over the diffusion module plus a watchlist of
the pair-bias `to_bias` layers. Reveals whether activations blow up in the
sigma window where the trajectory collapses.
"""
from __future__ import annotations

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

WATCH = [
    "atom_attention_decoder.atom_transformer.blocks.0.attention_pair_bias.to_bias",
    "atom_attention_decoder.atom_transformer.blocks.1.attention_pair_bias.to_bias",
    "diffusion_transformer.blocks.0.attention_pair_bias.to_bias",
    "diffusion_transformer.blocks.6.attention_pair_bias.to_bias",
]


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--timesteps", type=int, default=100)
@click.option("--seed", type=int, default=0)
def main(config: Path, ckpt: Path, timesteps: int, seed: int) -> None:
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

    bio_cfg = BioMolData.BioMolConfig(
        crop_config=data_cfg.crop, msa_config=data_cfg.msa, DB_config=data_cfg.train_db,
        sampler_config=data_cfg.sampler, tokenizer_config=data_cfg.tokenizer)
    dataset = BioMolData(bio_cfg)
    dataset.set_epoch(0)
    dl = dataset.create_ddp_dataloader(world_size=1, rank=0, seed=seed,
                                       drop_last=False, batch_size=1, num_workers=0, shuffle=True)
    batch = next(iter(dl)).to(device=client.device)
    target = str(batch.name[0])

    scheduler = client.solver.scheduler
    sigma_sched = scheduler.sampling_time_steps(timesteps).tolist()  # len T+1, step i -> sigma_sched[i]

    wrapper, batch = client.prepare(batch)

    fwd = client.model._forward_module  # noqa: SLF001
    state = {"step": -1}
    per_step: list[dict] = []
    orig = fwd.diffusion_forward

    def wrapped(*a, **kw):
        state["step"] += 1
        per_step.append({"gmax": 0.0, "gname": "", "watch": {}, "out": 0.0})
        out = orig(*a, **kw)
        if isinstance(out, torch.Tensor):
            per_step[-1]["out"] = float(out.detach().abs().max())
        return out

    fwd.diffusion_forward = wrapped

    def mk_hook(name):
        short = name.split("diffusion_module.")[-1]
        is_watch = any(name.endswith(w) for w in WATCH)
        # to_bias / ln_pair are driven by the frozen, cached trunk pair -> constant
        # across steps; exclude from the sigma-dependent global max.
        is_const = name.endswith("to_bias") or name.endswith("ln_pair")
        def hook(_m, _i, o):
            if state["step"] < 0 or not per_step:
                return
            t = o if isinstance(o, torch.Tensor) else (
                o[0] if isinstance(o, (tuple, list)) and o and isinstance(o[0], torch.Tensor) else None)
            if t is None or not t.is_floating_point():
                return
            a = float(t.detach().abs().max())
            rec = per_step[-1]
            if not is_const and a > rec["gmax"]:
                rec["gmax"], rec["gname"] = a, short
            if is_watch:
                rec["watch"][short] = a
        return hook

    handles = []
    for name, mod in client.model.named_modules():
        if "diffusion_module" in name and len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(mk_hook(name)))

    torch.manual_seed(seed)
    with torch.no_grad():
        client.sample(wrapper, batch, n_samples=1, timesteps=timesteps)
    for h in handles:
        h.remove()

    print(f"target={target} epoch={sd.get('epoch')} steps={len(per_step)}")
    print(f"{'step':>4} {'sigma':>8} {'x_update':>10} {'sigma_dep_max':>13}  {'sigma_dep_layer':<52}")
    for i, rec in enumerate(per_step):
        sig = sigma_sched[i] if i < len(sigma_sched) else float("nan")
        print(f"{i:4d} {sig:8.2f} {rec['out']:10.2e} {rec['gmax']:13.2e}  {rec['gname'][:52]:<52}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
