"""One-to-one diffusion-module activation diff between two checkpoints.

Feeds IDENTICAL input (same frozen-trunk condition injected into both, same
off-manifold x_t from the 528 collapse trajectory, same sigma) through the
diffusion module of checkpoint A (good, 505) and B (first-bad, 510), captures
every leaf-module output, and reports per-module relative-L2 / cosine
divergence in forward order. Since inputs are identical, all divergence is due
to the changed diffusion weights -> locates where the breakage originates.
"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf

from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.models.miniworld_no_single_at_trunk import Client
from run_miniworld_no_single_edm_inference import DataConfig  # type: ignore

torch.set_float32_matmul_precision("medium")


def build(ckpt: Path, fabric: Fabric) -> Client:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cc = Client.Config.model_validate(sd["config"])
    cc.train.use_ema = True
    client = Client(cc)
    client.setup(fabric=fabric)
    client.load_state_dict(sd, model_only=True)
    client.model.eval()
    return client


def capture(client: Client, wrapper, x_input, t_emb):
    acts: dict[str, torch.Tensor] = {}
    order: list[str] = []

    def mk(name):
        def hook(_m, _i, o):
            t = o if isinstance(o, torch.Tensor) else (
                o[0] if isinstance(o, (tuple, list)) and o and isinstance(o[0], torch.Tensor) else None)
            if t is None or not t.is_floating_point():
                return
            if name not in acts:
                order.append(name)
            acts[name] = t.detach().float().cpu()
        return hook

    handles = []
    for name, mod in client.model.named_modules():
        if "diffusion_module" in name and len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        wrapper(x_input, t_emb)
    for h in handles:
        h.remove()
    return acts, order


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt-a", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt-b", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--traj", type=click.Path(exists=True, path_type=Path), required=True,
              help="inter_traj.npy (1,T,L,3) to draw the off-manifold probe x_t from.")
@click.option("--step", type=int, default=60, help="trajectory step to probe.")
@click.option("--seed", type=int, default=0)
def main(config, ckpt_a, ckpt_b, traj, step, seed) -> None:
    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name)
    data_cfg = DataConfig.model_validate(OmegaConf.to_container(cfg.data, resolve=True))
    fabric = Fabric(devices=1, num_nodes=1)
    fabric.launch()
    fabric.seed_everything(seed)

    ca = build(ckpt_a, fabric)
    cb = build(ckpt_b, fabric)

    bio_cfg = BioMolData.BioMolConfig(
        crop_config=data_cfg.crop, msa_config=data_cfg.msa, DB_config=data_cfg.train_db,
        sampler_config=data_cfg.sampler, tokenizer_config=data_cfg.tokenizer)
    dataset = BioMolData(bio_cfg)
    dataset.set_epoch(0)
    dl = dataset.create_ddp_dataloader(world_size=1, rank=0, seed=seed, drop_last=False,
                                       batch_size=1, num_workers=0, shuffle=True)
    batch = next(iter(dl)).to(device=ca.device)

    # frozen-trunk condition built ONCE; injected into both -> identical inputs.
    wa, batch = ca.prepare(batch)
    cond = wa.condition
    wb, _ = cb.prepare(batch)
    wb.condition = cond  # force identical conditioning

    # off-manifold probe x_t: generated IN-RUN with model A so the crop matches
    # this batch (BioMolData random-crops, so a saved traj would mismatch L).
    sched = ca.solver.scheduler
    sig_sched = sched.sampling_time_steps(100).tolist()
    torch.manual_seed(seed)
    out = ca.sample(wa, batch, n_samples=1, timesteps=100)
    sigma = torch.tensor(float(sig_sched[step]), device=ca.device)
    x = torch.from_numpy(out.inter_traj[0, step]).to(ca.device).unsqueeze(0)  # (1, L, 3)
    c_in = sched.input_scale(sigma)
    x_input = (x * c_in).float()
    t_emb = sched.noise_condition(sigma)

    acts_a, order = capture(ca, wa, x_input, t_emb)
    acts_b, _ = capture(cb, wb, x_input, t_emb)

    print(f"probe step={step} sigma={float(sigma):.2f}  modules={len(order)}")
    rows = []
    for name in order:
        if name not in acts_b:
            continue
        a, b = acts_a[name], acts_b[name]
        if a.shape != b.shape:
            continue
        na = a.norm().item()
        rel = (b - a).norm().item() / (na + 1e-9)
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rows.append((name.split("diffusion_module.")[-1], rel, cos,
                     a.abs().max().item(), b.abs().max().item()))

    print("\n### forward order (where divergence first explodes) ###")
    print(f"{'relL2':>9} {'cos':>7} {'amaxA':>9} {'amaxB':>9}  module")
    for name, rel, cos, ama, amb in rows:
        flag = "  <==" if rel > 0.5 else ""
        print(f"{rel:9.3f} {cos:7.3f} {ama:9.1f} {amb:9.1f}  {name}{flag}")

    print("\n### top 15 by relative-L2 divergence ###")
    for name, rel, cos, ama, amb in sorted(rows, key=lambda r: r[1], reverse=True)[:15]:
        print(f"{rel:9.3f} {cos:7.3f} {ama:9.1f} {amb:9.1f}  {name}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
