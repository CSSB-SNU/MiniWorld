"""Per-sampling-step activation census across the collapse window (single model).

Records every diffusion-module leaf activation absmax at EACH sampling step of
one 528 trajectory, plus the denoiser input (x_t) and output (x_update) abs.
Then finds which modules jump most abruptly at the trajectory-collapse step.
"""
from __future__ import annotations

from pathlib import Path

import click
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf

from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.loss import metrics
from miniworld.models.miniworld_no_single_at_trunk import Client
from run_miniworld_no_single_edm_inference import DataConfig  # type: ignore

torch.set_float32_matmul_precision("medium")


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--timesteps", type=int, default=100)
@click.option("--lo", type=int, default=55, help="collapse-window start step for the jump report")
@click.option("--hi", type=int, default=66, help="collapse-window end step")
@click.option("--seed", type=int, default=0)
def main(config, ckpt, timesteps, lo, hi, seed) -> None:
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
    dl = dataset.create_ddp_dataloader(world_size=1, rank=0, seed=seed, drop_last=False,
                                       batch_size=1, num_workers=0, shuffle=True)
    batch = next(iter(dl)).to(device=client.device)
    gt = batch.structure.atom_pos[0]
    gtm = batch.structure.atom_mask[0]

    scheduler = client.solver.scheduler
    sig_sched = scheduler.sampling_time_steps(timesteps).tolist()
    wrapper, batch = client.prepare(batch)

    fwd = client.model._forward_module  # noqa: SLF001
    state = {"step": -1}
    per_step: list[dict] = []
    orig = fwd.diffusion_forward

    def wrapped(*a, **kw):
        state["step"] += 1
        x_t = kw.get("x_t", a[3] if len(a) > 3 else None)
        xin = float(x_t.detach().abs().max()) if torch.is_tensor(x_t) else float("nan")
        per_step.append({"xin": xin, "xout": 0.0, "mods": {}})
        out = orig(*a, **kw)
        if isinstance(out, torch.Tensor):
            per_step[-1]["xout"] = float(out.detach().abs().max())
        return out

    fwd.diffusion_forward = wrapped

    def mk(name):
        short = name.split("diffusion_module.")[-1]
        def hook(_m, _i, o):
            if not per_step:
                return
            t = o if isinstance(o, torch.Tensor) else (
                o[0] if isinstance(o, (tuple, list)) and o and isinstance(o[0], torch.Tensor) else None)
            if t is None or not t.is_floating_point():
                return
            per_step[-1]["mods"][short] = float(t.detach().abs().max())
        return hook

    handles = []
    for name, mod in client.model.named_modules():
        if "diffusion_module" in name and len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(mk(name)))

    torch.manual_seed(seed)
    out = client.sample(wrapper, batch, n_samples=1, timesteps=timesteps)
    for h in handles:
        h.remove()

    # per-step lDDT from the x0-hat trajectory
    mtraj = out.model_traj[0]
    lddt = [float(metrics.cal_atom_lddt(torch.from_numpy(mtraj[i]).to(gt), gt, gtm))
            for i in range(mtraj.shape[0])]

    print(f"steps={len(per_step)}  collapse-window jump report: step {lo} -> {hi}")
    print(f"\n{'step':>4} {'sigma':>8} {'lddt':>6} {'x_in':>9} {'x_out':>9}")
    for i in range(max(0, lo - 3), min(len(per_step), hi + 3)):
        print(f"{i:4d} {sig_sched[i]:8.2f} {lddt[i]:6.3f} {per_step[i]['xin']:9.1f} {per_step[i]['xout']:9.2e}")

    # dump full per-step x per-module matrix for arbitrary slicing
    import json
    names = sorted({k for r in per_step for k in r["mods"]})
    dump = {"lddt": lddt, "sigma": sig_sched[:len(per_step)], "names": names,
            "mat": [[r["mods"].get(n, 0.0) for n in names] for r in per_step]}
    Path("/tmp/stepcensus_full.json").write_text(json.dumps(dump))

    # per-module jump ratio absmax(hi)/absmax(lo)
    a, b = per_step[lo]["mods"], per_step[hi]["mods"]
    rows = []
    for k in a:
        if k in b and a[k] > 1e-6:
            rows.append((b[k] / a[k], a[k], b[k], k))
    rows.sort(reverse=True)
    print(f"\n### top 20 modules by absmax jump  step{lo} -> step{hi}  "
          f"(lddt {lddt[lo]:.3f} -> {lddt[hi]:.3f}) ###")
    print(f"{'ratio':>7} {'amax@lo':>9} {'amax@hi':>9}  module")
    for r, al, bh, k in rows[:20]:
        print(f"{r:7.2f} {al:9.1f} {bh:9.1f}  {k}")

    # per-step trajectory of the top movers across the window
    top = [k for _, _, _, k in rows[:8]]
    w0, w1 = max(0, lo - 4), min(len(per_step), hi + 4)
    print(f"\n### per-step absmax trajectory of top movers (step {w0}..{w1-1}) ###")
    hdr = "  ".join(f"{k.replace('diffusion_transformer.blocks.','tf').replace('.attention_pair_bias','')[:14]:>14}" for k in top)
    print(f"{'step':>4} {'lddt':>6}  {hdr}")
    for i in range(w0, w1):
        vals = "  ".join(f"{per_step[i]['mods'].get(k, 0.0):14.1f}" for k in top)
        print(f"{i:4d} {lddt[i]:6.3f}  {vals}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
