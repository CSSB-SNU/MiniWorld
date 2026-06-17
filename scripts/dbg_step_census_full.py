"""TRULY exhaustive per-step activation census for the diffusion module.

Captures, at every sampling step:
  - output AND input absmax of EVERY module (all nesting levels, so the
    residual stream / block outputs are included, not just leaf increments)
  - SDPA internals via monkeypatch: q/k/v/bias absmax, attention-logit max,
    softmax max-prob (saturation), and attention output absmax
Then finds the single largest activation anywhere per step, and the biggest
mover across the collapse window. If nothing explodes here, nothing explodes.
"""
from __future__ import annotations

import math
from pathlib import Path

import click
import torch
import torch.nn.functional as F
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
@click.option("--lo", type=int, default=61)
@click.option("--hi", type=int, default=63)
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
    gt, gtm = batch.structure.atom_pos[0], batch.structure.atom_mask[0]
    scheduler = client.solver.scheduler
    sig = scheduler.sampling_time_steps(timesteps).tolist()
    wrapper, batch = client.prepare(batch)

    state = {"step": -1}
    per_step: list[dict] = []

    def rec(key, t):
        if torch.is_tensor(t) and t.is_floating_point() and t.numel():
            v = float(t.detach().abs().max())
            d = per_step[state["step"]]["mods"]
            if v > d.get(key, -1):
                d[key] = v

    def tens(o):
        if isinstance(o, torch.Tensor):
            return [o]
        if isinstance(o, (tuple, list)):
            return [x for x in o if isinstance(x, torch.Tensor)]
        return []

    # ---- monkeypatch SDPA to capture attention internals ----
    orig_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(q, k, v, attn_mask=None, *args, **kwargs):
        out = orig_sdpa(q, k, v, attn_mask, *args, **kwargs)
        if state["step"] >= 0 and per_step:
            rec("SDPA.q", q); rec("SDPA.k", k); rec("SDPA.v", v); rec("SDPA.out", out)
            if attn_mask is not None and torch.is_tensor(attn_mask):
                rec("SDPA.bias", attn_mask)
            lq = q.shape[-2]; lk = k.shape[-2]
            if lq * lk <= 3_000_000:  # guard memory for the logit matrix
                scale = 1.0 / math.sqrt(q.shape[-1])
                logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
                if attn_mask is not None and torch.is_tensor(attn_mask):
                    logits = logits + attn_mask.float()
                d = per_step[state["step"]]["mods"]
                d["ATTN.logit_absmax"] = max(d.get("ATTN.logit_absmax", -1), float(logits.abs().max()))
                p = torch.softmax(logits, dim=-1)
                d["ATTN.softmax_maxprob"] = max(d.get("ATTN.softmax_maxprob", -1),
                                                float(p.max(dim=-1).values.mean()))
        return out

    F.scaled_dot_product_attention = patched_sdpa

    fwd = client.model._forward_module  # noqa: SLF001
    orig = fwd.diffusion_forward

    def wrapped(*a, **kw):
        state["step"] += 1
        per_step.append({"mods": {}})
        return orig(*a, **kw)

    fwd.diffusion_forward = wrapped

    handles = []
    for name, mod in client.model.named_modules():
        if "diffusion_module" not in name:
            continue
        short = name.split("diffusion_module.")[-1] or "<root>"

        def out_hook(_m, _i, o, s=short):
            if state["step"] < 0 or not per_step:
                return
            for t in tens(o):
                rec(f"out::{s}", t)

        def in_hook(_m, i, s=short):
            if state["step"] < 0 or not per_step:
                return
            for t in tens(i):
                rec(f"in::{s}", t)

        handles.append(mod.register_forward_hook(out_hook))
        handles.append(mod.register_forward_pre_hook(in_hook))

    torch.manual_seed(seed)
    out = client.sample(wrapper, batch, n_samples=1, timesteps=timesteps)
    for h in handles:
        h.remove()
    F.scaled_dot_product_attention = orig_sdpa

    mtraj = out.model_traj[0]
    lddt = [float(metrics.cal_atom_lddt(torch.from_numpy(mtraj[i]).to(gt), gt, gtm))
            for i in range(mtraj.shape[0])]

    n_keys = len(set().union(*[r["mods"].keys() for r in per_step]))
    print(f"steps={len(per_step)}  distinct activation tensors tracked = {n_keys}")

    # per-step: global max activation anywhere + the attention saturation
    print(f"\n{'step':>4} {'sigma':>7} {'lddt':>6} {'GLOBAL_MAX':>11} {'logit_amax':>11} {'sm_maxprob':>10}  argmax_tensor")
    for i in range(max(0, lo - 6), min(len(per_step), hi + 6)):
        d = per_step[i]["mods"]
        gk = max(d, key=lambda k: d[k])
        print(f"{i:4d} {sig[i]:7.2f} {lddt[i]:6.3f} {d[gk]:11.1f} "
              f"{d.get('ATTN.logit_absmax', float('nan')):11.1f} "
              f"{d.get('ATTN.softmax_maxprob', float('nan')):10.3f}  {gk[:50]}")

    a, b = per_step[lo]["mods"], per_step[hi]["mods"]
    rows = [(b[k] / a[k], a[k], b[k], k) for k in a if k in b and a[k] > 1e-6]
    rows.sort(reverse=True)
    print(f"\n### top 25 by jump step{lo}->step{hi} (lddt {lddt[lo]:.3f}->{lddt[hi]:.3f}) — ALL activations ###")
    print(f"{'ratio':>7} {'@lo':>10} {'@hi':>10}  tensor")
    for r, al, bh, k in rows[:25]:
        print(f"{r:7.2f} {al:10.1f} {bh:10.1f}  {k[:60]}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
