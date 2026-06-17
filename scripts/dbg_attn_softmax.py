"""Directly measure attention softmax saturation per sampling step.

The diffusion transformer uses AugmentedAttentionPairBias whose attention runs
inside a Triton kernel (softmax hidden). We monkeypatch its forward with an
exact replica that ALSO computes the logits/softmax in PyTorch (guarded by L
size) to record per (step, module): mean max-prob and normalized entropy
(1=uniform, 0=one-hot). The real output is still produced by the original
kernel, so model behaviour is unchanged.
"""
from __future__ import annotations

import math
from pathlib import Path

import click
import torch
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf

from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.loss import metrics
from miniworld.models.miniworld_no_single_at_trunk import Client
from run_miniworld_no_single_edm_inference import DataConfig  # type: ignore
from team_gm.modules.layers.augmented_attention import AugmentedAttentionPairBias
from team_gm.modules.layers.ops import sigmoid_gate

torch.set_float32_matmul_precision("medium")

STATE = {"step": -1}
REC: list[dict] = []
ORIG = AugmentedAttentionPairBias.forward
MAXLL = 1_500_000


def inst_forward(self, single, cond, pair, mask=None):
    # exact replica of augmented_attention.AugmentedAttentionPairBias.forward
    # (non-chunked path) + softmax recording.
    single = self.ada_ln_in(single, cond)
    pair = self.ln_pair(pair)
    bias = self.to_bias(pair)
    query = self.to_query(single)
    key = self.to_key(single)
    value = self.to_value(single)
    gate = self.to_gate(single)
    num_aug, batch, len_res = query.shape[:3]
    n_head, hidden = self.n_head, query.shape[-1] // self.n_head
    query, key, value = [x.view(num_aug, batch, len_res, n_head, hidden)
                         for x in (query, key, value)]
    if mask is not None and mask.ndim == 2:  # noqa: PLR2004
        mask = repeat(mask, "B L -> A B L", A=single.shape[0])
    if self.use_qk_norm:
        query = self.norm_query(query)
        key = self.norm_key(key)

    # ---- record softmax saturation (guarded; clones, no behaviour change) ----
    if STATE["step"] >= 0 and REC and len_res * len_res <= MAXLL:
        qs = query.float() * (hidden ** -0.5)
        att = torch.einsum("abihd,abjhd->abhij", qs, key.float())  # (A,B,H,L,L)
        att = att + bias.float().permute(0, 3, 1, 2)[None]
        if mask is not None:
            att = att.masked_fill(~mask[:, :, None, None, :], float("-inf"))
        p = torch.softmax(att, dim=-1)
        nk = att.shape[-1]
        ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1).mean()
        name = getattr(self, "_dbg_name", str(id(self)))
        REC[STATE["step"]][name] = (
            float(p.max(dim=-1).values.mean()),
            float(ent / math.log(nk)),
            float(att.abs().max()),
            nk,
        )

    out = self._kernel_attention_pair_bias(query, key, value, bias, mask)
    out = rearrange(out, "A B L H D -> A B L (H D)")
    out = sigmoid_gate(gate, out)
    out = self.to_out(out)
    return sigmoid_gate(self.to_scale(cond), out)


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--timesteps", type=int, default=100)
@click.option("--seed", type=int, default=0)
def main(config, ckpt, timesteps, seed) -> None:
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

    n_tag = 0
    for name, mod in client.model.named_modules():
        if isinstance(mod, AugmentedAttentionPairBias) and "diffusion_module" in name:
            mod._dbg_name = name.split("diffusion_module.")[-1]
            n_tag += 1
    print(f"tagged {n_tag} AugmentedAttentionPairBias modules")
    AugmentedAttentionPairBias.forward = inst_forward

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

    fwd = client.model._forward_module  # noqa: SLF001
    orig = fwd.diffusion_forward

    def wrapped(*a, **kw):
        STATE["step"] += 1
        REC.append({})
        return orig(*a, **kw)

    fwd.diffusion_forward = wrapped

    torch.manual_seed(seed)
    out = client.sample(wrapper, batch, n_samples=1, timesteps=timesteps)
    AugmentedAttentionPairBias.forward = ORIG

    mtraj = out.model_traj[0]
    lddt = [float(metrics.cal_atom_lddt(torch.from_numpy(mtraj[i]).to(gt), gt, gtm))
            for i in range(mtraj.shape[0])]
    captured = sorted(set().union(*[set(r) for r in REC])) if REC else []
    print(f"captured (Lq*Lk<= {MAXLL}) = {len(captured)} : {captured[:4]} ...")

    watch = [n for n in captured if any(f"blocks.{b}." in n for b in (0, 6, 12, 18, 23))]
    print(f"\n{'step':>4} {'sig':>6} {'lddt':>6}  " +
          "  ".join(f"{w.split('.blocks.')[1].split('.')[0]:>5}block" for w in watch))
    print("            cells = maxprob (norm_entropy)")
    for i in range(50, min(len(REC), 70)):
        cells = []
        for w in watch:
            if w in REC[i]:
                mp, ne, _, _ = REC[i][w]
                cells.append(f"{mp:.2f}({ne:.2f})".rjust(11))
            else:
                cells.append("-".rjust(11))
        print(f"{i:4d} {sig[i]:6.1f} {lddt[i]:6.3f}  " + "  ".join(cells))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
