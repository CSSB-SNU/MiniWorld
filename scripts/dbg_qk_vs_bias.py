"""Compare the two additive parts of the attention logit: q.k/sqrt(d) vs bias.

logit_ij = (q_i . k_j)/sqrt(d) + bias_ij  ->  softmax over keys j.
Only the variation ACROSS KEYS (per query) drives softmax, so we report the
std over the key axis of each term separately (and absmax). Probe = a single
denoiser forward on noised-GT at a fixed sigma, identical for both checkpoints.
"""
from __future__ import annotations

from pathlib import Path

import click
import torch
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf

from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.models.miniworld_no_single_at_trunk import Client
from run_miniworld_no_single_edm_inference import DataConfig  # type: ignore
from team_gm.modules.layers.augmented_attention import AugmentedAttentionPairBias
from team_gm.modules.layers.ops import sigmoid_gate

torch.set_float32_matmul_precision("medium")

REC: dict = {}
ORIG = AugmentedAttentionPairBias.forward
MAXLL = 9_000_000


def inst_forward(self, single, cond, pair, mask=None):
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

    name = getattr(self, "_dbg_name", None)
    if name is not None and len_res * len_res <= MAXLL:
        scale = hidden ** -0.5
        qk = torch.einsum("abihd,abjhd->abhij", query.float() * scale, key.float())
        bz = bias.float().permute(0, 3, 1, 2)  # (B,H,i,j)
        logit = qk + bz[None]                  # full logit into softmax
        raw_amax = float(logit.abs().max())    # what tanh soft-cap sees
        # what softmax actually uses: per-query spread over keys (common-mode removed)
        spread_mean = float((logit.amax(-1) - logit.amin(-1)).mean())
        spread_max = float((logit.amax(-1) - logit.amin(-1)).max())
        cen_amax = float((logit - logit.mean(-1, keepdim=True)).abs().max())
        REC[name] = (raw_amax, cen_amax, spread_mean, spread_max,
                     float(qk.abs().max()), float(bz.abs().max()))

    out = self._kernel_attention_pair_bias(query, key, value, bias, mask)
    out = rearrange(out, "A B L H D -> A B L (H D)")
    out = sigmoid_gate(gate, out)
    out = self.to_out(out)
    return sigmoid_gate(self.to_scale(cond), out)


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--sigma", type=float, default=16.0)
@click.option("--seed", type=int, default=0)
def main(config, ckpt, sigma, seed) -> None:
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
    for name, mod in client.model.named_modules():
        if isinstance(mod, AugmentedAttentionPairBias) and "diffusion_module" in name:
            mod._dbg_name = name.split("diffusion_module.")[-1]
    AugmentedAttentionPairBias.forward = inst_forward

    bio_cfg = BioMolData.BioMolConfig(
        crop_config=data_cfg.crop, msa_config=data_cfg.msa, DB_config=data_cfg.train_db,
        sampler_config=data_cfg.sampler, tokenizer_config=data_cfg.tokenizer)
    dataset = BioMolData(bio_cfg)
    dataset.set_epoch(0)
    dl = dataset.create_ddp_dataloader(world_size=1, rank=0, seed=seed, drop_last=False,
                                       batch_size=1, num_workers=0, shuffle=True)
    batch = next(iter(dl)).to(device=client.device)
    wrapper, batch = client.prepare(batch)

    sched = client.solver.scheduler
    x0 = batch.structure.atom_pos
    sig = torch.tensor(float(sigma), device=client.device)
    torch.manual_seed(seed)
    noisy = x0 + torch.randn_like(x0) * sig
    x_input = noisy * sched.input_scale(sig)
    t_emb = sched.noise_condition(sig)
    with torch.no_grad():
        wrapper(x_input, t_emb)
    AugmentedAttentionPairBias.forward = ORIG

    print(f"ckpt epoch={sd.get('epoch')}  probe sigma={sigma}")
    print("raw_amax = |q.k/sqrt(d)+bias| max (tanh sees this) ; "
          "spread = per-query max-min over keys (softmax uses this)")
    print(f"{'raw_amax':>9} {'cen_amax':>9} {'spread_mn':>10} {'spread_mx':>10} {'qk_amax':>8} {'bias_amax':>10}  module")
    for n in sorted(REC):
        ra, ca, sm, sx, qa, ba = REC[n]
        tag = "  <== atom-decoder" if "atom_attention_decoder" in n else ""
        if "blocks.0." in n or "blocks.6." in n or "blocks.12." in n or "atom_attention_decoder" in n:
            print(f"{ra:9.1f} {ca:9.1f} {sm:10.2f} {sx:10.1f} {qa:8.1f} {ba:10.1f}  {n}{tag}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
