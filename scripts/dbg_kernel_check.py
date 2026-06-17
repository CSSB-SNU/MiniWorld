"""Verify the augmented-attention Triton kernel against PyTorch naive.

For each AugmentedAttentionPairBias call during real sampling, runs the SAME
captured (q,k,v,bias,mask) through:
  (1) the model's real kernel (Triton),
  (2) PyTorch naive in the same dtype (bf16),
  (3) PyTorch naive in fp32 (reference).
Reports per step the worst relative error:
  tri-vs-torchbf16  -> kernel correctness bug (should be ~0)
  bf16-vs-fp32      -> pure bf16 precision
  tri-vs-fp32       -> total
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
from miniworld.loss import metrics
from miniworld.models.miniworld_no_single_at_trunk import Client
from run_miniworld_no_single_edm_inference import DataConfig  # type: ignore
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.layers.augmented_attention import AugmentedAttentionPairBias
from team_gm.modules.layers.ops import sigmoid_gate

torch.set_float32_matmul_precision("medium")

STATE = {"step": -1}
REC: list[dict] = []
ORIG = AugmentedAttentionPairBias.forward
MAXLL = 1_500_000


def _rel(x, y):
    return float((x.float() - y.float()).norm() / (y.float().norm() + 1e-9))


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

    if (STATE["step"] >= 0 and REC and len_res * len_res <= MAXLL
            and self.implementation == ImplementationType.TRITON):
        name = getattr(self, "_dbg_name", str(id(self)))
        impl0 = self.implementation
        with torch.no_grad():
            out_tri = self._kernel_attention_pair_bias(
                query.clone(), key.clone(), value.clone(), bias.clone(), mask)
            self.implementation = ImplementationType.PYTORCH
            out_bf16 = self._kernel_attention_pair_bias(
                query.clone(), key.clone(), value.clone(), bias.clone(), mask)
            out_fp32 = self._kernel_attention_pair_bias(
                query.float().clone(), key.float().clone(), value.float().clone(),
                bias.float().clone(), mask)
            self.implementation = impl0
        REC[STATE["step"]][name] = (
            _rel(out_tri, out_bf16),   # kernel bug?
            _rel(out_bf16, out_fp32),  # bf16 precision
            _rel(out_tri, out_fp32),   # total
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

    impls = set()
    for name, mod in client.model.named_modules():
        if isinstance(mod, AugmentedAttentionPairBias) and "diffusion_module" in name:
            mod._dbg_name = name.split("diffusion_module.")[-1]
            impls.add(str(mod.implementation))
    print(f"diffusion AugmentedAttentionPairBias implementations in use: {impls}")
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
    sig = client.solver.scheduler.sampling_time_steps(timesteps).tolist()
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
    cap = sorted(set().union(*[set(r) for r in REC])) if REC else []
    print(f"compared modules/step = {len(cap)}")

    print(f"\n{'step':>4} {'sig':>7} {'lddt':>6}  {'tri_vs_bf16':>12} {'bf16_vs_fp32':>13} {'tri_vs_fp32':>12}  worst_block(kernel)")
    glob = (0.0, "")
    for i in range(len(REC)):
        if not REC[i]:
            continue
        kb = max(REC[i].items(), key=lambda kv: kv[1][0])   # worst kernel-vs-bf16
        pb = max(v[1] for v in REC[i].values())             # worst bf16 precision
        tb = max(v[2] for v in REC[i].values())
        if kb[1][0] > glob[0]:
            glob = (kb[1][0], f"step{i} {kb[0]}")
        if i % 5 == 0 or 57 <= i <= 66:
            print(f"{i:4d} {sig[i]:7.2f} {lddt[i]:6.3f}  {kb[1][0]:12.2e} {pb:13.2e} {tb:12.2e}  {kb[0][:28]}")
    print(f"\nWORST kernel-vs-naive(bf16) relative error anywhere: {glob[0]:.3e} @ {glob[1]}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
