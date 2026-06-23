"""Sampling / inference script for the MiniWorld NO-SINGLE EDM variant.

Same as ``run_miniworld_edm_inference.py`` but for the
``miniworld_no_single_at_trunk`` model: the trunk carries no ``token_single``
track (pair-only Pairformer; diffusion conditioning built from
``token_single_input`` alone). It was trained with
``scripts/run_miniworld_no_single_edm_train.py`` on the
``large_msa3_24_3_no_single`` model config. The only difference from the plain
EDM inference script is the ``Client`` import — the trunk/sample API is
identical.

Generates 3D structures from the trained checkpoint by sampling targets out of
the training/validation LMDB and running the AF3 ODE solver. The
model / diffuser / loss config is read straight from the checkpoint, so the
architecture always matches the weights. EMA weights are applied automatically
on load (the ``ModelEMA`` callback swaps the trained-param EMA shadow into the
model). Per target we save the predicted + ground-truth CIF and log aligned
RMSD / atom lDDT / distogram loss.

Usage (single GPU):
    pixi run python scripts/run_miniworld_no_single_edm_inference.py sample \
        --config configs/miniworld/config_exp_msa3_24_3_no_single_edm.yaml \
        --ckpt epoch=0440.pt \
        --num-targets 4 --n-samples 2 --timesteps 100 \
        data=local_sample_edm
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import click
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf
from pydantic import BaseModel

from miniworld.configs import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TokenizerConfig,
)
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.io.to_cif import batch_to_cif
from miniworld.loss import metrics
from miniworld.models.miniworld_no_single_at_trunk import Client

torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)


class DataConfig(BaseModel):
    """Data-loading sub-config (mirrors the EDM training script)."""

    train_db: BioMolDBConfig
    crop: CropConfig
    msa: MSAConfig
    tokenizer: TokenizerConfig
    sampler: SamplerConfig


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Hydra config (used only for the `data` subtree).",
)
@click.option(
    "--ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="EDM checkpoint; model/diffuser/loss config is read from it.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/miniworld_no_single_edm_sample"),
    show_default=True,
)
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write directly into this dir (no date/time subdir). Lets multiple "
         "invocations share one output folder; structures are named per target "
         "so they don't collide.",
)
@click.option("--num-targets", type=int, default=4, show_default=True,
              help="Number of (kept) targets to sample from the DB.")
@click.option("--max-token", type=int, default=0, show_default=True,
              help="If >0, skip targets whose token_length >= this value "
                   "(i.e. cropped / too-large structures), so only small "
                   "fully-contained structures are evaluated.")
@click.option("--n-samples", type=int, default=2, show_default=True,
              help="Diffusion samples per target (augmentation axis).")
@click.option("--timesteps", type=int, default=100, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--compile/--no-compile", "do_compile", default=False, show_default=True)
@click.option("--ema/--no-ema", "use_ema", default=True, show_default=True,
              help="Use EMA weights (default) or the raw trained weights.")
@click.option("--fp32/--no-fp32", "force_fp32", default=False, show_default=True,
              help="Run the diffusion module in fp32 (disable bf16 autocast).")
@click.option("--save-traj/--no-save-traj", "save_traj", default=False, show_default=True,
              help="Dump per-step sampling trajectory (npy + per-step lDDT/RMSD + frame CIFs).")
@click.option("--attn-bf16/--no-attn-bf16", "attn_bf16", default=False, show_default=True,
              help="With --fp32: keep only the attention kernel (q,k,v,bias) in bf16, "
                   "rest of the diffusion forward in fp32. Requires --fp32.")
@click.option("--job-name", type=str, default=None)
@click.argument("overrides", type=str, nargs=-1)
def sample(  # noqa: PLR0915
    config: Path,
    ckpt: Path,
    output_dir: Path,
    run_dir: Path | None,
    num_targets: int,
    max_token: int,
    n_samples: int,
    timesteps: int,
    seed: int,
    do_compile: bool,
    use_ema: bool,
    force_fp32: bool,
    save_traj: bool,
    attn_bf16: bool,
    job_name: str | None,
    overrides: tuple[str, ...],
) -> None:
    # --- compose data config -------------------------------------------------
    with initialize_config_dir(str(config.parent.absolute()), version_base=None):
        cfg = compose(config_name=config.name, overrides=list(overrides))
    data_cfg = DataConfig.model_validate(OmegaConf.to_container(cfg.data, resolve=True))

    fabric = Fabric(devices=1, num_nodes=1)
    fabric.launch()
    fabric.seed_everything(seed)

    if run_dir is not None:
        # Explicit shared folder: no date/time subdir, so several invocations
        # (e.g. one per interface type) land in the same output directory.
        run_sub_dir = run_dir
    else:
        date_dir = output_dir / time.strftime("%Y-%m-%d")
        run_name = time.strftime("%H%M%S")
        if job_name:
            run_name += f"_{job_name}"
        run_sub_dir = date_dir / run_name
    run_sub_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = run_sub_dir / "structures"
    cif_dir.mkdir(parents=True, exist_ok=True)

    # --- build client from the checkpoint's own config -----------------------
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
    client_config = Client.Config.model_validate(state_dict["config"])
    # use_ema controls whether the ModelEMA callback is registered: when True the
    # checkpoint's EMA shadow is swapped into the (trained) params on load; when
    # False the raw trained weights from model_state_dict are used as-is.
    client_config.train.use_ema = use_ema
    client = Client(client_config)

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(run_sub_dir / "sample.log")
    fh.setFormatter(formatter)
    client.logger.addHandler(fh)
    client.logger.info(
        "ckpt=%s epoch=%s step=%s | num_targets=%d n_samples=%d timesteps=%d ema=%s",
        ckpt, state_dict.get("epoch"), state_dict.get("global_step"),
        num_targets, n_samples, timesteps, use_ema,
    )

    if do_compile:
        torch._dynamo.config.cache_size_limit = 128  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001
        client.model.compile(dynamic=False)
        client.logger.info("Compiled model")

    OmegaConf.save(
        OmegaConf.create({"data": OmegaConf.to_container(cfg.data, resolve=True),
                          "model": client_config.model.model_dump(mode="json"),
                          "diffuser": client_config.diffuser.model_dump(mode="json")}),
        run_sub_dir / (f"config_{job_name}.yaml" if job_name else "config.yaml"),
    )

    client.setup(fabric=fabric)
    # Applies EMA shadow to the trained (diffusion) params via ModelEMA callback.
    client.load_state_dict(state_dict, model_only=True)
    client.model.eval()

    if force_fp32:
        # Disable bf16 autocast in the diffusion forward. The frozen trunk still
        # emits bf16 conditioning, so cast those two tensors to fp32 before the
        # diffusion module (fp32 params) to avoid dtype-mismatch errors.
        fwd = client.model._forward_module  # noqa: SLF001
        fwd.autocast_bf16 = False
        _orig = fwd.diffusion_forward

        def _fp32_diffusion_forward(*a, **kw):
            a = list(a)
            # diffusion_forward(reference, scheme, structure, x_t, x_mask,
            #                   t_emb, token_single_input, token_pair_trunk)
            for i, x in enumerate(a):
                if torch.is_tensor(x) and x.dtype == torch.bfloat16:
                    a[i] = x.float()
            for k, x in kw.items():
                if torch.is_tensor(x) and x.dtype == torch.bfloat16:
                    kw[k] = x.float()
            return _orig(*a, **kw)

        fwd.diffusion_forward = _fp32_diffusion_forward
        client.logger.info("FP32 diffusion forward enabled (autocast_bf16=False)")

    if attn_bf16:
        if not force_fp32:
            msg = "--attn-bf16 requires --fp32 (otherwise everything is already bf16)."
            raise click.UsageError(msg)
        # fp32 everywhere EXCEPT the attention kernel: cast q,k,v,bias to bf16 right
        # before the kernel and cast its output back to fp32. Config can't express
        # this (autocast_bf16 is all-or-nothing), so we patch the class forward.
        from einops import rearrange, repeat
        from team_gm.modules.layers.augmented_attention import AugmentedAttentionPairBias
        from team_gm.modules.layers.ops import sigmoid_gate

        def _attn_bf16_forward(self, single, cond, pair, mask=None):
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
            # attention core in bf16, everything else fp32
            out = self._kernel_attention_pair_bias(
                query.bfloat16(), key.bfloat16(), value.bfloat16(),
                bias.bfloat16(), mask).float()
            out = rearrange(out, "A B L H D -> A B L (H D)")
            out = sigmoid_gate(gate, out)
            out = self.to_out(out)
            return sigmoid_gate(self.to_scale(cond), out)

        AugmentedAttentionPairBias.forward = _attn_bf16_forward
        client.logger.info("attn-bf16: attention kernel (q,k,v,bias) in bf16, rest fp32")

    # --- dataloader over the training DB -------------------------------------
    bio_cfg = BioMolData.BioMolConfig(
        crop_config=data_cfg.crop,
        msa_config=data_cfg.msa,
        DB_config=data_cfg.train_db,
        sampler_config=data_cfg.sampler,
        tokenizer_config=data_cfg.tokenizer,
    )
    dataset = BioMolData(bio_cfg)
    dataset.set_epoch(0)
    dataloader = dataset.create_ddp_dataloader(
        world_size=1,
        rank=0,
        seed=seed,
        drop_last=False,
        batch_size=1,
        num_workers=0,
        shuffle=True,
    )

    client.logger.info("Start EDM sampling")
    done = 0
    for raw_batch in dataloader:
        if done >= num_targets:
            break
        batch = raw_batch.to(device=client.device)
        name = str(batch.name[0])
        if max_token and int(batch.token_length) >= max_token:
            client.logger.info(
                "skip %s | n_tokens=%d >= max_token=%d (cropped/too large)",
                name, int(batch.token_length), max_token,
            )
            continue
        client.logger.info(
            "target %d/%d %s | n_tokens=%d n_atoms=%d n_msa=%d",
            done + 1, num_targets, name,
            batch.token_length, batch.atom_length, batch.msa_count,
        )

        # Run trunk once, then sample n_samples diffusion trajectories.
        wrapper, batch = client.prepare(batch)
        torch.manual_seed(seed * 100003 + done * 1009)
        output = client.sample(
            wrapper, batch, n_samples=n_samples, timesteps=timesteps,
        )

        # Ground truth once.
        batch_to_cif(batch, None, cif_dir / f"{name}_gt.cif")

        gt_pos = batch.structure.atom_pos[0]
        gt_mask = batch.structure.atom_mask[0]
        best_rmsd, best_lddt, best_k = float("inf"), 0.0, 0
        for k in range(n_samples):
            pred_k = output.atom_pos_pred[k:k + 1]
            batch_to_cif(batch, pred_k, cif_dir / f"{name}_pred_{k}.cif")
            rmsd = float(metrics.cal_aligned_rmsd(output.atom_pos_pred[k], gt_pos, gt_mask))
            lddt = float(metrics.cal_atom_lddt(output.atom_pos_pred[k], gt_pos, gt_mask))
            client.logger.info("  sample %d: rmsd=%.3f lddt=%.4f", k, rmsd, lddt)
            if rmsd < best_rmsd:
                best_rmsd, best_lddt, best_k = rmsd, lddt, k

        client.logger.info(
            "target %s DONE | best(sample=%d) rmsd=%.3f lddt=%.4f",
            name, best_k, best_rmsd, best_lddt,
        )

        if save_traj:
            import numpy as np
            traj_dir = run_sub_dir / "traj"
            traj_dir.mkdir(parents=True, exist_ok=True)
            # output.{model_traj,inter_traj}: (n_samples, T, L, 3)
            #   model_traj = model's x0-hat at each step; inter_traj = x_t path.
            np.save(traj_dir / f"{name}_model_traj.npy", output.model_traj)
            np.save(traj_dir / f"{name}_inter_traj.npy", output.inter_traj)
            mtraj = output.model_traj[best_k]  # (T, L, 3) x0-hat for best sample
            n_steps = mtraj.shape[0]
            # per-step lDDT/RMSD of the x0-hat vs GT
            curve = traj_dir / f"{name}_curve.tsv"
            with curve.open("w") as fh:
                fh.write("step\tlddt\trmsd\n")
                for ti in range(n_steps):
                    p = torch.from_numpy(mtraj[ti]).to(gt_pos)
                    ld = float(metrics.cal_atom_lddt(p, gt_pos, gt_mask))
                    rm = float(metrics.cal_aligned_rmsd(p, gt_pos, gt_mask))
                    fh.write(f"{ti}\t{ld:.4f}\t{rm:.3f}\n")
            client.logger.info("  saved per-step curve -> %s", curve)
            # subsampled frame CIFs (x0-hat) for visual inspection
            stride = max(1, n_steps // 20)
            for ti in list(range(0, n_steps, stride)) + [n_steps - 1]:
                p = torch.from_numpy(mtraj[ti:ti + 1]).to(gt_pos)
                batch_to_cif(batch, p, traj_dir / f"{name}_x0hat_step{ti:03d}.cif")
        done += 1

    client.logger.info("Sampling complete. %d targets -> %s", done, run_sub_dir)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
