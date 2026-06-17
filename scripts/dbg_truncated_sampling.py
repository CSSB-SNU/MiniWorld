"""Truncated sampling: start the reverse diffusion from a lower sigma.

The EDM schedule starts at sigma_0 = sigma_data * sigma_max (=16*160=2560).
At sigma~=160 the noised state is ~99% Gaussian (SNR ~0.1), so we can init
sampling there and skip the very-high-sigma regime where the diverged model's
score is worst. We sweep the start sigma (by overriding scheduler sigma_max)
and report lDDT/RMSD vs GT for the first DB target.
"""
from __future__ import annotations

from pathlib import Path

import click
import torch
from hydra import compose, initialize_config_dir
from lightning import Fabric
from omegaconf import OmegaConf

from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.io.to_cif import batch_to_cif
from miniworld.loss import metrics
from miniworld.models.miniworld_no_single_at_trunk import Client
from run_miniworld_no_single_edm_inference import DataConfig  # type: ignore

torch.set_float32_matmul_precision("medium")

# start_sigma = sigma_data(16) * sigma_max_override
START_SIGMAS = [2560.0, 640.0, 320.0, 160.0, 80.0, 40.0]


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--run-dir", type=click.Path(path_type=Path), required=True)
@click.option("--timesteps", type=int, default=100)
@click.option("--seed", type=int, default=0)
def main(config: Path, ckpt: Path, run_dir: Path, timesteps: int, seed: int) -> None:
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
    cif_dir = run_dir / "structures"
    cif_dir.mkdir(parents=True, exist_ok=True)
    batch_to_cif(batch, None, cif_dir / f"{target}_gt.cif")

    sched_cfg = client.solver.scheduler.config
    sigma_data = sched_cfg.sigma_data
    gt = batch.structure.atom_pos[0]
    gtm = batch.structure.atom_mask[0]

    print(f"target={target} epoch={sd.get('epoch')}  timesteps={timesteps}")
    print(f"{'start_sigma':>11} {'sigma_max':>9} {'lddt':>8} {'rmsd':>8}")
    for s0 in START_SIGMAS:
        sched_cfg.sigma_max = s0 / sigma_data  # start_sigma = sigma_data * sigma_max
        wrapper, b2 = client.prepare(batch)
        torch.manual_seed(seed)
        out = client.sample(wrapper, b2, n_samples=1, timesteps=timesteps)
        pred = out.atom_pos_pred[0]
        lddt = float(metrics.cal_atom_lddt(pred, gt, gtm))
        rmsd = float(metrics.cal_aligned_rmsd(pred, gt, gtm))
        print(f"{s0:11.0f} {sched_cfg.sigma_max:9.3f} {lddt:8.4f} {rmsd:8.3f}")
        batch_to_cif(batch, pred.unsqueeze(0), cif_dir / f"{target}_start{s0:g}.cif")

    print(f"structures -> {cif_dir}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
