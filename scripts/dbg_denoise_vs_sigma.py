"""One-step denoising vs noise scale (replicates the EDM TRAINING objective).

For a fixed sweep of sigma, add noise to the GT exactly as EuclideanDiffuser
does at train time, run the model once, form the EDM x0 estimate
(c_skip*noisy + c_out*update), and report the per-sigma training loss term plus
lDDT/RMSD of the denoised structure vs GT. Saves the denoised CIF per sigma.

This shows why train loss stays low (single-step denoising of noised-GT is
anchored) even when full multi-step sampling has collapsed.
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
from miniworld.utils.structure.align import weighted_align
from run_miniworld_no_single_edm_inference import DataConfig  # type: ignore

torch.set_float32_matmul_precision("medium")

SIGMAS = [0.5, 1.0, 2.0, 4.8, 8.0, 16.0, 32.0, 64.0, 160.0, 640.0, 2560.0]


@click.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--run-dir", type=click.Path(path_type=Path), required=True)
@click.option("--seed", type=int, default=0)
def main(config: Path, ckpt: Path, run_dir: Path, seed: int) -> None:
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

    scheduler = client.diffuser.scheduler
    batch_to_cif(batch, None, cif_dir / f"{target}_gt.cif")

    def flat_LL(t):  # leading dims -> first sample, return (L, ...)
        return t.reshape(-1, *t.shape[-2:])[0] if t.dim() >= 3 else t.reshape(-1, t.shape[-1])[0]

    print(f"target={target} epoch={sd.get('epoch')}  (train sigma median ~4.8, band ~[1,22])")
    print(f"{'sigma':>8} {'in_scale':>9} {'loss_w':>9} {'train_loss':>11} "
          f"{'noisy_lddt':>10} {'pred_lddt':>10} {'pred_rmsd':>10}")

    for sval in SIGMAS:
        # force fixed sigma in the train-time noising path
        scheduler.sample_noise = lambda n, _s=sval: torch.full((n,), float(_s))
        torch.manual_seed(seed)
        x0, x_input, x_mask, t_emb, sigma = client.diffuser.sample(
            x0=batch.structure.atom_pos, mask=batch.structure.atom_pos_mask, num_augment=1)
        with torch.no_grad():
            atom_pos_update, _ = client.model.forward(
                msa=batch.msa, template=batch.template, reference=batch.reference,
                scheme=batch.scheme, sequence=batch.sequence, structure=batch.structure,
                x_t=x_input, x_mask=x_mask, t_emb=t_emb)

        train_loss = float(client.diffuser.cal_loss(
            x0=x0, x_input=x_input, x_update=atom_pos_update, sigma=sigma, mask=x_mask))
        x_pred = client._edm_x0_hat(x_input, atom_pos_update.float(), sigma)  # noqa: SLF001
        in_scale = scheduler.input_scale(sigma).to(x_input.dtype)
        noisy_x = x_input / in_scale

        pred = flat_LL(x_pred).float()
        gt = flat_LL(x0).float()
        nx = flat_LL(noisy_x).float()
        mk = x_mask.reshape(-1, x_mask.shape[-1])[0]
        noisy_lddt = float(metrics.cal_atom_lddt(nx, gt, mk))
        pred_lddt = float(metrics.cal_atom_lddt(pred, gt, mk))
        pred_rmsd = float(metrics.cal_aligned_rmsd(pred, gt, mk))
        print(f"{sval:8.2f} {float(in_scale.mean()):9.4f} "
              f"{float(scheduler.loss_weight(sigma).mean()):9.3f} {train_loss:11.5f} "
              f"{noisy_lddt:10.4f} {pred_lddt:10.4f} {pred_rmsd:10.3f}")
        batch_to_cif(batch, pred.unsqueeze(0), cif_dir / f"{target}_denoised_sigma{sval:g}.cif")

    print(f"structures -> {cif_dir}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
