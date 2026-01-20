from pathlib import Path
import torch
import click
import tempfile
import subprocess
import random
import numpy as np

from omegaconf import OmegaConf
import copy

from team_gm.utils import metrics
from MiniWorld.data.dataloader.dataloader_multistate import (
    BioMolMonomerData,
)
from MiniWorld.data.features.features_multistate import Batch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


@click.group()
def cli():
    pass


def add_slurm_options(func):
    slurm_options = [
        click.option("--slurm", is_flag=True, help="Run on SLURM"),
        click.option("--mem", default="32G", type=str),
        click.option("--cpus", default=8, type=int),
        click.option("--gpus", default="A6000:1", type=str),
    ]
    for option in reversed(slurm_options):
        func = option(func)
    return func


def submit_to_slurm(command, job_name, mem, cpus, gpus) -> str:
    script = (
        f"#!/bin/bash\n"
        "#SBATCH -p gpu\n"
        f"#SBATCH -J {job_name}\n"
        f"#SBATCH -c {cpus}\n"
        f"#SBATCH --mem={mem}\n"
        f"#SBATCH --gres=gpu:{gpus}\n"
        f"#SBATCH -o {job_name}.log\n\n"
        f"{command}"
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sh") as f:
        f.write(script)
        path = f.name
    subprocess.run(["sbatch", path])
    return path


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="config file")
@click.option(
    "--ckpt", type=click.Path(exists=True), help="checkpoint file", required=True
)
@click.option("--timesteps", type=int, default=100, help="number of timesteps")
@click.option("--out_dir", type=click.Path(), default="output/", help="output dir")
@click.option("--device", default="cuda", type=str, help="device to use")
@add_slurm_options
def validate(
    config: str,
    ckpt: str,
    timesteps: int = 100,
    out_dir: str = "outputs",
    device: str = "cuda",
    slurm: bool = False,
    mem: str = "32G",
    cpus: int = 8,
    gpus: str = "A6000:1",
):
    """Validation mode for AF3."""
    if slurm:
        job_name = Path(ckpt).stem
        path = submit_to_slurm(
            f"pixi run python {__file__} validate --ckpt {ckpt}"
            f" --timesteps {timesteps}"
            f" --out_dir {out_dir} --device {device}",
            job_name=job_name,
            mem=mem,
            cpus=cpus,
            gpus=gpus,
        )
        click.echo(f"✅ Submitted Slurm job: {job_name} ({path})")
        return
    from MiniWorld.models.af3_psk_multistate_monomer import AF3Client

    if torch.device(device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")


    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot found config file: {config_path}")
    config = OmegaConf.load(config)

    ckpt_path = Path(ckpt)
    client = AF3Client.from_checkpoint(ckpt_path)

    valid_data_config = BioMolMonomerData.BioMolConfig(
        crop_config=client.config.data.crop,
        msa_config=client.config.data.msa,
        kmer_fast_align_config = client.config.data.kmer_fast_align,
        multistate_config = client.config.data.multistate,
        preprocess_config=client.config.data.valid_preprocessing,
    )

    valid_data = BioMolMonomerData(valid_data_config)

    valid_loader = valid_data.create_dataloader(
        drop_last=False,
        batch_size=config.experiment.num_batch,  # or 1
        num_workers=1,
        # prefetch_factor=0,
    )

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    client = client.to(device=device)

    max_distance = 15.0
    distance_bins = (0.5, 1.0, 2.0, 4.0)
    lddt_results = {}

    print(f"Starting validation on {len(valid_loader)} batches.")

    for ii, batch in enumerate(valid_loader):
        batch : Batch = batch.to(device=device)
        af3_inference_output = client.inference(
            batch=batch,
            timesteps=timesteps,
        )
        true_mmcif_path = out_dir_path / f"{batch.name[0]}_true.mmcif"
        batchID = batch.name[0]
        queryID, query_cifmol_ID = batchID.split("_")
        cifmols = valid_data.load_cifmols(queryID)
        cifmol = [c for c in cifmols if c.id[0] == query_cifmol_ID][0]
        crop_indices = batch.scheme.crop_indices[0].cpu().numpy()
        cifmol = cifmol.residues[crop_indices].extract()
        cifmol.to_cif(Path(true_mmcif_path))

        # exchange strcuture
        # gt_pos = batch.structure.atom_pos # (B, N_str, L_atom, 3)
        gt_pos = cifmol.atoms.xyz.value
        true_atom_pos_mask = cifmol.atoms.xyz.value
        true_atom_pos_mask = np.isfinite(true_atom_pos_mask).any(-1)
        atom_pos_pred = af3_inference_output.atom_pos_pred[:,0] # (N_str, L_atom, 3)
        N_str = atom_pos_pred.shape[0]
        lddt_list = []
        for n_str in range(N_str):
            denoised_mmcif_path = out_dir_path / f"{batch.name[0]}_denoised_{n_str}.mmcif"
            cifmol_dict = copy.deepcopy(cifmol.to_dict())
            atom_pos_pred = af3_inference_output.atom_pos_pred[n_str,0].cpu().numpy()
            atom_pos_pred[~true_atom_pos_mask] = np.nan
            cifmol_dict["atoms"]["nodes"]["xyz"]["value"]= atom_pos_pred

            cifmol_denoised = type(cifmol).from_dict(cifmol_dict)
            cifmol_denoised.to_cif(denoised_mmcif_path)

            lddt = metrics.cal_atom_lddt(
                pred_atom_pos=atom_pos_pred,
                gt_atom_pos=gt_pos,
                atom_mask=true_atom_pos_mask,
                max_distance=max_distance,
                distance_bins=distance_bins,
            )
            lddt_list.append(lddt)
            breakpoint()


if __name__ == "__main__":
    cli()
