import torch
import click
import tempfile
import subprocess
import random
import numpy as np

from omegaconf import OmegaConf
from pathlib import Path

from MiniWorld.data.dataloader_BioMol import (
    BioMolData,
    BioMolPreProcessing,
    to_mmcif,
)
from MiniWorld.validate.lddt import category_lddt
from BioMol.BioMol import BioMol

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
    from MiniWorld.models.af3_psk_2 import AF3Client

    if torch.device(device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")


    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot found config file: {config_path}")
    config = OmegaConf.load(config)

    ckpt_path = Path(ckpt)
    client = AF3Client.from_checkpoint(ckpt_path)

    valid_preprocessing_config = BioMolPreProcessing.Config(
        meta=config.data.meta,
        pipeline=config.data.valid,
        mol_types=config.data.mol_types,
    )
    valid_data_config = BioMolData.BioMolConfig(
        crop_config=config.data.crop, # full cropping
        mol_types=config.data.mol_types,
        msa_config=config.data.msa,
        data_preprocessing_config=valid_preprocessing_config,
    )

    valid_loader = BioMolData(valid_data_config).create_ddp_dataloader(
        rank=0,
        world_size=1,
        drop_last=False,
        batch_size=config.experiment.num_batch,  # or 1
        num_workers=16,
        prefetch_factor=32,
    )

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    client = client.to(device=device)

    lddt_dict = {
        'intra_protein' : [[],0], # 0
        'intra_DNA' : [[],0], # 1
        'intra_RNA' : [[],0], # 2
        'intra_ligand' : [[],0], # 3
        'protein-protein' : [[],0], # 0
        'protein-DNA' : [[],0], # 1
        'protein-RNA' : [[],0], # 2
        'protein-ligand' : [[],0], # 3
        'total' : [[],0], # 4
    }
    max_len = config.data.max_len if 'max_len' in config.data else 1024

    for ii, batch in enumerate(valid_loader):
        batch = batch.to(device=device)
        residue_type = batch.sequence.residue_type[0]
        na_included = ((residue_type >= 21) & (residue_type <= 30)).any()
        if not na_included :
            print(f"Skip {batch.name[0]} without NA")
            continue
        if batch.residue_length > max_len or batch.atom_length < 30:
            print(f"Skip {batch.name[0]} with length {batch.residue_length}")
            continue

        try:
            af3_inference_output = client.inference(
                batch=batch,
                timesteps=timesteps,
            )
        except:
            breakpoint()

        true_mmcif_path = f"{out_dir_path}/{batch.name[0]}_true.mmcif"
        denoised_mmcif_path = f"{out_dir_path}/{batch.name[0]}_denoised.mmcif"
        to_mmcif(
            af3_inference_output.batch,
            af3_inference_output.atom_pos_pred,
            true_mmcif_path,
            denoised_mmcif_path,
        )
        
        pred_atom_pos = af3_inference_output.atom_pos_pred
        try:
            _lddt_dict = category_lddt(
                batch=batch,
                pred_atom_pos=pred_atom_pos[0],
            )
        except Exception as e:
            print(f"Error in lddt calculation for {batch.name[0]}: {e}")
            continue

        for key in lddt_dict.keys():
            if _lddt_dict[key] is not None:
                lddt_dict[key][0].append(_lddt_dict[key][0])
                lddt_dict[key][1] += _lddt_dict[key][1]

        if ii % 1000 == 0 :
            np.save(out_dir_path / f"lddt_dict_{ii}.npy", lddt_dict)


    # save final results
    np.save(out_dir_path / "lddt_dict.npy", lddt_dict)

    def average(key):
        return (
            float(np.mean(lddt_dict[key][0]))
            if len(lddt_dict[key][0]) > 0
            else None
        )

    intra_protein_lddt = average('intra_protein')
    intra_DNA_lddt = average('intra_DNA')
    intra_RNA_lddt = average('intra_RNA')
    intra_ligand_lddt = average('intra_ligand')
    protein_protein_lddt = average('protein-protein')
    protein_DNA_lddt = average('protein-DNA')
    protein_RNA_lddt = average('protein-RNA')
    protein_ligand_lddt = average('protein-ligand')
    print(
        f"intra_protein_lddt: {intra_protein_lddt} ({lddt_dict['intra_protein'][1]})"
    )
    print(f"intra_DNA_lddt: {intra_DNA_lddt} ({lddt_dict['intra_DNA'][1]})")
    print(f"intra_RNA_lddt: {intra_RNA_lddt} ({lddt_dict['intra_RNA'][1]})")
    print(f"intra_ligand_lddt: {intra_ligand_lddt} ({lddt_dict['intra_ligand'][1]})")
    print(
        f"protein_protein_lddt: {protein_protein_lddt} ({lddt_dict['protein-protein'][1]})"
    )
    print(f"protein_DNA_lddt: {protein_DNA_lddt} ({lddt_dict['protein-DNA'][1]})")
    print(f"protein_RNA_lddt: {protein_RNA_lddt} ({lddt_dict['protein-RNA'][1]})")
    print(f"protein_ligand_lddt: {protein_ligand_lddt} ({lddt_dict['protein-ligand'][1]})")

if __name__ == "__main__":
    cli()
