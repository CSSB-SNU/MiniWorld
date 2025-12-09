import torch
import click
import tempfile
import subprocess
import random
import numpy as np

from omegaconf import OmegaConf
from pathlib import Path
import os

from team_gm.loggers import WandbLogger
from team_gm.callbacks import SaveCheckpointPeriodic
from MiniWorld.data.dataloader_BioMol import (
    BioMolData,
    BioMolPreProcessing,
    to_mmcif,
)
from BioMol.BioMol import BioMol
from BioMol.utils.hierarchy import MoleculeType, PolymerType


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# torch.set_float32_matmul_precision("high")

# anomaly detection
torch.autograd.set_detect_anomaly(False)


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
    "--resume_from_ckpt", type=click.Path(exists=True), help="checkpoint file"
)
@click.option("-w", is_flag=True, help="Use wandb for logging")
@click.option("--ckpt_dir", default="checkpoints/", help="dir for save checkpoint")
@click.option("--device", default="cuda", type=str, help="device to use")
@click.option(
    "--local_rank",
    default=lambda: int(os.environ.get("LOCAL_RANK", 0)),
    type=int,
    help="DDP local rank (torchrun sets this)",
)
@add_slurm_options
def analysis(
    config: str | None = None,
    resume_from_ckpt: str | None = None,
    w: bool = False,
    ckpt_dir: str = "checkpoints/",
    device: str = "cuda",
    seed: int | None = 1123,
    slurm: bool = False,
    **slurm_kwargs,
):
    """Train AF3 model."""
    if (config or resume_from_ckpt) is None:
        raise ValueError("You must provide either a config file or a checkpoint file.")
    if config and resume_from_ckpt:
        raise ValueError("You cannot provide both a config file and a checkpoint file.")

    if slurm:
        if config:
            job_name = Path(config).stem
            command = f"pixi run python -u {__file__} train --config {config}"
        else:
            job_name = Path(resume_from_ckpt).stem
            command = f"pixi run python {__file__} train --resume_from_ckpt {resume_from_ckpt}"
        command += f" --ckpt_dir {ckpt_dir}"
        command += f" --device {device}"
        if seed is not None:
            command += f" --seed {seed}"
        if w:
            command += " -w"
        path = submit_to_slurm(command, job_name=job_name, **slurm_kwargs)
        click.echo(f"✅ Submitted Slurm job: {job_name} ({path})")
        return


    # Load client
    if resume_from_ckpt is None:
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot found config file: {config_path}")
        config = OmegaConf.load(config)
    else:
        ckpt_path = Path(resume_from_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Cannot found checkpoint file: {ckpt_path}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ckpt_dir_path = Path(ckpt_dir)
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)

    valid_preprocessing_config = BioMolPreProcessing.Config(
        meta=config.data.meta,
        pipeline=config.data.valid,
        mol_types=config.data.mol_types,
    )

    valid_data_config = BioMolData.BioMolConfig(
        crop_config=config.data.crop,
        mol_types=config.data.mol_types,
        msa_config=config.data.msa,
        data_preprocessing_config=valid_preprocessing_config,
    )

    # always ddp
    if config.experiment.prefetch_factor == 0:
        prefetch_factor = None
    else:
        prefetch_factor = config.experiment.prefetch_factor
    valid_data = BioMolData(valid_data_config)

    biomol_ID_list = []
    for idx in range(len(valid_data.preprocessing.items)):
        id_list = list(valid_data.preprocessing.items[idx].values())
        # id_list = [x for sub in id_list for x in sub]
        id_list = [sub[0] for sub in id_list]
        biomol_ID_list.extend(id_list)

    node_frequency = {
        'protein' : 0, # 0
        'DNA' : 0, # 1
        'RNA' : 0, # 2
        'ligand' : 0, # 3
    }
    edge_frequency = {
        'protein-protein' : 0, # 0
        'protein-DNA' : 0, # 1
        'protein-RNA' : 0, # 2
        'protein-ligand' : 0, # 3
    }

    len_cutoff = 1024

    for biomol_id in biomol_ID_list:

        _node_freq = {
            'protein' : 0, # 0
            'DNA' : 0, # 1
            'RNA' : 0, # 2
            'ligand' : 0, # 3
        }
        _edge_freq = {
            'protein-protein' : 0, # 0
            'protein-DNA' : 0, # 1
            'protein-RNA' : 0, # 2
            'protein-ligand' : 0, # 3
        }

        pdb_ID, assembly_ID, model_ID, alt_ID = biomol_id.split('_')
        biomol = BioMol(pdb_ID=pdb_ID)
        biomol.choose(assembly_ID, model_ID, alt_ID)

        entity_type = []
        entity_list = biomol.structure.entity_list
        for entity in entity_list:
            if entity.get_type() == MoleculeType.POLYMER:
                polyer_type = entity.get_polymer_type()
                match polyer_type:
                    case PolymerType.PROTEIN:
                        entity_type.append(0)
                    case PolymerType.DNA:
                        entity_type.append(1)
                    case PolymerType.RNA:
                        entity_type.append(2)
                    case _:
                        entity_type.append(3)
            else:
                entity_type.append(3)
        for _type in entity_type:
            match _type:
                case 0:
                    _node_freq['protein'] = 1
                case 1:
                    _node_freq['DNA'] = 1
                case 2:
                    _node_freq['RNA'] = 1
                case 3:
                    _node_freq['ligand'] = 1

        contact_edges = biomol.structure.contact_graph.graphs[(assembly_ID,model_ID,alt_ID)]['edges']
        for edge in contact_edges:
            node1_type = entity_type[edge[0]]
            node2_type = entity_type[edge[1]]
            if node1_type == 0:
                match node2_type:
                    case 0:
                        _edge_freq['protein-protein'] = 1
                    case 1:
                        _edge_freq['protein-DNA'] = 1
                    case 2:
                        _edge_freq['protein-RNA'] = 1
                    case 3:
                        _edge_freq['protein-ligand'] = 1
            elif node2_type == 0:
                match node1_type:
                    case 1:
                        _edge_freq['protein-DNA'] = 1
                    case 2:
                        _edge_freq['protein-RNA'] = 1
                    case 3:
                        _edge_freq['protein-ligand'] = 1

        for key in node_frequency.keys():
            node_frequency[key] += _node_freq[key]
        for key in edge_frequency.keys():
            edge_frequency[key] += _edge_freq[key]

    print("Node Frequency:")
    print(node_frequency)
    print("Edge Frequency:")
    print(edge_frequency)


if __name__ == "__main__":
    cli()
