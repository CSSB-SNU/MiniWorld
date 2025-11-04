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
def train(
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

    from MiniWorld.models.af3_psk_2 import AF3Client

    # Load client
    if resume_from_ckpt is None:
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot found config file: {config_path}")
        config = OmegaConf.load(config)
        config = AF3Client.Config(**config)
        client = AF3Client(config, name="MiniWorld")
    else:
        ckpt_path = Path(resume_from_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Cannot found checkpoint file: {ckpt_path}")
        client = AF3Client.from_checkpoint(ckpt_path)

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Setup wandb
    if w and client.is_distributed:
        client.add_logger(
            WandbLogger("-".join([client.name, client.config.experiment.comment]))
        )

    if seed is not None:
        set_seed(seed)
        client.log_message(f"Set random seed: {seed}")

    ckpt_dir_path = Path(ckpt_dir)
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir_path / f"{client.name}.pt"

    train_preprocessing_config = BioMolPreProcessing.Config(
        meta=client.config.data.meta.model_dump(),
        pipeline=client.config.data.train.model_dump(),
        mol_types=client.config.data.mol_types.model_dump(),
    )
    valid_preprocessing_config = BioMolPreProcessing.Config(
        meta=client.config.data.meta.model_dump(),
        pipeline=client.config.data.valid.model_dump(),
        mol_types=client.config.data.mol_types.model_dump(),
    )

    train_data_config = BioMolData.BioMolConfig(
        crop_config=client.config.data.crop.model_dump(),
        mol_types=client.config.data.mol_types.model_dump(),
        msa_config=client.config.data.msa.model_dump(),
        data_preprocessing_config=train_preprocessing_config,
    )
    valid_data_config = BioMolData.BioMolConfig(
        crop_config=client.config.data.crop.model_dump(),
        mol_types=client.config.data.mol_types.model_dump(),
        msa_config=client.config.data.msa.model_dump(),
        data_preprocessing_config=valid_preprocessing_config,
    )

    # always ddp
    if client.config.experiment.prefetch_factor == 0:
        prefetch_factor = None
    else:
        prefetch_factor = client.config.experiment.prefetch_factor
    train_loader = BioMolData(train_data_config).create_ddp_dataloader(
        rank=client.local_rank,
        world_size=world_size,
        drop_last=True,
        batch_size=client.config.experiment.num_batch,
        num_workers=client.config.experiment.num_workers,
        prefetch_factor=prefetch_factor,
    )
    valid_loader = BioMolData(valid_data_config).create_ddp_dataloader(
        rank=client.local_rank,
        world_size=world_size,
        drop_last=False,
        batch_size=client.config.experiment.num_batch,  # or 1
        num_workers=0,
    )

    client.log_message("-" * 70)
    client.log_message("")
    client.log_message("Start training".center(70))
    client.log_message("")
    client.log_message("-" * 70)

    client.add_callback(SaveCheckpointPeriodic(ckpt_dir, every_n_epochs=10))
    train_num_item = client.config.experiment.train_item // world_size
    valid_num_item = client.config.experiment.valid_item // world_size
    for epoch in range(client.epoch, client.config.experiment.num_epoch):
        train_loader.sampler.set_epoch(epoch)
        client.training_epoch(train_loader, train_num_item)
        if (client.epoch - 1) % client.config.experiment.eval_freq == 0:
            valid_loader.sampler.set_epoch(epoch)
            client.validation_epoch(valid_loader, valid_num_item)


@cli.command()
@click.option(
    "--ckpt", type=click.Path(exists=True), help="checkpoint file", required=True
)
@click.option("--num_sample", type=int, default=1, help="number of samples")
@click.option("--timesteps", type=int, default=100, help="number of timesteps")
@click.option("--out_dir", type=click.Path(), default="output/", help="output dir")
@click.option("--device", default="cuda", type=str, help="device to use")
@add_slurm_options
def inference(
    ckpt: str,
    num_sample: int = 5,
    timesteps: int = 100,
    out_dir: str = "outputs",
    device: str = "cuda",
    slurm: bool = False,
    mem: str = "32G",
    cpus: int = 8,
    gpus: str = "A6000:1",
):
    """Inference mode for AF3."""
    if slurm:
        job_name = Path(ckpt).stem
        path = submit_to_slurm(
            f"pixi run python {__file__} inference --ckpt {ckpt}"
            f" --num_sample {num_sample} --timesteps {timesteps}"
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

    ckpt_path = Path(ckpt)
    # ckpt_path2 = Path("checkpoints/MiniWorld_epoch=0000.pt")
    client = AF3Client.from_checkpoint(ckpt_path)
    # client_epoch0 = AF3Client.from_checkpoint(ckpt_path2)

    valid_preprocessing_config = BioMolPreProcessing.Config(
        meta=client.config.data.meta,
        pipeline=client.config.data.valid,
    )

    valid_data_config = BioMolData.BioMolConfig(
        crop_config=client.config.data.crop,
        mol_types=client.config.data.mol_types,
        msa_config=client.config.data.msa,
        data_preprocessing_config=valid_preprocessing_config,
    )

    valid_loader = BioMolData(valid_data_config).create_ddp_dataloader(
        rank=0,
        world_size=1,
        drop_last=False,
        batch_size=client.config.experiment.num_batch,  # or 1
        num_workers=0,
    )

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    client = client.to(device=device)

    for ii, batch in enumerate(valid_loader):
        batch = batch.to(device=device, dtype=client.config.model.precision.input)
        print(f"batch.name: {batch.name[0]}")

        af3_inference_output = client.inference(
            batch=batch,
            timesteps=timesteps,
        )

        true_mmcif_path = f"{out_dir_path}/{batch.name[0]}_true.mmcif"
        denoised_mmcif_path = f"{out_dir_path}/{batch.name[0]}_denoised.mmcif"
        to_mmcif(
            af3_inference_output.batch,
            af3_inference_output.atom_pos_pred,
            true_mmcif_path,
            denoised_mmcif_path,
        )
        if ii >= num_sample - 1:
            break


if __name__ == "__main__":
    cli()
