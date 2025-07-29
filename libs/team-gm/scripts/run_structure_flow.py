import time
import pickle
import torch
import click
import subprocess
import random
import numpy as np

from omegaconf import OmegaConf
from pathlib import Path

from team_gm.loggers import WandbLogger, FileLogger
from team_gm.callbacks import SaveCheckpointBest
from team_gm.utils.data_utils import write_prot_pdb
from team_gm.data.dataloader import ProteinData


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@click.group()
def cli():
    pass


def add_slurm_options(func):
    func = click.option(
        "--slurm",
        is_flag=True,
        help="Submit job to SLURM",
    )(func)
    func = click.option(
        "--job-name",
        type=str,
        help="Job name",
    )(func)
    func = click.option(
        "--partition",
        type=str,
        help="Partition name",
    )(func)
    func = click.option(
        "--mem",
        default="32G",
        show_default=True,
        type=str,
        help="Memory per node",
    )(func)
    func = click.option(
        "--cpus-per-task",
        default=8,
        show_default=True,
        type=int,
        help="CPUs per task",
    )(func)
    func = click.option(
        "--gpus-per-node",
        type=str,
        help="GPUs per node (e.g., 'H100:1')",
    )(func)
    func = click.option(
        "--ntasks-per-node",
        default=1,
        show_default=True,
        type=int,
        help="Number of tasks per node",
    )(func)
    func = click.option(
        "--time",
        type=str,
        help="Time limit (e.g., 1:00:00)",
    )(func)
    func = click.option(
        "--nodelist",
        type=str,
        help="Request specific nodes (e.g., 'node001,node002' or 'node[001-004]')",
    )(func)
    return func


def submit_to_slurm(command: str, log_dir: str = "logs", **slurm_kwargs) -> None:
    ESSENTIAL_KEYS = [
        "job_name",
        "partition",
        "gpus_per_node",
        "mem",
        "cpus_per_task",
        "ntasks_per_node",
    ]
    for key in ESSENTIAL_KEYS:
        if key not in slurm_kwargs or slurm_kwargs[key] is None:
            raise ValueError(f"Missing required SLURM option: {key}")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm_kwargs['job_name']}",
        f"#SBATCH --partition={slurm_kwargs['partition']}",
        f"#SBATCH --mem={slurm_kwargs['mem']}",
        f"#SBATCH --cpus-per-task={slurm_kwargs['cpus_per_task']}",
        f"#SBATCH --ntasks-per-node={slurm_kwargs['ntasks_per_node']}",
        f"#SBATCH --gpus-per-node={slurm_kwargs['gpus_per_node']}",
        f"#SBATCH --output={log_dir}/{slurm_kwargs['job_name']}_%j.out",
    ]
    if slurm_kwargs.get("time"):
        script_lines.append(f"#SBATCH --time={slurm_kwargs['time']}")
    if slurm_kwargs.get("nodelist"):
        script_lines.append(f"#SBATCH --nodelist={slurm_kwargs['nodelist']}")
    script_lines.append("")

    script_lines.extend(
        [
            "if [ $SLURM_NNODES -gt 1 ]; then",
            "    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)",  # noqa: E501
            "    export MASTER_PORT=$((${SLURM_JOB_ID: -4} + 15000))",
            "    pixi run torchrun \\",
            "        --nproc_per_node=$SLURM_NTASKS_PER_NODE \\",
            "        --nnodes=$SLURM_NNODES \\",
            "        --node_rank=$SLURM_NODEID \\",
            "        --master_addr=$MASTER_ADDR \\",
            "        --master_port=$MASTER_PORT \\",
            f"        {command}",
        ]
    )
    script_lines.extend(
        [
            "elif [ $SLURM_NTASKS_PER_NODE -gt 1 ]; then",
            "    pixi run torchrun \\",
            "        --nproc_per_node=$SLURM_NTASKS_PER_NODE \\",
            "        --standalone \\",
            f"        {command}",
        ]
    )
    script_lines.extend(
        [
            "else",
            f"    pixi run python {command}",
            "fi",
        ]
    )

    path = Path(log_dir) / f"{slurm_kwargs['job_name']}.sh"
    path.write_text("\n".join(script_lines))

    result = subprocess.run(["sbatch", path], capture_output=True, text=True)
    output = result.stdout.strip()
    if result.returncode != 0:
        click.echo(f"❌ Failed to submit Slurm job: {slurm_kwargs['job_name']} ({path})")
        raise RuntimeError(f"Slurm submission failed: {result.stderr}")
    elif not output.startswith("Submitted batch job"):
        click.echo(
            f"❌ Unexpected output from Slurm: {output} ({path}). "
            "Expected format: 'Submitted batch job <job_id>'"
        )
        raise RuntimeError(f"Unexpected Slurm output: {output}")
    else:
        job_id = output.split()[-1]
        click.echo(
            f"✅ Submitted Slurm job {job_id}: {slurm_kwargs['job_name']} ({path})"
        )


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="config file")
@click.option("--resume_from_ckpt", type=click.Path(exists=True), help="checkpoint file")
@click.option("-w", is_flag=True, help="Use wandb for logging")
@click.option("--ckpt_dir", default="checkpoints/", help="dir for save checkpoint")
@click.option("--seed", type=int, help="random seed")
@add_slurm_options
def train(
    config: str | None,
    resume_from_ckpt: str | None,
    w: bool,
    ckpt_dir: str,
    seed: int | None,
    **slurm_kwargs,
):
    """Train StructureFlow model."""
    if (config or resume_from_ckpt) is None:
        raise ValueError("You must provide either a config file or a checkpoint file.")
    if config and resume_from_ckpt:
        raise ValueError("You cannot provide both a config file and a checkpoint file.")

    if slurm_kwargs.get("slurm"):
        if config:
            job_name = Path(config).stem
            command = f"{__file__} train --config {config}"
        else:
            job_name = Path(resume_from_ckpt).stem
            command = f"{__file__} train --resume_from_ckpt {resume_from_ckpt}"
        if slurm_kwargs.get("job_name") is None:
            slurm_kwargs["job_name"] = job_name
        command += f" --ckpt_dir {ckpt_dir}"
        if seed is not None:
            command += f" --seed {seed}"
        if w:
            command += " -w"
        submit_to_slurm(command, **slurm_kwargs)
        return
    from team_gm.models.structure_flow import StructureFlowClient

    # Load client
    if resume_from_ckpt is None:
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot found config file: {config_path}")
        config = OmegaConf.load(config)
        config = StructureFlowClient.Config(**config)
        client = StructureFlowClient(config, name=config_path.stem)
    else:
        ckpt_path = Path(resume_from_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Cannot found checkpoint file: {ckpt_path}")
        client = StructureFlowClient.from_checkpoint(ckpt_path)
        client.log_message("-" * 70)
        client.log_message("")
        client.log_message("Finetuning model".center(70))
        client.log_message(
            f"Load pretrain weight: {ckpt_path} ({client.epoch} epoch)".center(70)
        )
        client.log_message("")
        client.log_message("-" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if w:
        client.add_logger(
            WandbLogger("-".join([client.name, client.config.experiment.comment]))
        )
    client.add_logger(FileLogger(f"logs/{client.name}.log"))

    if seed is not None:
        set_seed(seed)
        client.log_message(f"Set random seed: {seed}")

    with open(client.config.data.data_pkl_path, "rb") as f:
        pdb_ids, pdb_weights, train_pdb, valid_pdb = pickle.load(f)

    train_dataset = ProteinData(
        item_dict=train_pdb,
        input_dir_path=client.config.data.input_dir_path,
        crop_length=client.config.experiment.crop_length,
    )

    valid_pdb = {
        k: v
        for i, (k, v) in enumerate(valid_pdb.items())
        if i < client.config.experiment.eval_input_num
    }
    valid_dataset = ProteinData(
        item_dict=valid_pdb,
        input_dir_path=client.config.data.input_dir_path,
        crop_length=client.config.experiment.eval_crop_length,
    )

    if not client.is_distributed:
        client = client.to(device="cuda")
        train_sampler = None
        valid_sampler = None
    else:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, shuffle=False)

    train_dataloader = train_dataset.create_dataloader(
        batch_size=client.config.experiment.num_batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    valid_dataloader = valid_dataset.create_dataloader(
        sampler=valid_sampler,
    )

    client.log_message("-" * 70)
    client.log_message("")
    client.log_message("Start training".center(70))
    client.log_message("")
    client.log_message("-" * 70)

    client.add_callback(SaveCheckpointBest(ckpt_dir))
    while client.epoch < client.config.experiment.num_epoch:
        client.training_epoch(train_dataloader)
        if (client.epoch - 1) % client.config.experiment.eval_freq == 0:
            client.validation_epoch(valid_dataloader)


@cli.command()
@click.option(
    "--ckpt", type=click.Path(exists=True), help="checkpoint file", required=True
)
@click.option("--seq", "-s", type=str, help="Query sequence", required=True)
@click.option("--num_sample", type=int, default=5, help="number of samples")
@click.option("--timesteps", type=int, default=100, help="number of timesteps")
@click.option("--out_dir", type=click.Path(), default="output/", help="output dir")
@add_slurm_options
def inference(
    ckpt: str,
    seq: str,
    num_sample: int,
    timesteps: int,
    out_dir: str,
    **slurm_kwargs,
):
    """Inference mode for StructureFlow."""
    if slurm_kwargs.get("slurm"):
        if slurm_kwargs.get("job_name") is None:
            slurm_kwargs["job_name"] = Path(ckpt).stem
        submit_to_slurm(
            f"{__file__} inference --ckpt {ckpt} --seq {seq}"
            f" --num_sample {num_sample} --timesteps {timesteps}"
            f" --out_dir {out_dir}",
            **slurm_kwargs,
        )
        return
    from team_gm.models.structure_flow import StructureFlowClient

    ckpt_path = Path(ckpt)
    client = StructureFlowClient.from_checkpoint(ckpt_path)

    if torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not client.is_distributed:
        client = client.to(device="cuda")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    client.log_message("-" * 70)
    client.log_message("")
    client.log_message("Start inference".center(70))
    client.log_message("")
    client.log_message("-" * 70)

    start = time.time()
    output = client.sample(seq, num_sample=num_sample, timesteps=timesteps)
    atom_mask = output.build_atom_mask()
    client.log_message(
        f"Sampled {num_sample} structures in {time.time() - start:.2f} seconds"
    )

    write_prot_pdb(
        output.atom_pos_pred,
        atom_mask=atom_mask,
        seq_idx=output.sequence.seq_idx,
        res_names=output.sequence.res_name,
        file_path=out_dir_path / "sample.pdb",
    )

    write_prot_pdb(
        output.inter_traj[0],
        atom_mask=atom_mask[0],
        seq_idx=output.sequence.seq_idx[0],
        res_names=output.sequence.res_name[0],
        file_path=out_dir_path / "inter_traj.pdb",
    )

    write_prot_pdb(
        output.model_traj[0],
        atom_mask=atom_mask[0],
        seq_idx=output.sequence.seq_idx[0],
        res_names=output.sequence.res_name[0],
        file_path=out_dir_path / "model_traj.pdb",
    )


if __name__ == "__main__":
    cli()
