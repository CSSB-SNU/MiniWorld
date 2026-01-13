import datetime
import json
import logging
from pathlib import Path

import click
import numpy as np
import torch
from lightning import Fabric
from omegaconf import OmegaConf
from team_gm.utils.script_utils import MetricsAggregator, set_seed

import wandb
from miniworld.data.dataloader.dataloader_edge_backprop import (
    BioMolData,
)
from miniworld.data.to_cif import batch_to_cif
from miniworld.loss import metrics  # , losses
from miniworld.models.cmap_template_af3 import AF3Client
from miniworld.utils.structure.distance import get_contact_map

# torch.set_float32_matmul_precision("high")  # noqa: ERA001
# anomaly detection
torch.autograd.set_detect_anomaly(False)


def setup_logger(client: AF3Client) -> None:
    if not client.is_global_zero:
        return

    client.logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    client.logger.addHandler(handler)

    now = datetime.datetime.now(datetime.timezone.utc)
    file_handler = logging.FileHandler(
        f"logs/cmap_template_af3/cmap_template_af3_{now:%Y%m%d_%H%M%S}.log",
    )
    file_handler.setFormatter(formatter)
    client.logger.addHandler(file_handler)


def get_step_decay_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 1e3,
    decay_steps: int = 5e4,
    decay_factor: float = 0.95,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Return a LambdaLR scheduler that
    1) linearly warms up from 0 → 1 over the first `warmup_steps`
    2) thereafter, multiplies the lr by `decay_factor` every `decay_steps`
    The scheduler multiplies the optimizer's base_lr by the returned factor.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # warmup: 0 -> 1
            return step / float(warmup_steps)
        # step decay: factor ** floor((step - warmup_steps) / decay_steps)
        num_decays = (step - warmup_steps) // decay_steps
        return decay_factor**num_decays

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="config file",
)
@click.option(
    "--resume-from-ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="checkpoint file",
)
@click.option(
    "-w",
    is_flag=True,
    help="Use wandb for logging",
)
@click.option(
    "--ckpt-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default="checkpoints/",
    help="dir for save checkpoint",
)
@click.option(
    "--seed",
    type=int,
    help="random seed",
)
def train(  # noqa: PLR0912, PLR0915
    config: Path | None,
    resume_from_ckpt: Path | None,
    w: bool,
    ckpt_dir: Path,
    seed: int | None,
):
    if config and not resume_from_ckpt:
        cfg = OmegaConf.load(config)
        cfg = AF3Client.Config.model_validate(cfg)
        client = AF3Client(cfg)
    elif not config and resume_from_ckpt:
        client = AF3Client.from_checkpoint(resume_from_ckpt)
    else:
        msg = (
            "You must provide either a config file or a checkpoint file, but not both."
        )
        raise ValueError(msg)

    fabric = Fabric()
    fabric.launch()

    optimizer = torch.optim.AdamW(
        client.model.parameters(),
        client.config.experiment.max_lr,
    )
    scheduler = get_step_decay_scheduler_with_warmup(
        optimizer=optimizer,
        warmup_steps=client.config.experiment.warmup_steps,
        decay_steps=client.config.experiment.decay_steps,
        decay_factor=client.config.experiment.decay_factor,
    )

    client.setup(
        fabric=fabric,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation_steps=client.config.experiment.grad_accum_steps,
        gradient_clip_norm=client.config.experiment.grad_clip_max_norm,
    )
    setup_logger(client)

    if resume_from_ckpt is not None:
        client.load_optimizer_state(resume_from_ckpt)
        client.logger.info(
            "Load pretrain weight: %s (%d epoch)",
            resume_from_ckpt.name,
            client.epoch,
        )

    if w and client.is_global_zero:
        wandb.init(name=client.config.experiment.comment)
        wandb.config.update(client.config.model_dump())
    msg = f"Config:\n{json.dumps(client.config.model_dump(), indent=4, default=str)}"
    client.logger.info(msg)

    if seed is not None:
        set_seed(seed)
        client.logger.info("Set random seed: %d", seed)

    train_data_config = BioMolData.BioMolConfig(
        crop_config=client.config.data.crop.model_dump(),
        msa_config=client.config.data.msa.model_dump(),
        DB_config=client.config.data.train_db.model_dump(),
        edge_weight_config=client.config.data.edge_weight.model_dump(),
    )
    valid_data_config = BioMolData.BioMolConfig(
        crop_config=client.config.data.crop.model_dump(),
        msa_config=client.config.data.msa.model_dump(),
        DB_config=client.config.data.valid_db.model_dump(),
        edge_weight_config=client.config.data.edge_weight.model_dump(),
    )

    prefetch_factor = (
        None
        if client.config.experiment.prefetch_factor == 0
        else int(client.config.experiment.prefetch_factor)
    )

    train_loader = BioMolData(train_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=True,
        use_adaptive_sampler=True,
        batch_size=client.config.experiment.num_batch,
        num_workers=client.config.experiment.num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,
    )
    valid_loader = BioMolData(valid_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=False,
        use_adaptive_sampler=False,
        batch_size=client.config.experiment.num_batch,  # or 1
        num_workers=0,
    )

    client.logger.info("-" * 70)
    client.logger.info("")
    client.logger.info("Start training".center(70))
    client.logger.info("")
    client.logger.info("-" * 70)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_aggregator = MetricsAggregator(client, "train", use_wandb=w)
    valid_aggregator = MetricsAggregator(client, "valid", use_wandb=w)

    world_size = fabric.world_size
    train_num_item = client.config.experiment.train_item // world_size
    valid_num_item = client.config.experiment.valid_item // world_size
    min_train_loss = float("inf")
    comment = client.config.experiment.comment

    for epoch in range(client.epoch, client.config.experiment.num_epoch):
        client.logger.info("Training Epoch %d", client.epoch)
        train_loader.sampler.set_epoch(epoch)
        for n_item, result in enumerate(client.training_epoch(train_loader)):
            train_aggregator.log_step(result)
            if n_item + 1 >= train_num_item:
                client._epoch += 1  # noqa: SLF001
                client.call_callbacks("on_train_epoch_end")
                break

        means = train_aggregator.log_epoch()

        if client.is_global_zero:
            train_loss = means["total_loss"]
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                checkpoint_path = ckpt_dir / f"cmap_template_af3_{comment}_best.pt"

        if (client.epoch - 1) % client.config.experiment.eval_freq == 0:
            valid_loader.sampler.set_epoch(epoch)
            client.logger.info("Validation Epoch %d", client.epoch)
            for n_item, result in enumerate(client.validation_epoch(valid_loader)):
                valid_aggregator.log_step(result, ignore_step=True)
                if n_item + 1 >= valid_num_item:
                    client.call_callbacks("on_validation_epoch_end")
                    break

            valid_aggregator.log_epoch()

            checkpoint_path = ckpt_dir / f"cmap_template_af3_{comment}_{epoch}.pt"
            client.save_checkpoint(checkpoint_path)


@cli.command()
@click.option(
    "--ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="checkpoint file",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="config file",
)
@click.option(
    "--seed",
    type=int,
    help="random seed",
)
@click.option(
    "--save-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="directory to save predicted structures",
)
def sample(
    ckpt: Path | None,
    config: Path | None,
    seed: int | None,
    save_dir: Path | None,
):
    config = OmegaConf.load(config)
    config = AF3Client.Config.model_validate(config)
    if not ckpt:
        msg = "You must provide a checkpoint file."
        raise ValueError(msg)
    client = AF3Client.from_checkpoint(ckpt)

    fabric = Fabric()
    fabric.launch()

    client.setup(
        fabric=fabric,
        optimizer=torch.optim.AdamW(
            client.model.parameters(),
            client.config.experiment.max_lr,
        ),
        gradient_accumulation_steps=client.config.experiment.grad_accum_steps,
        gradient_clip_norm=client.config.experiment.grad_clip_max_norm,
    )
    setup_logger(client)

    client.logger.info(
        "Load pretrain weight: %s (%d epoch)",
        ckpt.name,
        client.epoch,
    )

    msg = f"Config:\n{json.dumps(client.config.model_dump(), indent=4, default=str)}"
    client.logger.info(msg)

    if seed is not None:
        set_seed(seed)
        client.logger.info("Set random seed: %d", seed)

    crop_config = client.config.data.crop.model_dump()
    crop_config["crop_length"] = 1024
    crop_config["contiguous_prob"] = 1.0
    crop_config["spatial_prob"] = 0.0
    crop_config["interface_simple_prob"] = 0.0
    msa_config = client.config.data.msa.model_dump()
    msa_config["max_msa_depth"] = 32

    crop_indices1 = [ii for ii in range(15,200)]
    crop_indices2 = [ii for ii in range(259,459)]
    crop_indices = crop_indices1 + crop_indices2
    crop_indices = np.array(crop_indices, dtype=np.int32)
    train_data_config = BioMolData.BioMolConfig(
        crop_config=crop_config,
        msa_config=msa_config,
        DB_config=config.data.train_db.model_dump(),
        edge_weight_config=config.data.edge_weight.model_dump(),
    )
    train_dataset = BioMolData(train_data_config)
    transporter_open_batch = train_dataset.get_item_by_id(
        cif_id="6lyy_1_1_._(C_1)_(A_1)",
        chain_bias="A_1",
        remain_invalid_residues=False,
        crop_indices=crop_indices,
    )
    transporter_closed_batch = train_dataset.get_item_by_id(
        cif_id="7cko_1_1_._(C_1)_(A_1)",
        chain_bias="A_1",
        remain_invalid_residues=False,
        crop_indices=crop_indices,
    )
    open_contact_map, open_contact_map_mask = get_contact_map(
        transporter_open_batch.structure.atom_pos,
        transporter_open_batch.structure.atom_pos_mask,
        transporter_open_batch.scheme.atom_to_residue_idx_map,
    ) # (B, N_res, N_res)
    closed_contact_map, closed_contact_map_mask = get_contact_map(
        transporter_closed_batch.structure.atom_pos,
        transporter_closed_batch.structure.atom_pos_mask,
        transporter_closed_batch.scheme.atom_to_residue_idx_map,
    )
    B, L, _ = open_contact_map.shape
    diag_mask = ~torch.eye(L, dtype=torch.bool, device=open_contact_map.device)
    diag_mask = diag_mask.unsqueeze(0)  # [1, L, L]
    tri_mask  = torch.triu(torch.ones(L, L, dtype=torch.bool, device=open_contact_map.device), diagonal=1)[None, :, :]  # i<j만

    open_contact_map = open_contact_map & open_contact_map_mask
    open_noncontact_map = (~open_contact_map) & open_contact_map_mask
    closed_contact_map = closed_contact_map & closed_contact_map_mask
    closed_noncontact_map = (~closed_contact_map) & closed_contact_map_mask

    msa_depth = msa_config["max_msa_depth"]
    save_dir = f"./transporter/msa_depth_{msa_depth}/"
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    open_only_contact = open_contact_map & closed_noncontact_map & tri_mask
    open_only_noncontact = open_noncontact_map & closed_contact_map & tri_mask
    closed_only_contact = closed_contact_map & open_noncontact_map & tri_mask
    closed_only_noncontact = closed_noncontact_map & open_contact_map & tri_mask

    open_only_contact = open_only_contact.to(client.device)
    open_only_noncontact = open_only_noncontact.to(client.device)
    closed_only_contact = closed_only_contact.to(client.device)
    closed_only_noncontact = closed_only_noncontact.to(client.device)

    client.model.eval()

    # sample_ratio_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]
    sample_ratio_list = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04]
    for sample_ratio in sample_ratio_list:
        for steered_state in ["open", "closed"]:
            batch = transporter_open_batch.duplicate(client.config.experiment.eval_sample_num)
            batch = batch.to(device=client.device)

            residue_length = batch.scheme.residue_idx.shape[1]
            contact_map = torch.zeros(
                (1, residue_length, residue_length, 3), device=client.device
            )
            contact_map[:, :, :, 2] = 1.0  # unknown
            if steered_state == "open":
                # one-hot encode open contact map
                num_contact_sampled = int(open_only_contact.sum().item() * sample_ratio)
                num_noncontact_sampled = int(open_only_noncontact.sum().item() * sample_ratio)
                sampled_contact_indices = torch.nonzero(open_only_contact, as_tuple=False)
                perm = torch.randperm(sampled_contact_indices.shape[0], device=client.device)[:num_contact_sampled]
                sampled_contact_indices = sampled_contact_indices[perm]
                sampled_noncontact_indices = torch.nonzero(open_only_noncontact, as_tuple=False)
                perm = torch.randperm(sampled_noncontact_indices.shape[0], device=client.device)[:num_noncontact_sampled]
                sampled_noncontact_indices = sampled_noncontact_indices[perm]
                for idx in sampled_contact_indices:
                    i, j = idx[1], idx[2]
                    contact_map[0, i, j, 2] = 0.0  # unknown
                    contact_map[0, j, i, 2] = 0.0  # unknown
                    contact_map[0, i, j, 1] = 1.0  # contact
                    contact_map[0, j, i, 1] = 1.0  # contact
                for idx in sampled_noncontact_indices:
                    i, j = idx[1], idx[2]
                    contact_map[0, i, j, 2] = 0.0  # unknown
                    contact_map[0, j, i, 2] = 0.0  # unknown
                    contact_map[0, i, j, 0] = 1.0  # noncontact
                    contact_map[0, j, i, 0] = 1.0  # noncontact
                # contact_map[..., 0] = ((open_contact_map == 0) & (open_contact_map_mask==1)).float()  # noncontact
                # contact_map[..., 1] = ((open_contact_map == 1) & (open_contact_map_mask==1)).float()  # contact
            else:
                # one-hot encode closed contact map
                num_contact_sampled = int(closed_only_contact.sum().item() * sample_ratio)
                num_noncontact_sampled = int(closed_only_noncontact.sum().item() * sample_ratio)
                sampled_contact_indices = torch.nonzero(closed_only_contact, as_tuple=False)
                perm = torch.randperm(sampled_contact_indices.shape[0], device=client.device)[:num_contact_sampled]
                sampled_contact_indices = sampled_contact_indices[perm]
                sampled_noncontact_indices = torch.nonzero(closed_only_noncontact, as_tuple=False)
                perm = torch.randperm(sampled_noncontact_indices.shape[0], device=client.device)[:num_noncontact_sampled]
                sampled_noncontact_indices = sampled_noncontact_indices[perm]
                for idx in sampled_contact_indices:
                    i, j = idx[1], idx[2]
                    contact_map[0, i, j, 2] = 0.0  # unknown
                    contact_map[0, j, i, 2] = 0.0  # unknown
                    contact_map[0, i, j, 1] = 1.0  # contact
                    contact_map[0, j, i, 1] = 1.0  # contact
                for idx in sampled_noncontact_indices:
                    i, j = idx[1], idx[2]
                    contact_map[0, i, j, 2] = 0.0  # unknown
                    contact_map[0, j, i, 2] = 0.0  # unknown
                    contact_map[0, i, j, 0] = 1.0  # noncontact
                    contact_map[0, j, i, 0] = 1.0  # noncontact
                # contact_map[..., 0] = ((closed_contact_map == 0) & (closed_contact_map_mask==1)).float()  # noncontact
                # contact_map[..., 1] = ((closed_contact_map == 1) & (closed_contact_map_mask==1)).float()  # contact
            contact_map_mask = open_contact_map_mask if steered_state == "open" else closed_contact_map_mask
            contact_map_mask = contact_map_mask.to(client.device)

            output = client.inference(
                batch,
                timesteps=client.config.experiment.eval_timesteps,
                contact_map=contact_map,
                contact_map_mask=contact_map_mask,
            )

            max_lddt, min_rmsd = 0, float("inf")

            lddt = metrics.cal_atom_lddt(
                output.atom_pos_pred[0],
                batch.structure.atom_pos[0],
                batch.structure.atom_mask[0],
            )
            max_lddt = max(max_lddt, lddt)

            rmsd = metrics.cal_aligned_rmsd(
                output.atom_pos_pred[0],
                batch.structure.atom_pos[0],
                batch.structure.atom_mask[0],
            )
            category_lddt = metrics.category_lddt(
                batch,
                output.atom_pos_pred[0],
            )
            predicted_contact_map, _ = get_contact_map(
                output.atom_pos_pred,
                batch.structure.atom_pos_mask,
                batch.scheme.atom_to_residue_idx_map,
            )
            if steered_state == "open":
                true_contact_map = open_only_contact
                true_noncontact_map = open_only_noncontact
            else:
                true_contact_map = closed_only_contact
                true_noncontact_map = closed_only_noncontact
            correct_contacts = (predicted_contact_map.bool() & true_contact_map).sum().item()
            total_contacts = true_contact_map.sum().item()
            correct_noncontacts = ((~predicted_contact_map.bool()) & true_noncontact_map).sum().item()
            total_noncontacts = true_noncontact_map.sum().item()
            contact_accuracy = correct_contacts / total_contacts if total_contacts > 0 else 0.0
            noncontact_accuracy = correct_noncontacts / total_noncontacts if total_noncontacts > 0 else 0.0
            click.echo(f"Steered state: {steered_state}, Sample ratio: {sample_ratio}")
            click.echo(f"Contact accuracy: {contact_accuracy:.4f}, Non-contact accuracy: {noncontact_accuracy:.4f}")
            
            min_rmsd = min(min_rmsd, rmsd)
            click.echo(f"<<<category_lddt[{batch.name}]: {category_lddt}>>>")
            batch_name = batch.name[0]
            batch_name = batch_name.split("_")[0]
            batch_to_cif(
                batch,
                atom_pos_pred=output.atom_pos_pred,
                save_path=Path(f"{save_dir}/sample_{batch_name}_{steered_state}_{sample_ratio}.cif"),
            )
    breakpoint()

    valid_data_config = BioMolData.BioMolConfig(
        crop_config=crop_config,
        msa_config=client.config.data.msa.model_dump(),
        DB_config=client.config.data.valid_db.model_dump(),
        edge_weight_config=client.config.data.edge_weight.model_dump(),
    )

    prefetch_factor = (
        None
        if client.config.experiment.prefetch_factor == 0
        else int(client.config.experiment.prefetch_factor)
    )
    valid_loader = BioMolData(valid_data_config).create_ddp_dataloader(
        world_size=fabric.world_size,
        rank=fabric.local_rank,
        drop_last=False,
        batch_size=client.config.experiment.num_batch,  # or 1
        num_workers=client.config.experiment.num_workers,
        prefetch_factor=prefetch_factor,
    )

    client.logger.info("-" * 70)
    client.logger.info("")
    client.logger.info("Start training".center(70))
    client.logger.info("")
    client.logger.info("-" * 70)

    for _batch in valid_loader:
        batch = _batch.duplicate(client.config.experiment.eval_sample_num)
        batch = batch.to(device=client.device)

        output = client.inference(
            batch,
            timesteps=client.config.experiment.eval_timesteps,
        )

        max_lddt, min_rmsd = 0, float("inf")

        lddt = metrics.cal_atom_lddt(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_mask[0],
        )
        max_lddt = max(max_lddt, lddt)

        rmsd = metrics.cal_aligned_rmsd(
            output.atom_pos_pred[0],
            batch.structure.atom_pos[0],
            batch.structure.atom_mask[0],
        )
        category_lddt = metrics.category_lddt(
            batch,
            output.atom_pos_pred[0],
        )
        min_rmsd = min(min_rmsd, rmsd)
        click.echo(f"<<<category_lddt[{batch.name}]: {category_lddt}>>>")
        batch_to_cif(
            batch,
            atom_pos_pred=output.atom_pos_pred,
            save_path=Path(f"sample_{batch.name}.cif"),
        )
        breakpoint()


if __name__ == "__main__":
    # set mp start method
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
