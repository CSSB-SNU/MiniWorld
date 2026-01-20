import datetime
import json
import logging
from pathlib import Path

import click
import numpy as np
import torch
from lightning import Fabric
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from team_gm.utils.script_utils import set_seed

from miniworld.data.dataloader.dataloader_edge_backprop import (
    BioMolData,
)
from miniworld.data.to_cif import batch_to_cif
from miniworld.loss import metrics  # , losses
from miniworld.models.cmap_template_af3 import AF3Client
from miniworld.models.contact_map_prediction import ContactMapPredictionClient
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

def draw_contact_map(
    contact_map_prob: torch.Tensor,
    contact_state1: torch.Tensor,
    contact_state2: torch.Tensor,
    residue_pair_mask: torch.Tensor,
    save_path: Path,
):
    # assume shape: (B, L, L)
    contact_map_prob = contact_map_prob[0].cpu().numpy()
    contact_state1 = contact_state1[0].cpu().numpy()
    contact_state2 = contact_state2[0].cpu().numpy()
    residue_pair_mask = residue_pair_mask[0].cpu().numpy()

    # apply mask
    contact_map_prob = contact_map_prob * residue_pair_mask
    contact_state1 = contact_state1 * residue_pair_mask
    contact_state2 = contact_state2 * residue_pair_mask

    # 3 panels: predicted, target1, target2
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(contact_map_prob, cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title("Predicted Contact Map")

    axes[1].imshow(contact_state1, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Target Contact Map (State 1)")

    axes[2].imshow(contact_state2, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Target Contact Map (State 2)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def sample_pairs_from_prob(
    p: torch.Tensor,              # [1,L,L]
    valid: torch.Tensor,          # [1,L,L] boolean
    sample_ratio: float,
    temperature: float = 1.0,
    low_th: float = 0.2,
    high_th: float = 0.8,
    eps: float = 1e-8,
):
    L = p.shape[1]
    assert p.shape == valid.shape

    pf = p[0].flatten().clamp(eps, 1 - eps)   # [L*L]
    v0 = valid[0].flatten()                   # [L*L]

    # 후보 분리 (서로 독립)
    v_c = v0 & (pf >= high_th)
    v_n = v0 & (pf <= low_th)

    n_c = int(v_c.sum().item())
    n_n = int(v_n.sum().item())

    # 각각에서 ratio 적용
    k_c = int(n_c * sample_ratio)
    k_n = int(n_n * sample_ratio)

    # weights (선택 확률)
    wc = torch.zeros_like(pf)
    wn = torch.zeros_like(pf)
    wc[v_c] = pf[v_c] ** (1.0 / temperature)
    wn[v_n] = (1.0 - pf[v_n]) ** (1.0 / temperature)

    def _draw(w, k):
        if k <= 0:
            return torch.empty((0,), dtype=torch.long, device=w.device)
        s = w.sum()
        if s <= 0:
            return torch.empty((0,), dtype=torch.long, device=w.device)
        w = w / s
        # k가 후보 수보다 크면 전부 뽑도록 clamp
        # (multinomial replacement=False 제약)
        k = min(k, int((w > 0).sum().item()))
        if k <= 0:
            return torch.empty((0,), dtype=torch.long, device=w.device)
        return torch.multinomial(w, num_samples=k, replacement=False)

    idx_c = _draw(wc, k_c)
    idx_n = _draw(wn, k_n)

    i_c, j_c = idx_c // L, idx_c % L
    i_n, j_n = idx_n // L, idx_n % L
    return (i_c, j_c), (i_n, j_n)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--model1_ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="checkpoint file",
)
@click.option(
    "--model2_ckpt",
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
    model1_ckpt: Path | None,
    model2_ckpt: Path | None,
    config: Path | None,
    seed: int | None,
    save_dir: Path | None,
):
    config = OmegaConf.load(config)
    config = AF3Client.Config.model_validate(config)
    if not model2_ckpt:
        msg = "You must provide a checkpoint file."
        raise ValueError(msg)
    contactmap_prediction_client = ContactMapPredictionClient.from_checkpoint(model1_ckpt)
    structure_client = AF3Client.from_checkpoint(model2_ckpt)
    contactmap_prediction_client.model.eval()
    structure_client.model.eval()

    fabric = Fabric()
    fabric.launch()

    contactmap_prediction_client.setup(
        fabric=fabric,
        optimizer=torch.optim.AdamW(
            contactmap_prediction_client.model.parameters(),
            contactmap_prediction_client.config.experiment.max_lr,
        ),
        gradient_accumulation_steps=contactmap_prediction_client.config.experiment.grad_accum_steps,
        gradient_clip_norm=contactmap_prediction_client.config.experiment.grad_clip_max_norm,
    )

    structure_client.setup(
        fabric=fabric,
        optimizer=torch.optim.AdamW(
            structure_client.model.parameters(),
            structure_client.config.experiment.max_lr,
        ),
        gradient_accumulation_steps=structure_client.config.experiment.grad_accum_steps,
        gradient_clip_norm=structure_client.config.experiment.grad_clip_max_norm,
    )
    device = structure_client.device

    setup_logger(structure_client)

    structure_client.logger.info(
        "Load pretrain weight: %s (%d epoch)",
        model2_ckpt.name,
        structure_client.epoch,
    )

    msg = f"Config:\n{json.dumps(structure_client.config.model_dump(), indent=4, default=str)}"
    structure_client.logger.info(msg)
    if seed is not None:
        set_seed(seed)
        structure_client.logger.info("Set random seed: %d", seed)

    crop_config = structure_client.config.data.crop.model_dump()
    crop_config["crop_length"] = 1024
    crop_config["contiguous_prob"] = 1.0
    crop_config["spatial_prob"] = 0.0
    crop_config["interface_simple_prob"] = 0.0
    cmap_msa_config = contactmap_prediction_client.config.data.msa.model_dump()
    structure_msa_config = structure_client.config.data.msa.model_dump()

    cmap_msa_depth = 1024
    str_msa_depth = 128
    cmap_msa_config["max_msa_depth"] = cmap_msa_depth
    structure_msa_config["max_msa_depth"] = str_msa_depth
    save_dir = f"./transporter/full/msa_depth_{cmap_msa_depth}_{str_msa_depth}/"
    cmap_save_dir = f"./transporter/full/msa_depth_{cmap_msa_depth}_{str_msa_depth}/cmap_prediction/"

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    if cmap_save_dir is not None:
        cmap_save_dir = Path(cmap_save_dir)
        cmap_save_dir.mkdir(parents=True, exist_ok=True)


    crop_indices1 = list(range(15,200))
    crop_indices2 = list(range(259,459))
    crop_indices = crop_indices1 + crop_indices2
    crop_indices = np.array(crop_indices, dtype=np.int32)
    contact_map_data_config = BioMolData.BioMolConfig(
        crop_config=crop_config,
        msa_config=cmap_msa_config,
        DB_config=config.data.train_db.model_dump(),
        edge_weight_config=config.data.edge_weight.model_dump(),
    )
    structure_data_config = BioMolData.BioMolConfig(
        crop_config=crop_config,
        msa_config=structure_msa_config,
        DB_config=config.data.train_db.model_dump(),
        edge_weight_config=config.data.edge_weight.model_dump(),
    )
    cmap_dataset = BioMolData(contact_map_data_config)
    structure_dataset = BioMolData(structure_data_config)
    transporter_open_batch = structure_dataset.get_item_by_id(
        cif_id="6lyy_1_1_._(C_1)_(A_1)",
        chain_bias="A_1",
        remain_invalid_residues=False,
        crop_indices=crop_indices,
    )
    transporter_closed_batch = structure_dataset.get_item_by_id(
        cif_id="7cko_1_1_._(C_1)_(A_1)",
        chain_bias="A_1",
        remain_invalid_residues=False,
        crop_indices=crop_indices,
    )
    transporter_open_batch = transporter_open_batch.to(device=device)
    transporter_closed_batch = transporter_closed_batch.to(device=device)

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

    transporter_cmap_batch = cmap_dataset.get_item_by_id(
        cif_id="6lyy_1_1_._(C_1)_(A_1)",
        chain_bias="A_1",
        remain_invalid_residues=False,
        crop_indices=crop_indices,
    )

    contact_map_mask = open_contact_map_mask & closed_contact_map_mask

    predicted_contact_map_prob = contactmap_prediction_client.predict_contact_map(
        transporter_cmap_batch,
    )
    draw_contact_map(
        predicted_contact_map_prob,
        open_contact_map,
        closed_contact_map,
        contact_map_mask,
        save_path=cmap_save_dir / "transporter_contact_map_prediction.png",
    )
    B, L, _ = open_contact_map.shape
    diag_mask = ~torch.eye(L, dtype=torch.bool, device=open_contact_map.device)
    diag_mask = diag_mask.unsqueeze(0)  # [1, L, L]
    tri_mask  = torch.triu(torch.ones(L, L, dtype=torch.bool, device=open_contact_map.device), diagonal=1)[None, :, :]  # i<j만
    valid = contact_map_mask & diag_mask & tri_mask  # [B,L,L] (보통 B=1)

    # sample_ratio_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]
    sample_ratio_list = [0.5, 0.8, 1.0]

    open_contact_map = open_contact_map & contact_map_mask
    closed_contact_map = closed_contact_map & contact_map_mask

    combined_contact_map = open_contact_map | closed_contact_map  # 두 상태에서 contact인 쌍
    open_only_contact_map = open_contact_map & (~closed_contact_map)  # open 상태에서만 contact인 쌍
    closed_only_contact_map = closed_contact_map & (~open_contact_map)  # closed 상태에서만 contact인 쌍

    for sample_ratio in sample_ratio_list:
        batch = transporter_open_batch.duplicate(structure_client.config.experiment.eval_sample_num).to(device=device)
        residue_length = batch.scheme.residue_idx.shape[1]

        contact_map = torch.zeros((1, residue_length, residue_length, 3), device=device)
        contact_map[..., 2] = 1.0  # unknown

        (ic, jc), (in_, jn_) = sample_pairs_from_prob(
            p=predicted_contact_map_prob,
            valid=valid,
            sample_ratio=sample_ratio,
            temperature=0.7,
            low_th=0.2,
            high_th=0.8,
        )

        correct_contacts = combined_contact_map[0, ic, jc].sum().item()
        correct_open_only = open_only_contact_map[0, ic, jc].sum().item()
        correct_closed_only = closed_only_contact_map[0, ic, jc].sum().item()

        click.echo(f"Correct contacts (sample_ratio={sample_ratio}): {correct_contacts} / {len(ic)}")
        click.echo(f"  - Open only contacts: {correct_open_only}")
        click.echo(f"  - Closed only contacts: {correct_closed_only}")

        # filter open correct pairs only
        # contact_correct = open_only_contact_map[0, ic, jc]
        # non_contact_correct = closed_only_contact_map[0, in_, jn_]
        contact_correct = closed_only_contact_map[0, ic, jc]
        non_contact_correct = open_only_contact_map[0, in_, jn_]
        ic = ic[contact_correct]
        jc = jc[contact_correct]
        in_ = in_[non_contact_correct]
        jn_ = jn_[non_contact_correct]

        # one-hot 채우기 (대칭 포함)
        # contact
        contact_map[0, ic, jc, 2] = 0.0
        contact_map[0, jc, ic, 2] = 0.0
        contact_map[0, ic, jc, 1] = 1.0
        contact_map[0, jc, ic, 1] = 1.0

        # # noncontact
        contact_map[0, in_, jn_, 2] = 0.0
        contact_map[0, jn_, in_, 2] = 0.0
        contact_map[0, in_, jn_, 0] = 1.0
        contact_map[0, jn_, in_, 0] = 1.0

        contact_map_mask = contact_map_mask.to(device)

        output = structure_client.inference(
            batch,
            timesteps=structure_client.config.experiment.eval_timesteps,
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

        min_rmsd = min(min_rmsd, rmsd)
        # click.echo(f"<<<category_lddt[{batch.name}]: {category_lddt}>>>")
        batch_name = batch.name[0]
        batch_name = batch_name.split("_")[0]
        batch_to_cif(
            batch,
            atom_pos_pred=output.atom_pos_pred,
            save_path=Path(f"{save_dir}/sample_{batch_name}_{sample_ratio}.cif"),
        )
    breakpoint()


if __name__ == "__main__":
    # set mp start method
    torch.multiprocessing.set_start_method("spawn", force=True)
    cli()
