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
import matplotlib.pyplot as plt
from MiniWorld.utils.structure.sdist import get_shortest_distances, pairwise_kabsch_rmsd, save_rmsd_boxplot, plot_rmsd_heatmap, cal_radius_of_gyration, compare_radius_of_gyration_distributions

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


def visualize_output(
    pred_dist: torch.Tensor,  # (L, L)
    true_dist: torch.Tensor,  # (L, L)
    mask: torch.Tensor,       # (L, L)
    out_path: Path,
):
    """
    Visualize predicted vs true distance maps and their difference using matplotlib only.

    Args:
        pred_dist: Predicted residue-level distances. (L, L)
        true_dist: Ground truth residue-level distances. (L, L)
        mask: Boolean mask for valid residue pairs. (L, L)
        out_path: Path to save the resulting figure.
    """
    pred_dist = pred_dist.detach().cpu().float()
    true_dist = true_dist.detach().cpu().float()
    mask = mask.detach().cpu().bool()

    # Apply mask (invalid → NaN)
    pred_dist = torch.where(mask, pred_dist, torch.tensor(float("nan")))
    true_dist = torch.where(mask, true_dist, torch.tensor(float("nan")))
    diff = pred_dist - true_dist

    # Convert to numpy for matplotlib
    pred_dist = pred_dist.numpy()
    true_dist = true_dist.numpy()
    diff = diff.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Common color scale for distances
    vmin = float(np.nanmin(true_dist))
    vmax = float(np.nanmax(true_dist))

    im0 = axes[0].imshow(true_dist, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("True Distance (Å)")

    im1 = axes[1].imshow(pred_dist, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted Distance (Å)")

    diff_lim = float(np.nanmax(np.abs(diff)))
    im2 = axes[2].imshow(diff, cmap="coolwarm", vmin=-diff_lim, vmax=diff_lim)
    axes[2].set_title("Difference (Pred - True)")

    for ax in axes:
        ax.set_xlabel("Residue index j")
        ax.set_ylabel("Residue index i")

    # Add colorbars
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


@cli.command("sdist_predict")
@click.option("--config", type=click.Path(exists=True), help="config file")
@click.option(
    "--ckpt", type=click.Path(exists=True), help="checkpoint file", required=True
)
@click.option("--timesteps", type=int, default=100, help="number of timesteps")
@click.option("--out_dir", type=click.Path(), default="output/", help="output dir")
@click.option("--device", default="cuda", type=str, help="device to use")
@click.option("--seqID", "seqID", default=None, type=str, help="sequence ID to predict")
@add_slurm_options
def sdist_predict(
    config: str,
    ckpt: str,
    timesteps: int = 100,
    out_dir: str = "outputs",
    device: str = "cuda",
    seqID: str = None,
    slurm: bool = False,
    mem: str = "32G",
    cpus: int = 8,
    gpus: str = "A6000:1",
):
    """Validation mode for SdistPredict."""
    if slurm:
        job_name = Path(ckpt).stem
        path = submit_to_slurm(
            f"pixi run python {__file__} sdist_predict --ckpt {ckpt}"
            f" --timesteps {timesteps}"
            f" --out_dir {out_dir} --device {device}",
            job_name=job_name,
            mem=mem,
            cpus=cpus,
            gpus=gpus,
        )
        click.echo(f"✅ Submitted Slurm job: {job_name} ({path})")
        return
    from MiniWorld.models.sdist_prediction import SdistClient

    if torch.device(device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")


    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot found config file: {config_path}")
    config = OmegaConf.load(config)

    ckpt_path = Path(ckpt)
    client = SdistClient.from_checkpoint(ckpt_path)

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
        shuffle=True,
    )

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    client = client.to(device=device)

    if seqID is not None:
        valid_loader = [valid_data.get_item_by_seqID(seqID)]

    print(f"Starting validation on {len(valid_loader)} batches.")

    for ii, batch in enumerate(valid_loader):
        batch : Batch = batch.to(device=device)
        expected_dist, residue_dists, residue_pair_mask = client.inference(
            batch=batch,
        )
        out_path = out_dir_path / f"{batch.name[0]}_distogram_validation.png"
        visualize_output(
            pred_dist=expected_dist[0],
            true_dist=residue_dists[0],
            mask=residue_pair_mask[0],
            out_path=out_path,
        )
        breakpoint()

@cli.command("sdist2str")
@click.option("--config", type=click.Path(exists=True), help="config file")
@click.option(
    "--ckpt", type=click.Path(exists=True), help="checkpoint file", required=True
)
@click.option("--timesteps", type=int, default=100, help="number of timesteps")
@click.option("--out_dir", type=click.Path(), default="output/", help="output dir")
@click.option("--device", default="cuda", type=str, help="device to use")
@click.option("--seqID", "seqID", default=None, type=str, help="sequence ID to predict")
@click.option("--n_sample", "n_sample", default=None, type=int, help="number of structures to predict")
@add_slurm_options
def sdist2str(
    config: str,
    ckpt: str,
    timesteps: int = 100,
    out_dir: str = "outputs",
    device: str = "cuda",
    seqID: str = None,
    n_sample: int = 48,
    slurm: bool = False,
    mem: str = "32G",
    cpus: int = 8,
    gpus: str = "A6000:1",
):
    """sdist2str"""
    if slurm:
        job_name = Path(ckpt).stem
        path = submit_to_slurm(
            f"pixi run python {__file__} sdist2str --ckpt {ckpt}"
            f" --timesteps {timesteps}"
            f" --out_dir {out_dir} --device {device}",
            job_name=job_name,
            mem=mem,
            cpus=cpus,
            gpus=gpus,
        )
        click.echo(f"✅ Submitted Slurm job: {job_name} ({path})")
        return
    from MiniWorld.models.sdist_to_str import Sdist2StrClient

    if torch.device(device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")


    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot found config file: {config_path}")
    config = OmegaConf.load(config)

    ckpt_path = Path(ckpt)
    client = Sdist2StrClient.from_checkpoint(ckpt_path)

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

    if seqID is not None:
        valid_loader = [valid_data.get_full_item_by_seqID(seqID)]

    for ii, batch in enumerate(valid_loader):
        batch : Batch = batch.to(device=device)

        residue_dists, residue_pair_mask = get_shortest_distances(
            atom_pos=batch.structure.atom_pos,
            atom_pos_mask=batch.structure.atom_pos_mask,
            atom_to_res_idx=batch.scheme.atom_to_residue_idx_map,
            min_distance=2.0,
            max_distance=22.0,
        )  # residue_dists: (*, L, L), residue_pair_mask: (*, L, L)

        visualize_output(
            pred_dist=residue_dists[0],
            true_dist=residue_dists[0],
            mask=residue_pair_mask[0],
            out_path=out_dir_path / f"{batch.name[0]}_input_distogram.png",
        )

        af3_inference_output = client.best_of_N_sample(
            batch=batch,
            timesteps=timesteps,
            n_sample=n_sample,
            batch_size=min(48,n_sample),
        )
        atom_pos_pred = af3_inference_output.atom_pos_pred[:,0] # (n_sample, L_atom, 3)
        n_atoms = atom_pos_pred.shape[1]
        queryID = batch.name[0]
        cifmols = valid_data.load_cifmols(queryID)
        cifmols = [cifmol for cifmol in cifmols if len(cifmol.atoms) == n_atoms]
        true_atom_pos = batch.structure.atom_pos[0]  # (N_str, L_atom, 3)
        true_atom_pos_mask = batch.structure.atom_pos_mask[0]  # (N_str, L_atom)

        pairwise_kabsch_rmsd_values = pairwise_kabsch_rmsd(
            pred=atom_pos_pred,          # (S, L, 3)
            true=true_atom_pos,          # (N, L, 3)
            true_mask=true_atom_pos_mask,     # (N, L)
        )  # (S, N)

        pred_pairwise_kabsch_rmsd_values = pairwise_kabsch_rmsd(
            pred=atom_pos_pred,          # (S, L, 3)
            true=atom_pos_pred,          # (S, L, 3)
            true_mask=torch.ones_like(atom_pos_pred[:,:,0], dtype=bool),     # (S, L)
        )  # (S, S)

        true_pairwise_kabsch_rmsd_values = pairwise_kabsch_rmsd(
            pred=true_atom_pos,          # (N, L, 3)
            true=true_atom_pos,          # (N, L, 3)
            true_mask=true_atom_pos_mask,     # (N, L)
        )  # (N, N)

        pred_rg_values = cal_radius_of_gyration(
            atom_pos=atom_pos_pred,          # (S, L, 3)
            atom_pos_mask=torch.ones_like(atom_pos_pred[:,:,0], dtype=bool),     # (S, L)
        )  # (S,)
        true_rg_values = cal_radius_of_gyration(
            atom_pos=true_atom_pos,          # (N, L, 3)
            atom_pos_mask=true_atom_pos_mask,     # (N, L)
        )  # (N,)

        save_rmsd_boxplot(
            data_2d=pairwise_kabsch_rmsd_values.T,  # (N, S)
            save_path=out_dir_path / f"{batch.name[0]}_rmsd_per_true.png",
            title=f"RMSD per predicted structure for {batch.name[0]}",
            xlabel="True idx",
        )
        save_rmsd_boxplot(
            data_2d=pairwise_kabsch_rmsd_values,  # (S, N)
            save_path=out_dir_path / f"{batch.name[0]}_rmsd_per_pred.png",
            title=f"RMSD per true structure for {batch.name[0]}",
            xlabel="Pred idx",
            ylabel="RMSD",
        )
        plot_rmsd_heatmap(
            rmsd_pp=pred_pairwise_kabsch_rmsd_values,  # (S, S)
            save_path=out_dir_path / f"{batch.name[0]}_pred_rmsd_heatmap.png",
            title=f"RMSD Heatmap for {batch.name[0]}",
        )

        compare_radius_of_gyration_distributions(
            pred_rg_values=pred_rg_values,  # (S,)
            true_rg_values=true_rg_values,  # (N,)
            save_path=out_dir_path / f"{batch.name[0]}_rg_comparison.png",
            title=f"Radius of Gyration Comparison for {batch.name[0]}",
        )
        breakpoint()

        # for cifmol in cifmols:
        #     cif_path = out_dir_path / f"{cifmol.id[0]}.mmcif"
        #     cifmol.to_cif(cif_path)

        # exchange strcuture
        # gt_pos = batch.structure.atom_pos # (B, N_str, L_atom, 3)
        true_atom_pos_mask = batch.structure.atom_pos_mask[0] # (N_str, L_atom)
        # find most complete structure
        true_atom_pos_sum = true_atom_pos_mask.sum(-1)  # (N_str,)
        best_str_idx = torch.argmax(true_atom_pos_sum).item()
        cifmol = cifmols[best_str_idx]
        true_atom_pos_mask = true_atom_pos_mask[best_str_idx]  # (L_atom,) 
        true_atom_pos_mask = true_atom_pos_mask.cpu().numpy()
        # lddt_list = []
        for ii in range(n_sample):
            denoised_mmcif_path = out_dir_path / f"{batch.name[0]}_denoised_{ii}.mmcif"
            cifmol_dict = copy.deepcopy(cifmol.to_dict())
            atom_pos_pred_ii = atom_pos_pred[ii].cpu().numpy()
            atom_pos_pred_ii[~true_atom_pos_mask] = np.nan
            cifmol_dict["atoms"]["nodes"]["xyz"]["value"]= atom_pos_pred_ii

            cifmol_denoised = type(cifmol).from_dict(cifmol_dict)
            cifmol_denoised.to_cif(denoised_mmcif_path)

            # lddt = metrics.cal_atom_lddt(
            #     pred_atom_pos=atom_pos_pred_ii,
            #     gt_atom_pos=gt_pos,
            #     atom_mask=true_atom_pos_mask,
            #     max_distance=max_distance,
            #     distance_bins=distance_bins,
            # )
            # lddt_list.append(lddt)
        breakpoint()




if __name__ == "__main__":
    cli()
