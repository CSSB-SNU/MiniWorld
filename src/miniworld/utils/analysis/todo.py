from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from jaxtyping import Float, Bool, Int

from team_gm import typecheck

@typecheck
def get_shortest_distances(
    atom_pos: Float[torch.Tensor, "* N L 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L"],
    atom_to_res_idx: Int[torch.Tensor, "* L"],
    min_distance: float = 2.0,
    max_distance: float = 22.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute residue-level shortest distances and corresponding mask from atom coordinates.

    Args:
        atom_pos: Atomic coordinates. Shape: (B, N, L, 3)
        atom_pos_mask: Atom mask. True for valid atoms. Shape: (B, N, L)
        atom_to_res_idx: Residue index per position. Shape: (B, L)
        min_distance: Minimum allowed distance.
        max_distance: Maximum allowed distance.

    Returns:
        residue_dists: Residue-level shortest distances. Shape: (B, R_max, R_max)
        residue_mask: Valid residue pair mask. Shape: (B, R_max, R_max)
    """
    device = atom_pos.device
    B, N, L, _ = atom_pos.shape

    # 1) Atom-level pairwise distances (B, N, L, L)
    diff = atom_pos[:, :, :, None, :] - atom_pos[:, :, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)  # (B, N, L, L)

    # Apply atom mask
    mask_i = atom_pos_mask[:, :, :, None]   # (B, N, L, 1)
    mask_j = atom_pos_mask[:, :, None, :]   # (B, N, 1, L)
    valid_atom_mask = mask_i & mask_j       # (B, N, L, L)

    dist = dist.masked_fill(~valid_atom_mask, max_distance)
    dist = dist.clamp(min=min_distance, max=max_distance)

    # Shortest atom-level distance between residues i, j
    shortest_dist = dist.min(dim=1).values  # (B, L, L)

    # 2) Build residue existence mask (B, R_max)
    R_max = int(atom_to_res_idx.max().item()) + 1

    # A position is valid if any atom at that position is valid
    pos_mask = atom_pos_mask.any(dim=1)  # (B, L)

    residue_exists = torch.zeros(B, R_max, dtype=torch.bool, device=device)

    # Flatten and mark residues that actually appear
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L).reshape(-1)
    flat_atom_to_res_idx = atom_to_res_idx.reshape(-1)
    flat_pos_mask = pos_mask.reshape(-1)

    valid_batch_idx = batch_idx[flat_pos_mask]
    valid_res_idx = flat_atom_to_res_idx[flat_pos_mask]

    residue_exists[valid_batch_idx, valid_res_idx] = True

    # Residue pair mask: both residues must exist
    mask_i_res = residue_exists.unsqueeze(2)  # (B, R_max, 1)
    mask_j_res = residue_exists.unsqueeze(1)  # (B, 1, R_max)
    residue_mask = mask_i_res & mask_j_res    # (B, R_max, R_max)

    # 3) Aggregate shortest distances to residue level using scatter-reduce (min)
    # Map (i, j) residue pairs to flat indices per batch
    ri = atom_to_res_idx.unsqueeze(2).expand(B, L, L)  # (B, L, L)
    rj = atom_to_res_idx.unsqueeze(1).expand(B, L, L)  # (B, L, L)
    pair_idx = ri * R_max + rj                  # (B, L, L)

    block_size = R_max * R_max
    batch_offsets = (torch.arange(B, device=device)
                     .view(B, 1, 1)
                     * block_size)             # (B, 1, 1)

    scatter_idx = batch_offsets + pair_idx      # (B, L, L)
    scatter_idx_flat = scatter_idx.reshape(-1)  # (B * L * L,)

    src = shortest_dist.reshape(-1)             # (B * L * L,)

    out = torch.full(
        (B * block_size,),
        max_distance,
        dtype=shortest_dist.dtype,
        device=device,
    )

    # Use scatter_reduce_ with amin to take minimum per index
    out.scatter_reduce_(
        dim=0,
        index=scatter_idx_flat,
        src=src,
        reduce="amin",
        include_self=True,
    )

    residue_dists = out.view(B, R_max, R_max)

    return residue_dists, residue_mask

def pairwise_kabsch_rmsd(
    pred: torch.Tensor,          # (S, L, 3)
    true: torch.Tensor,          # (N, L, 3)
    true_mask: torch.Tensor,     # (N, L) (bool or 0/1)
) -> torch.Tensor:
    """
    Compute (S, N) RMSD between S predicted structures and N true structures
    using Kabsch alignment and per-true-structure atom masks.
    """
    device = pred.device
    pred = pred.to(device)
    true = true.to(device)
    true_mask = true_mask.to(device).bool()

    S, L, _ = pred.shape
    N = true.shape[0]
    rmsd_list = []

    # (L,) — 모든 S 예측이 유효한 위치만 True
    pred_valid_L = torch.isfinite(pred).all(dim=0).all(dim=-1)

    for j in range(N):
        # (L,)
        true_valid_L = torch.isfinite(true[j]).all(dim=-1)
        valid = true_mask[j] & true_valid_L & pred_valid_L

        M = int(valid.sum())
        if M < 3:
            rmsd_list.append(torch.full((S,), float("nan"), device=device))
            continue

        xyz_t = true[j, valid]      # (M, 3)
        xyz_p = pred[:, valid]      # (S, M, 3)

        # Center
        t_c = xyz_t - xyz_t.mean(dim=0, keepdim=True)        # (M, 3)
        p_c = xyz_p - xyz_p.mean(dim=1, keepdim=True)        # (S, M, 3)

        # Covariance
        t_c_exp = t_c.unsqueeze(0).expand(S, -1, -1)         # (S, M, 3)
        C = torch.bmm(p_c.transpose(1, 2), t_c_exp)          # (S, 3, 3)

        # SVD
        U, _, Vh = torch.linalg.svd(C, full_matrices=False)  # U,Vh: (S,3,3)
        V = Vh.transpose(-2, -1)                              # (S, 3, 3)

        # Reflection correction using det(V U^T)
        detVU = torch.linalg.det(V @ U.transpose(-2, -1))     # (S,)
        sign = torch.where(detVU < 0, -torch.ones_like(detVU), torch.ones_like(detVU))
        D = torch.zeros_like(C)
        D[..., 0, 0] = 1.0
        D[..., 1, 1] = 1.0
        D[..., 2, 2] = sign
        R = V @ D @ U.transpose(-2, -1)                       # (S, 3, 3)

        # Align and RMSD
        p_aligned = torch.bmm(p_c, R)                         # (S, M, 3)
        diff2 = (p_aligned - t_c_exp).pow(2).sum(dim=-1)      # (S, M)
        rmsd_j = torch.sqrt(diff2.mean(dim=-1))               # (S,)
        rmsd_list.append(rmsd_j)

    return torch.stack(rmsd_list, dim=1)                      # (S, N)

def save_rmsd_boxplot(
    data_2d: np.ndarray | torch.Tensor,
    save_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str = "RMSD",
):
    """
    Common helper to draw and save a boxplot from a 2D array.
    Each row in `data_2d` is treated as one box.
    Rows are sorted by their minimum RMSD values.
    """
    if isinstance(data_2d, torch.Tensor):
        data_2d = data_2d.detach().cpu().numpy()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort rows by their minimum value
    order = np.argsort(np.nanmin(data_2d, axis=1))
    data_sorted = data_2d[order]

    num_boxes = data_sorted.shape[0]
    ylim = [0, np.nanmax(data_2d) * 1.1]

    fig, ax = plt.subplots(figsize=(num_boxes * 0.25+2, 4))
    ax.set_ylim(ylim)
    min_vals = np.nanmin(data_2d, axis=1)
    min_sorted = min_vals[order]

    num_structs = data_sorted.shape[0]

    # Plot all RMSD values as small gray points
    for i in range(num_structs):
        ax.scatter(
            np.full_like(data_sorted[i], i + 1, dtype=float),
            data_sorted[i],
            s=8,
            color="gray",
            alpha=0.5,
        )

    # Overlay min RMSD as red points
    ax.scatter(
        np.arange(1, num_structs + 1),
        min_sorted,
        color="red",
        s=20,
        label="min RMSD",
        zorder=3,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel + " (sorted by min RMSD)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_rmsd_heatmap(rmsd_pp, save_path: str | Path, title: str = "Pred–Pred RMSD"):
    """
    Plot and save a 2D RMSD map between predicted structures.

    Args:
        rmsd_pp: (S, S) tensor/ndarray, pairwise RMSD between S predicted structures.
        save_path: Path to save the figure.
        title: Title of the plot.
    """
    # Convert to numpy
    if isinstance(rmsd_pp, torch.Tensor):
        rmsd = rmsd_pp.detach().cpu().numpy()
    else:
        rmsd = np.asarray(rmsd_pp)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    S = rmsd.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, S * 0.3), max(5, S * 0.3)))

    im = ax.imshow(rmsd, origin="lower", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("Pred idx")
    ax.set_ylabel("Pred idx")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RMSD")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def cal_radius_of_gyration(
    atom_pos: Float[torch.Tensor, "* N L 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L"],
) -> Float[torch.Tensor, "* N"]:
    """
    Calculate radius of gyration for each structure in the batch.

    Args:
        atom_pos: Atomic coordinates. Shape: (N, L, 3)
        atom_pos_mask: Atom mask. True for valid atoms. Shape: (N, L)

    Returns:
        radius_of_gyration: Radius of gyration for each structure. Shape: (N,)
    """
    device = atom_pos.device
    N, L, _ = atom_pos.shape

    # Compute center of mass for each structure
    masked_atom_pos = atom_pos * atom_pos_mask.unsqueeze(-1)  # (N, L, 3)
    num_valid_atoms = atom_pos_mask.sum(dim=1).unsqueeze(-1)   # (N, 1)

    center_of_mass = masked_atom_pos.sum(dim=1) / num_valid_atoms  # (N, 3)

    # Compute squared distances from center of mass
    diff = masked_atom_pos - center_of_mass.unsqueeze(1)  # (N, L, 3)
    sq_distances = (diff ** 2).sum(dim=-1) * atom_pos_mask  # (N, L)

    # Compute radius of gyration
    radius_of_gyration = torch.sqrt(sq_distances.sum(dim=1) / num_valid_atoms.squeeze(-1))  # (N,)

    return radius_of_gyration

def compare_radius_of_gyration_distributions(
    true_rg_values: np.ndarray | torch.Tensor,
    pred_rg_values: np.ndarray | torch.Tensor,
    save_path: str | Path,
    title: str = "Radius of Gyration Distribution (True vs Pred)",
    xlabel: str = "Radius of Gyration (Å)",
    ylabel: str = "Density",
    bins: int = 30,
):
    """
    Plot and save overlaid distributions (histograms) of true vs predicted radius of gyration.

    Args:
        true_rg_values: Ground-truth radius of gyration values. Shape: (N,)
        pred_rg_values: Predicted radius of gyration values. Shape: (M,)
        save_path: Path to save the figure.
        title: Title of the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        bins: Number of histogram bins.
    """
    # Convert to numpy if necessary
    if isinstance(true_rg_values, torch.Tensor):
        true_rg_values = true_rg_values.detach().cpu().numpy()
    if isinstance(pred_rg_values, torch.Tensor):
        pred_rg_values = pred_rg_values.detach().cpu().numpy()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove NaN or inf values
    true_rg_values = true_rg_values[np.isfinite(true_rg_values)]
    pred_rg_values = pred_rg_values[np.isfinite(pred_rg_values)]

    # Determine shared range
    all_values = np.concatenate([true_rg_values, pred_rg_values])
    range_min, range_max = np.nanmin(all_values), np.nanmax(all_values)

    # Create overlaid histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        true_rg_values,
        bins=bins,
        range=(range_min, range_max),
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        label=f"True (mean={np.mean(true_rg_values):.2f})",
    )
    ax.hist(
        pred_rg_values,
        bins=bins,
        range=(range_min, range_max),
        density=True,
        alpha=0.6,
        color="orange",
        edgecolor="black",
        label=f"Pred (mean={np.mean(pred_rg_values):.2f})",
    )

    # Plot mean lines
    ax.axvline(np.mean(true_rg_values), color="steelblue", linestyle="--", lw=1.5)
    ax.axvline(np.mean(pred_rg_values), color="orange", linestyle="--", lw=1.5)

    # Labels and aesthetics
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
