from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from numpy import ndarray
from team_gm import typecheck

from miniworld.data.mols import CIFMolAttached


@typecheck
def get_shortest_distances_from_multistructures(
    atom_pos: Float[torch.Tensor, "* N L 3"],
    atom_pos_mask: Bool[torch.Tensor, "* N L"],
    atom_to_res_idx: Int[torch.Tensor, "* L"],
    min_distance: float = 2.0,
    max_distance: float = 22.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute residue-level shortest distances and corresponding mask from atom coordinates.

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
    mask_i = atom_pos_mask[:, :, :, None]  # (B, N, L, 1)
    mask_j = atom_pos_mask[:, :, None, :]  # (B, N, 1, L)
    valid_atom_mask = mask_i & mask_j  # (B, N, L, L)

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
    residue_mask = mask_i_res & mask_j_res  # (B, R_max, R_max)

    # 3) Aggregate shortest distances to residue level using scatter-reduce (min)
    # Map (i, j) residue pairs to flat indices per batch
    ri = atom_to_res_idx.unsqueeze(2).expand(B, L, L)  # (B, L, L)
    rj = atom_to_res_idx.unsqueeze(1).expand(B, L, L)  # (B, L, L)
    pair_idx = ri * R_max + rj  # (B, L, L)

    block_size = R_max * R_max
    batch_offsets = (
        torch.arange(B, device=device).view(B, 1, 1) * block_size
    )  # (B, 1, 1)

    scatter_idx = batch_offsets + pair_idx  # (B, L, L)
    scatter_idx_flat = scatter_idx.reshape(-1)  # (B * L * L,)

    src = shortest_dist.reshape(-1)  # (B * L * L,)

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


@typecheck
def get_shortest_distances(
    atom_pos: Float[torch.Tensor, "* L 3"],
    atom_pos_mask: Bool[torch.Tensor, "* L"],
    atom_to_res_idx: Int[torch.Tensor, "* L"],
    min_distance: float = 2.0,
    max_distance: float = 22.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute residue-level shortest distances from atom positions."""
    device = atom_pos.device
    B, L, _ = atom_pos.shape

    # 1) Atom-level pairwise distances (B, N, L, L)
    diff = atom_pos[:, :, None, :] - atom_pos[:, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)  # (B, L, L)

    # Apply atom mask
    mask_i = atom_pos_mask[:, :, None]  # (B, L, 1)
    mask_j = atom_pos_mask[:, None, :]  # (B, 1, L)
    valid_atom_mask = mask_i & mask_j  # (B, L, L)

    dist = dist.masked_fill(~valid_atom_mask, max_distance)
    dist = dist.clamp(min=min_distance, max=max_distance)

    # 2) Build residue existence mask (B, R_max)
    R_max = int(atom_to_res_idx.max().item()) + 1

    # A position is valid if any atom at that position is valid
    residue_exists = torch.zeros(B, R_max, dtype=torch.bool, device=device)

    # Flatten and mark residues that actually appear
    batch_idx = torch.arange(B, device=device).expand(B, L).reshape(-1)
    flat_atom_to_res_idx = atom_to_res_idx.reshape(-1)
    flat_pos_mask = atom_pos_mask.reshape(-1)

    valid_batch_idx = batch_idx[flat_pos_mask]
    valid_res_idx = flat_atom_to_res_idx[flat_pos_mask]

    residue_exists[valid_batch_idx, valid_res_idx] = True

    # Residue pair mask: both residues must exist
    mask_i_res = residue_exists.unsqueeze(2)  # (B, R_max, 1)
    mask_j_res = residue_exists.unsqueeze(1)  # (B, 1, R_max)
    residue_mask = mask_i_res & mask_j_res  # (B, R_max, R_max)

    # 3) Aggregate shortest distances to residue level using scatter-reduce (min)
    # Map (i, j) residue pairs to flat indices per batch
    ri = atom_to_res_idx.unsqueeze(2).expand(B, L, L)  # (B, L, L)
    rj = atom_to_res_idx.unsqueeze(1).expand(B, L, L)  # (B, L, L)
    pair_idx = ri * R_max + rj  # (B, L, L)

    block_size = R_max * R_max
    batch_offsets = (
        torch.arange(B, device=device).view(B, 1, 1) * block_size
    )  # (B, 1, 1)

    scatter_idx = batch_offsets + pair_idx  # (B, L, L)
    scatter_idx_flat = scatter_idx.reshape(-1)  # (B * L * L,)

    src = dist.reshape(-1)  # (B * L * L,)

    out = torch.full(
        (B * block_size,),
        max_distance,
        dtype=dist.dtype,
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


@typecheck
def get_contact_map(
    atom_pos: Float[torch.Tensor, "* L 3"],
    atom_pos_mask: Bool[torch.Tensor, "* L"],
    atom_to_res_idx: Int[torch.Tensor, "* L"],
    contact_threshold: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute residue-level contact map from atom positions."""
    residue_dists, residue_mask = get_shortest_distances(
        atom_pos=atom_pos,
        atom_pos_mask=atom_pos_mask,
        atom_to_res_idx=atom_to_res_idx,
    )

    contact_map = (
        residue_dists <= contact_threshold
    ) & residue_mask  # (B, R_max, R_max)

    return contact_map, residue_mask


def pairwise_kabsch_rmsd(
    pred: torch.Tensor,  # (S, L, 3)
    true: torch.Tensor,  # (N, L, 3)
    true_mask: torch.Tensor,  # (N, L) (bool or 0/1)
) -> torch.Tensor:
    """Compute (S, N) RMSD between S predicted structures and N true structures using Kabsch alignment and per-true-structure atom masks."""
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
        true_valid_L = torch.isfinite(true[j]).all(dim=-1)
        valid = true_mask[j] & true_valid_L & pred_valid_L

        M = int(valid.sum())
        if M < 3:
            rmsd_list.append(torch.full((S,), float("nan"), device=device))
            continue

        xyz_t = true[j, valid]  # (M, 3)
        xyz_p = pred[:, valid]  # (S, M, 3)

        # Center
        t_c = xyz_t - xyz_t.mean(dim=0, keepdim=True)  # (M, 3)
        p_c = xyz_p - xyz_p.mean(dim=1, keepdim=True)  # (S, M, 3)

        # Covariance
        t_c_exp = t_c.unsqueeze(0).expand(S, -1, -1)  # (S, M, 3)
        C = torch.bmm(p_c.transpose(1, 2), t_c_exp)  # (S, 3, 3)

        # SVD
        U, _, Vh = torch.linalg.svd(C, full_matrices=False)  # U,Vh: (S,3,3)
        V = Vh.transpose(-2, -1)  # (S, 3, 3)

        # Reflection correction using det(V U^T)
        detVU = torch.linalg.det(V @ U.transpose(-2, -1))  # (S,)
        sign = torch.where(detVU < 0, -torch.ones_like(detVU), torch.ones_like(detVU))
        D = torch.zeros_like(C)
        D[..., 0, 0] = 1.0
        D[..., 1, 1] = 1.0
        D[..., 2, 2] = sign
        R = V @ D @ U.transpose(-2, -1)  # (S, 3, 3)

        # Align and RMSD
        p_aligned = torch.bmm(p_c, R)  # (S, M, 3)
        diff2 = (p_aligned - t_c_exp).pow(2).sum(dim=-1)  # (S, M)
        rmsd_j = torch.sqrt(diff2.mean(dim=-1))  # (S,)
        rmsd_list.append(rmsd_j)

    return torch.stack(rmsd_list, dim=1)  # (S, N)


def save_rmsd_boxplot(
    data_2d: np.ndarray | torch.Tensor,
    save_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str = "RMSD",
) -> None:
    """Save a boxplot of RMSD values."""
    if isinstance(data_2d, torch.Tensor):
        data_2d = data_2d.detach().cpu().numpy()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort rows by their minimum value
    order = np.argsort(np.nanmin(data_2d, axis=1))
    data_sorted = data_2d[order]

    num_boxes = data_sorted.shape[0]
    ylim = [0, np.nanmax(data_2d) * 1.1]

    fig, ax = plt.subplots(figsize=(num_boxes * 0.25 + 2, 4))
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


def plot_rmsd_heatmap(
    rmsd_pp: torch.Tensor | np.ndarray,
    save_path: str | Path,
    title: str = "Pred-Pred RMSD",
) -> None:
    """Plot and save a 2D RMSD map between predicted structures.

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
    """Calculate radius of gyration for each structure in the batch.

    Args:
        atom_pos: Atomic coordinates. Shape: (N, L, 3)
        atom_pos_mask: Atom mask. True for valid atoms. Shape: (N, L)

    Returns:
        radius_of_gyration: Radius of gyration for each structure. Shape: (N,)

    """
    N, L, _ = atom_pos.shape

    # Compute center of mass for each structure
    masked_atom_pos = atom_pos * atom_pos_mask.unsqueeze(-1)  # (N, L, 3)
    num_valid_atoms = atom_pos_mask.sum(dim=1).unsqueeze(-1)  # (N, 1)

    center_of_mass = masked_atom_pos.sum(dim=1) / num_valid_atoms  # (N, 3)

    # Compute squared distances from center of mass
    diff = masked_atom_pos - center_of_mass.unsqueeze(1)  # (N, L, 3)
    sq_distances = (diff**2).sum(dim=-1) * atom_pos_mask  # (N, L)

    # Compute radius of gyration
    return torch.sqrt(
        sq_distances.sum(dim=1) / num_valid_atoms.squeeze(-1),
    )  # (N,)


def compare_radius_of_gyration_distributions(
    true_rg_values: np.ndarray | torch.Tensor,
    pred_rg_values: np.ndarray | torch.Tensor,
    save_path: str | Path,
    title: str = "Radius of Gyration Distribution (True vs Pred)",
    xlabel: str = "Radius of Gyration (Å)",
    ylabel: str = "Density",
    bins: int = 30,
) -> None:
    """Plot and save overlaid distributions (histograms) of true vs predicted radius of gyration.

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


def neighbor_list_grid(  # noqa: C901, PLR0915
    xyz: np.ndarray,
    d_thr: float,
    n_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute neighbor list using grid-based spatial partitioning."""
    n_atom = xyz.shape[0]
    nbrs = np.full((n_atom, n_max), -1, dtype=np.int64)
    counts = np.zeros(n_atom, dtype=np.int32)

    # 1) Mask invalid atoms (any NaN)
    valid = np.all(np.isfinite(xyz), axis=1)
    if not np.any(valid):
        return nbrs, counts

    # 2) Compressed array of valid points
    valid_xyz = xyz[valid]
    n_valid = valid_xyz.shape[0]

    # 3) Discretize into cells of side length d_thr (int64 coordinates)
    cell = np.floor(valid_xyz / d_thr).astype(np.int64)  # (n_valid, 3)

    # 4) Group points by cell via lexicographic sort on (x, y, z)
    order = np.lexsort((cell[:, 2], cell[:, 1], cell[:, 0]))
    cell_sorted = cell[order]

    # 5) Unique cells and their spans [start, end) in the sorted index space
    if n_valid > 1:
        change = np.any(np.diff(cell_sorted, axis=0) != 0, axis=1)
        starts = np.concatenate(([0], np.nonzero(change)[0] + 1))
    else:
        starts = np.array([0], dtype=np.int64)
    ends = np.concatenate((starts[1:], [n_valid]))
    unique_cells = cell_sorted[starts]  # (n_unique, 3)
    n_unique = unique_cells.shape[0]

    # 6) Helper: view (n,3) int64 as a structured dtype for consistent numeric lex compare
    #    (little-endian int64 x,y,z). This matches the lexsort order above.
    def as_struct3(a_int64x3: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(a_int64x3)
        dt = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
        return a.view(dt).ravel()

    unique_struct = as_struct3(unique_cells)

    # 7) Precompute inverse map from compressed->sorted if needed later
    inv_order = np.empty(n_valid, dtype=np.int64)
    inv_order[order] = np.arange(n_valid)

    # 8) 27 neighbor-cell offsets (-1,0,1)^3
    offsets = (
        np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"))
        .reshape(3, -1)
        .T
    )  # (27, 3)

    # 9) Accumulate candidate (i,j) pairs in compressed indices
    pair_i = []
    pair_j = []

    # For each offset, find matching neighbor cells and generate Cartesian pairs
    for off in offsets:
        # neighbor cells for ALL existing unique cells under this offset
        nei_cells = unique_cells + off  # (n_unique, 3)
        nei_struct = as_struct3(nei_cells)

        # Search where these neighbor cells would be, under the SAME structured ordering
        pos = np.searchsorted(unique_struct, nei_struct, side="left")
        in_bounds = pos < n_unique

        ok = np.zeros_like(in_bounds, dtype=bool)
        if np.any(in_bounds):
            ok[in_bounds] = unique_struct[pos[in_bounds]] == nei_struct[in_bounds]

        if not np.any(ok):
            continue

        # Source cell ids are those indices where a neighbor cell exists
        src_c = np.nonzero(ok)[0]  # indices in [0..n_unique)
        dst_c = pos[ok]  # matching neighbor cell ids

        # Build index ranges for points in each cell's span (sorted space indices)
        src_ranges = [np.arange(starts[c], ends[c], dtype=np.int64) for c in src_c]
        dst_ranges = [np.arange(starts[c], ends[c], dtype=np.int64) for c in dst_c]

        if len(src_ranges) == 0:
            continue

        # Cartesian product per (src_cell, dst_cell) pair (vectorized at cell level)
        src_idx_sorted = np.concatenate(
            [
                np.repeat(r, len(d))
                for r, d in zip(src_ranges, dst_ranges, strict=False)
            ],
        )
        if src_idx_sorted.size == 0:
            continue
        dst_idx_sorted = np.concatenate(
            [np.tile(d, len(r)) for r, d in zip(src_ranges, dst_ranges, strict=False)],
        )

        # Map back from sorted space → compressed (unsorted) space
        src_idx = order[src_idx_sorted]
        dst_idx = order[dst_idx_sorted]

        # Drop self-pairs
        keep = src_idx != dst_idx
        if not np.any(keep):
            continue

        pair_i.append(src_idx[keep])
        pair_j.append(dst_idx[keep])

    if not pair_i:
        # No candidate pairs; return empty neighbor lists
        return nbrs, counts

    pair_i = np.concatenate(pair_i)
    pair_j = np.concatenate(pair_j)

    # 10) Distance filtering: keep pairs with ||valid_xyz[i]-valid_xyz[j]|| <= d_thr
    dvec = valid_xyz[pair_i] - valid_xyz[pair_j]
    dist2 = np.einsum("ij,ij->i", dvec, dvec)
    keep = dist2 <= (d_thr * d_thr)
    if not np.any(keep):
        return nbrs, counts

    pair_i = pair_i[keep]
    pair_j = pair_j[keep]

    # 11) Remove duplicate pairs (same (i,j) can appear via multiple offsets)
    ij = np.stack([pair_i, pair_j], axis=1).astype(np.int64)
    ij_packed = ij.view(np.dtype((np.void, ij.dtype.itemsize * 2))).ravel()
    uniq_idx = np.unique(ij_packed, return_index=True)[1]
    ij = ij[uniq_idx]
    pair_i, pair_j = ij[:, 0], ij[:, 1]

    # 12) Map compressed indices back to original n_atom-space
    valid_true_idx = np.flatnonzero(valid)
    gi = valid_true_idx[pair_i]
    gj = valid_true_idx[pair_j]

    # 13) Fill neighbor matrix: group by gi, keep up to n_max in order of (gi, gj)
    order_fill = np.lexsort((gj, gi))
    gi = gi[order_fill]
    gj = gj[order_fill]

    uniq_i, first_pos, counts_all = np.unique(gi, return_index=True, return_counts=True)
    take_counts = np.minimum(counts_all, n_max)

    if uniq_i.size > 0:
        gather_idx = np.concatenate(
            [
                np.arange(s, s + t, dtype=np.int64)
                for s, t in zip(first_pos, take_counts, strict=True)
            ],
        )
        gi_take = gi[gather_idx]
        gj_take = gj[gather_idx]

        # Relative slot [0..taken-1] within each group
        rel = np.arange(gj.size, dtype=np.int64) - np.repeat(first_pos, counts_all)
        rel = rel[gather_idx]

        nbrs[gi_take, rel] = gj_take
        counts[uniq_i] = take_counts.astype(np.int32)

    return nbrs, counts


def cdist_clipped(
    xyz1: ndarray,
    xyz2: ndarray | None = None,
    d_thr: float = 32.0,
    n_max: int = 128,  # max neighbors per atom
) -> ndarray:
    """Compute a dense (n1, n2) distance map clipped at d_thr using neighbor_list_grid."""
    n1 = xyz1.shape[0]
    n2 = 0 if xyz2 is None else xyz2.shape[0]
    # Dense map clipped at threshold
    dist = np.full((n1, n2), d_thr, dtype=xyz1.dtype)

    # Combine xyz1 + xyz2 (neighbor_list_grid works on one array)
    xyz = np.concatenate([xyz1, xyz2], axis=0) if xyz2 is not None else xyz1

    # Build neighbor list over combined space
    nbrs, _ = neighbor_list_grid(xyz, d_thr, n_max)  # (n1+n2, n_max)

    # Only keep neighbors from xyz1 → xyz2
    valid_mask = (nbrs >= n1) & (nbrs < n1 + n2)
    if not np.any(valid_mask):
        return dist

    src_idx = np.broadcast_to(
        np.arange(n1 + n2, dtype=np.int64)[:, None],
        nbrs.shape,
    )[valid_mask]

    dst_idx = nbrs[valid_mask]

    mask_1_to_2 = (src_idx < n1) & (dst_idx >= n1)
    if not np.any(mask_1_to_2):
        return dist

    i = src_idx[mask_1_to_2]
    j = dst_idx[mask_1_to_2] - n1  # shift to [0..n2)

    dvec = xyz1[i] - xyz2[j]
    dist_ij = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))

    dist[i, j] = dist_ij

    return dist


def pdist_clipped(
    xyz: ndarray,
    d_thr: float = 32.0,
    n_max: int = 128,
) -> ndarray:
    """Compute a dense (n_atom, n_atom) distance map clipped at d_thr using neighbor_list_grid."""
    n_atom = xyz.shape[0]

    # Initialize with clipped distance
    dist = np.full((n_atom, n_atom), d_thr, dtype=xyz.dtype)
    np.fill_diagonal(dist, 0.0)

    # Use existing neighbor list (fast spatial grid)
    nbrs, _ = neighbor_list_grid(xyz, d_thr, n_max)  # (n_atom, n_max), -1 padded

    # Gather all valid (i, j) pairs in one shot
    valid_mask = nbrs != -1
    if not np.any(valid_mask):
        return dist

    row_idx = np.broadcast_to(
        np.arange(n_atom, dtype=np.int64)[:, None],
        nbrs.shape,
    )[valid_mask]
    col_idx = nbrs[valid_mask]

    # Compute exact distances for neighbor pairs
    dvec = xyz[row_idx] - xyz[col_idx]
    dist_ij = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))

    # Write symmetric distances (no Python loops)
    dist[row_idx, col_idx] = dist_ij
    dist[col_idx, row_idx] = dist_ij

    return dist


def extract_residue_com(cifmol: CIFMolAttached) -> np.ndarray:
    """Extract residue center of mass coordinates from a CIFMolAttached object."""
    atom_to_residue_idx_map = cifmol.index_table.atom_to_res
    xyz = cifmol.atoms.xyz.value

    n_res = atom_to_residue_idx_map.max() + 1

    valid_mask = ~np.isnan(xyz)  # shape (n_atoms, 3)

    xyz_zeroed = np.nan_to_num(xyz, nan=0.0)

    res_xyz_sum = np.zeros((n_res, 3), dtype=xyz.dtype)
    res_valid_count = np.zeros((n_res, 3), dtype=int)

    for i in range(3):
        res_xyz_sum[:, i] = np.bincount(
            atom_to_residue_idx_map,
            weights=xyz_zeroed[:, i],
            minlength=n_res,
        )
        res_valid_count[:, i] = np.bincount(
            atom_to_residue_idx_map,
            weights=valid_mask[:, i].astype(int),
            minlength=n_res,
        )

    res_center = np.full((n_res, 3), np.nan, dtype=xyz.dtype)
    nonzero = res_valid_count > 0
    res_center[nonzero] = res_xyz_sum[nonzero] / res_valid_count[nonzero]

    return res_center
