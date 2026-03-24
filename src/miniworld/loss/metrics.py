from collections.abc import Sequence

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from team_gm import typecheck

from miniworld.data.constants import EntityMapping, MoleculeType
from miniworld.data.features import Batch
from miniworld.utils import to_numpy

Array = np.ndarray | torch.Tensor


# Derived from https://github.com/nghiaho12/rigid_transform_3D
@typecheck
def align_pos(
    prb_pos: Float[np.ndarray, "L 3"],
    ref_pos: Float[np.ndarray, "L 3"],
) -> tuple[
    Float[np.ndarray, "L 3"],
    Float[np.ndarray, "3 3"],
    Float[np.ndarray, "3"],
]:
    """Align probe positions to reference positions.

    Parameters
    ----------
    prb_pos: np.ndarray, [L, 3]
        Probe atom positions.
    ref_pos: np.ndarray, [L, 3]
        Reference atom positions.

    Returns
    -------
    aligned_prb_pos: np.ndarray, [L, 3]
        Aligned probe atom positions.
    R: np.ndarray, [3, 3]
        Rotation matrix.
    T: np.ndarray, [3]
        Translation vector.

    """
    if np.isnan(prb_pos).any() or np.isnan(ref_pos).any():
        msg = f"NaN in input positions. {prb_pos} {ref_pos}"
        raise ValueError(msg)
    prb_CoM = np.mean(prb_pos, axis=0)
    ref_CoM = np.mean(ref_pos, axis=0)

    # find rotation
    H = (prb_pos - prb_CoM).T @ (ref_pos - ref_CoM)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    T = -R @ prb_CoM + ref_CoM
    aligned_prb_pos = R @ prb_pos.T + T.reshape(3, 1)

    return aligned_prb_pos.T, R, T


@typecheck
def cal_aligned_rmsd(
    prb_pos: Float[Array, "L 3"],
    ref_pos: Float[Array, "L 3"],
    res_mask: Bool[Array, "L"] | None = None,  # noqa: F821
) -> float:
    """Calculate RMSD of two sets of atom positions.

    Positions will be aligned before calculating RMSD.

    Parameters
    ----------
    prb_pos: ndarray or FloatTensor, (L, 3)
        Predicted atom positions.
    ref_pos: ndarray or FloatTensor, (L, 3)
        Reference atom positions.
    res_mask: ndarray or BoolTensor, (L)
        Mask of valid residues.

    Returns
    -------
    rmsd: float
        RMSD of two sets of atom positions.

    """
    if isinstance(prb_pos, torch.Tensor):
        prb_pos = to_numpy(prb_pos)
    if isinstance(ref_pos, torch.Tensor):
        ref_pos = to_numpy(ref_pos)
    if isinstance(res_mask, torch.Tensor):
        res_mask = to_numpy(res_mask)

    if res_mask is not None:
        non_gap_idx = np.where(~np.isnan(ref_pos).any(-1) & res_mask)[0]
    else:
        non_gap_idx = np.where(~np.isnan(ref_pos).any(-1))[0]
    if np.isnan(prb_pos[non_gap_idx]).any():
        msg = f"NaN in predicted positions. {prb_pos[non_gap_idx]}"
        raise ValueError(msg)
    aligned_prb_pos, _, _ = align_pos(prb_pos[non_gap_idx], ref_pos[non_gap_idx])
    rmsd = np.mean(np.linalg.norm(aligned_prb_pos - ref_pos[non_gap_idx], axis=-1))
    return rmsd.item()


@typecheck
def cal_atom_lddt(
    pred_atom_pos: Float[Array, "L 3"] | Float[Array, "B L 3"],
    gt_atom_pos: Float[Array, "L 3"],
    atom_mask: Bool[Array, "L"],  # noqa: F821
    max_distance: float = 15.0,
    distance_bins: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """Calculate lDDT score of two sets of atom positions.

    Supports both [L, 3] and [B, L, 3] predicted positions.
    Ground-truth positions and atom mask are shared across the batch.
    When a batch dimension exists, the function returns the mean lDDT
    score over the batch.
    """
    # Convert numpy arrays to tensors
    if isinstance(pred_atom_pos, np.ndarray):
        pred_atom_pos = torch.from_numpy(pred_atom_pos)
    if isinstance(gt_atom_pos, np.ndarray):
        gt_atom_pos = torch.from_numpy(gt_atom_pos)
    if isinstance(atom_mask, np.ndarray):
        atom_mask = torch.from_numpy(atom_mask)

    # Detect and normalize batch dimension on predicted positions
    single = pred_atom_pos.ndim == 2  # [L, 3]
    if single:
        pred_atom_pos = pred_atom_pos.unsqueeze(0)  # [1, L, 3]

    # Ensure dtypes and device
    device = pred_atom_pos.device
    pred_atom_pos = pred_atom_pos.to(device=device, dtype=torch.float32)  # [B, L, 3]
    gt_atom_pos = gt_atom_pos.to(device=device, dtype=torch.float32)  # [L, 3]
    atom_mask = atom_mask.to(device=device, dtype=torch.bool)  # [L]

    B, L, _ = pred_atom_pos.shape

    # Pairwise distance matrix for predicted positions: [B, L, L]
    pred_diff = pred_atom_pos[:, None, :, :] - pred_atom_pos[:, :, None, :]
    pred_dist = torch.norm(pred_diff, dim=-1)  # [B, L, L]

    # Pairwise distance matrix for ground truth: [L, L]
    gt_diff = gt_atom_pos[None, None, :, :] - gt_atom_pos[None, :, None, :]
    gt_dist = torch.norm(gt_diff, dim=-1)[0]  # [L, L]

    # Valid pair mask based on atom mask and ground-truth distances: [L, L]
    pair_mask = atom_mask[:, None] & atom_mask[None, :]
    pair_mask &= gt_dist > 0.0
    pair_mask &= gt_dist < max_distance

    # Delta distance: [B, L, L]
    delta = torch.abs(pred_dist - gt_dist)  # gt_dist is broadcast over batch

    # Vectorized distance bin handling
    bins = torch.tensor(distance_bins, dtype=torch.float32, device=device)  # [K]
    # cond: atoms within each distance bin and valid pairs -> [B, L, L, K]
    cond = (delta.unsqueeze(-1) <= bins) & pair_mask.unsqueeze(-1)

    # Count neighbors per atom and bin: sum over neighbor dimension (j)
    num_neighbors_in_bin = cond.sum(dim=2)

    # Total valid neighbors per atom (shared across batch): [L, 1]
    total_neighbors = pair_mask.sum(dim=-1, keepdim=True).float()  # [L, 1]

    # Fraction per bin: [B, L, K] (broadcast over batch)
    frac_per_bin = num_neighbors_in_bin.float() / (total_neighbors + 1e-8)

    # Mean over bins to get per-atom lDDT: [B, L]
    per_atom_lddt = frac_per_bin.mean(dim=-1)

    # Aggregate per-structure lDDT using atom mask: [B]
    atom_mask_f = atom_mask.float().unsqueeze(0)  # [1, L] -> broadcast to [B, L]
    per_struct_lddt = (per_atom_lddt * atom_mask_f).sum(dim=-1) / (
        atom_mask_f.sum(dim=-1) + 1e-8
    )  # [B]

    # Return scalar
    if single:
        return float(per_struct_lddt[0].item())
    return float(per_struct_lddt.mean().item())


@typecheck
def cal_atom_interface_lddt(
    pred_atom_pos: Float[Array, "L 3"],
    gt_atom_pos: Float[Array, "L 3"],
    atom_mask: Bool[Array, "L"],  # noqa: F821
    chain_mask: Bool[Array, "L L"],
    max_distance: float = 15.0,
    distance_bins: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """Calculate interface lDDT score of two sets of atom positions.

    Supports both [L, 3] and [B, L, 3] predicted positions.
    Ground-truth positions and atom mask are shared across the batch.
    Only inter-chain atom pairs are considered when computing lDDT.
    """
    # Convert numpy arrays to tensors
    if isinstance(pred_atom_pos, np.ndarray):
        pred_atom_pos = torch.from_numpy(pred_atom_pos)
    if isinstance(gt_atom_pos, np.ndarray):
        gt_atom_pos = torch.from_numpy(gt_atom_pos)
    if isinstance(atom_mask, np.ndarray):
        atom_mask = torch.from_numpy(atom_mask)
    if isinstance(chain_mask, np.ndarray):
        chain_mask = torch.from_numpy(chain_mask)

    # Detect and normalize batch dimension on predicted positions
    single = pred_atom_pos.ndim == 2  # [L, 3]
    if single:
        pred_atom_pos = pred_atom_pos.unsqueeze(0)  # [1, L, 3]

    # Ensure dtypes and device
    device = pred_atom_pos.device
    pred_atom_pos = pred_atom_pos.to(device=device, dtype=torch.float32)  # [B, L, 3]
    gt_atom_pos = gt_atom_pos.to(device=device, dtype=torch.float32)  # [L, 3]
    atom_mask = atom_mask.to(device=device, dtype=torch.bool)  # [L]

    B, L, _ = pred_atom_pos.shape

    # Pairwise distance matrix for predicted positions: [B, L, L]
    pred_diff = pred_atom_pos[:, None, :, :] - pred_atom_pos[:, :, None, :]
    pred_dist = torch.norm(pred_diff, dim=-1)  # [B, L, L]

    # Pairwise distance matrix for ground truth: [L, L]
    gt_diff = gt_atom_pos[None, None, :, :] - gt_atom_pos[None, :, None, :]
    gt_dist = torch.norm(gt_diff, dim=-1)[0]  # [L, L]

    # Base valid pair mask (same for all batches): [L, L]
    pair_mask = atom_mask[:, None] & atom_mask[None, :]
    pair_mask &= gt_dist > 0.0
    pair_mask &= gt_dist < max_distance

    # Keep only inter-chain pairs
    pair_mask &= ~chain_mask.bool()  # [L, L]

    # Delta distance: [B, L, L]
    delta = torch.abs(pred_dist - gt_dist)

    # Vectorized distance bin handling
    bins = torch.tensor(distance_bins, dtype=torch.float32, device=device)  # [K]
    # cond: atoms within each distance bin and valid interface pairs -> [B, L, L, K]
    cond = (delta.unsqueeze(-1) <= bins) & pair_mask.unsqueeze(-1)

    # Count neighbors per atom and bin: sum over neighbor dimension (j)
    num_neighbors_in_bin = cond.sum(dim=2)

    # Total valid neighbors per atom (shared across batch): [L, 1]
    total_neighbors = pair_mask.sum(dim=-1, keepdim=True).float()  # [L, 1]

    # Fraction per bin: [B, L, K] (broadcast over batch)
    frac_per_bin = num_neighbors_in_bin.float() / (total_neighbors + 1e-8)

    # Mean over bins to get per-atom lDDT: [B, L]
    per_atom_lddt = frac_per_bin.mean(dim=-1)

    # Aggregate per-structure lDDT using atom mask: [B]
    atom_mask_f = atom_mask.float().unsqueeze(0)  # [1, L] -> broadcast to [B, L]
    per_struct_lddt = (per_atom_lddt * atom_mask_f).sum(dim=-1) / (
        atom_mask_f.sum(dim=-1) + 1e-8
    )  # [B]

    # Return scalar
    if single:
        return float(per_struct_lddt[0].item())
    return float(per_struct_lddt.mean().item())


def build_atom_chain_map_and_mask(
    atom_to_residue_idx_map: Int[torch.Tensor, "B L_atom"],
    residue_asym_id: Int[torch.Tensor, "B L_res"],
) -> tuple[Int[torch.Tensor, "B L_atom"], Int[torch.Tensor, "L_atom L_atom"]]:
    """Build atom to chain index mapping and chain mask."""
    atom_to_residue_idx_map = atom_to_residue_idx_map[0]  # (L_atom,)
    residue_asym_id = residue_asym_id[0]  # (L_res,)
    atom_to_chain_idx_map = residue_asym_id[atom_to_residue_idx_map]  # (L_atom,)

    chain_i = atom_to_chain_idx_map[:, None]  # (L_atom, 1)
    chain_j = atom_to_chain_idx_map[None, :]  # (1, L_atom)

    same_chain_bool: Bool[torch.Tensor, "L_atom L_atom"] = chain_i == chain_j
    chain_mask: Int[torch.Tensor, "L_atom L_atom"] = same_chain_bool.to(torch.int64)

    return atom_to_chain_idx_map, chain_mask


def category_lddt(  # noqa: C901, PLR0915, PLR0912
    batch: Batch,
    pred_atom_pos: Float[Array, "L 3"],
    distance_bins: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> dict[str, list[float]]:
    """Calculate lDDT for different categories of molecular complexes."""
    gt_atom_pos = batch.structure.atom_pos[0]
    atom_mask = batch.structure.atom_mask[0].bool()
    lddt_dict = {
        "intra_protein": [],  # 0
        "intra_dprotein": [],  # 1
        "intra_DNA": [],  # 2
        "intra_RNA": [],  # 3
        "intra_ligand": [],  # 4
        "protein-antibody": [],  # 5
        "protein-protein": [],  # 6
        "protein-DNA": [],  # 7
        "protein-RNA": [],  # 8
        "protein-ligand": [],  # 9
        "DNA-DNA": [],  # 10
        "DNA-RNA": [],  # 11
        "RNA-RNA": [],  # 12
        "NA-ligand": [],  # 13
        "total": [],  # 14
    }
    NA_related = [
        "intra_DNA",
        "intra_RNA",
        "protein-DNA",
        "protein-RNA",
        "DNA-DNA",
        "DNA-RNA",
        "RNA-RNA",
        "NA-ligand",
    ]

    entity_type_list = batch.chain.entity_type[0].tolist()
    contact_edges = batch.chain.contact_edges[0].tolist()
    atom_to_chain_idx_map, chain_mask = build_atom_chain_map_and_mask(
        atom_to_residue_idx_map=batch.scheme.atom_to_residue_idx_map,
        residue_asym_id=batch.scheme.residue_asym_id,
    )
    entity_mapping = EntityMapping()

    for chain_idx, entity_type_idx in enumerate(entity_type_list):
        atom_idx = torch.where(atom_to_chain_idx_map == chain_idx)[0]
        entity_type = entity_mapping.idx_to_type(entity_type_idx)
        match entity_type:
            case MoleculeType.ANTIBODY | MoleculeType.PROTEIN:
                _type = "intra_protein"
            case MoleculeType.DPROTEIN:
                _type = "intra_dprotein"
            case MoleculeType.DNA:
                _type = "intra_DNA"
            case MoleculeType.RNA:
                _type = "intra_RNA"
            case MoleculeType.LIGAND | MoleculeType.BRANCHED:
                _type = "intra_ligand"
            case _:
                _type = None
        if _type is None:
            continue
        pred_pos = pred_atom_pos[atom_idx]
        gt_pos = gt_atom_pos[atom_idx]
        mask = atom_mask[atom_idx]
        if mask.sum() < 10:  # too small to calculate lddt
            continue
        max_distance = 30.0 if _type in NA_related else 15.0
        lddt = cal_atom_lddt(
            pred_atom_pos=pred_pos,
            gt_atom_pos=gt_pos,
            atom_mask=mask,
            max_distance=max_distance,
            distance_bins=distance_bins,
        )
        lddt_dict[_type].append(lddt)
    if len(contact_edges) == 0:
        NA_included = any(
            key in NA_related for key in lddt_dict if len(lddt_dict[key]) > 0
        )
        # no inter-chain contacts
        total_lddt = cal_atom_lddt(
            pred_atom_pos=pred_atom_pos,
            gt_atom_pos=gt_atom_pos,
            atom_mask=atom_mask,
            max_distance=30.0 if NA_included else 15.0,
            distance_bins=distance_bins,
        )
        lddt_dict["total"].append(total_lddt)
        return lddt_dict
    contact_src, contact_dst = zip(*contact_edges, strict=False)

    for _src, _dst in zip(contact_src, contact_dst, strict=True):
        src, dst = sorted((_src, _dst))
        atom_idx1 = torch.where(atom_to_chain_idx_map == src)[0]
        atom_idx2 = torch.where(atom_to_chain_idx_map == dst)[0]
        atom_idx = torch.cat((atom_idx1, atom_idx2), dim=0)
        entity_type1 = entity_mapping.idx_to_type(entity_type_list[src])[0]
        entity_type2 = entity_mapping.idx_to_type(entity_type_list[dst])[0]
        edge_type = None
        if entity_type1 == MoleculeType.PROTEIN:
            match entity_type2:
                case MoleculeType.ANTIBODY:
                    edge_type = "protein-antibody"
                case MoleculeType.PROTEIN:
                    edge_type = "protein-protein"
                case MoleculeType.DNA:
                    edge_type = "protein-DNA"
                case MoleculeType.RNA:
                    edge_type = "protein-RNA"
                case MoleculeType.LIGAND | MoleculeType.BRANCHED:
                    edge_type = "protein-ligand"
        elif entity_type1 == MoleculeType.DNA:
            match entity_type2:
                case MoleculeType.DNA:
                    edge_type = "DNA-DNA"
                case MoleculeType.RNA:
                    edge_type = "DNA-RNA"
                case MoleculeType.LIGAND | MoleculeType.BRANCHED:
                    edge_type = "NA-ligand"
        elif entity_type1 == MoleculeType.RNA:
            match entity_type2:
                case MoleculeType.RNA:
                    edge_type = "RNA-RNA"
                case MoleculeType.LIGAND | MoleculeType.BRANCHED:
                    edge_type = "NA-ligand"
        if edge_type is None:
            continue
        pred_pos = pred_atom_pos[atom_idx]
        gt_pos = gt_atom_pos[atom_idx]
        mask = atom_mask[atom_idx]
        max_distance = 30.0 if edge_type in NA_related else 15.0
        lddt = cal_atom_interface_lddt(
            pred_atom_pos=pred_pos,
            gt_atom_pos=gt_pos,
            atom_mask=mask,
            chain_mask=chain_mask[atom_idx][:, atom_idx],
            max_distance=max_distance,
            distance_bins=distance_bins,
        )
        lddt_dict[edge_type].append(lddt)

    NA_included = any(
        key in NA_related and len(lddt_dict[key]) > 0 for key in NA_related
    )

    total_lddt = cal_atom_lddt(
        pred_atom_pos=pred_atom_pos,
        gt_atom_pos=gt_atom_pos,
        atom_mask=atom_mask,
        max_distance=30.0 if NA_included else 15.0,
        distance_bins=distance_bins,
    )
    lddt_dict["total"].append(total_lddt)

    return lddt_dict
