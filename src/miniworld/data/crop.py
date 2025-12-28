import random

import numpy as np
from biomol.cif.mol import CIFMol
from numpy import ndarray

from miniworld.utils.structure import extract_residue_com, pdist_clipped


class NoInterfaceError(Exception):
    """Raised when no interface is found in the biomolecular structure."""


def get_chain_crop_indices(
    cifmol: CIFMol,
    crop_indices: ndarray,
) -> dict[str, ndarray]:
    """Get chain-wise crop indices."""
    chain_crop = {}
    chain_list = cifmol.chains.chain_id.value

    for chain in chain_list:
        residue_indices = cifmol.index_table.chains_to_residues(
            np.where(cifmol.chains.chain_id == chain)[0],
        )
        residue_indices = np.unique(residue_indices)
        chain_crop_indices = (
            np.intersect1d(residue_indices, crop_indices) - residue_indices[0]
        )
        if chain_crop_indices.size == 0:
            continue
        chain_crop[chain] = chain_crop_indices

    return chain_crop


def get_valid_indices(cifmol: CIFMol) -> ndarray:
    """Get valid residue indices with non-nan atom coordinates."""
    valid_atoms_indices = np.any(~np.isnan(cifmol.atoms.xyz), axis=-1)
    valid_residue_indices = cifmol.index_table.atoms_to_residues(valid_atoms_indices)
    return np.unique(valid_residue_indices)


def crop_contiguous_monomer(
    interface_bias: str | None,
    cifmol: CIFMol,
    crop_length: int,
) -> tuple[ndarray, dict[str, ndarray]]:
    """Crop the structure and sequence into a contiguous region (polymer only)."""
    valid_indices = get_valid_indices(cifmol)
    valid_chain_list = cifmol.residues[valid_indices].chains.chain_id.value
    if len(valid_chain_list) == 0:
        cifID = cifmol.id[0]
        msg = f"No valid residues found in the biomolstructure {cifID}."
        raise ValueError(msg)

    # choose a random polymer chain. (If there is no polymer chain, choose a random chain.)
    polymer_chain_list = cifmol.chains[
        cifmol.chains.entity_type != "non-polymer"
    ].chain_id.value
    valid_polymer_chain_list = np.intersect1d(valid_chain_list, polymer_chain_list)
    if len(valid_polymer_chain_list) > 0:
        valid_chain_list = valid_polymer_chain_list

    if interface_bias is not None:
        if (
            interface_bias[0] not in valid_chain_list
            and interface_bias[1] not in valid_chain_list
        ):
            msg = f"Invalid chain: {interface_bias} \
            valid_chain_list = {valid_chain_list} at cifmol {cifmol.id}"
            raise ValueError(msg)
        if interface_bias[0] not in valid_chain_list:
            pivot_chain_id = interface_bias[1]
        elif interface_bias[1] not in valid_chain_list:
            pivot_chain_id = interface_bias[0]
        else:
            pivot_chain_id = random.choice(interface_bias)
    else:
        pivot_chain_id = np.random.choice(valid_chain_list)
    pivot_chain_indices = np.where(cifmol.chains.chain_id == pivot_chain_id)[0]
    pivot_residue_indices = cifmol.index_table.chains_to_residues(pivot_chain_indices)
    pivot_residue_indices = np.unique(pivot_residue_indices)
    pivot_valid_residue_indices = np.intersect1d(pivot_residue_indices, valid_indices)
    n_k = pivot_valid_residue_indices.size
    if n_k < crop_length:
        crop_indices = pivot_valid_residue_indices
    else:
        # uniformly sample a contiguous region
        n_start = random.randint(0, n_k - crop_length)
        crop_indices = pivot_valid_residue_indices[n_start : n_start + crop_length]
    return crop_indices, get_chain_crop_indices(cifmol, crop_indices)


def crop_spatial_atom(
    chain_bias: str | None,
    cifmol: CIFMol,
    crop_length: int,
) -> tuple[ndarray, dict[str, ndarray]]:
    """Crop the structure and sequence into a spatial region."""
    valid_indices = get_valid_indices(cifmol)
    if valid_indices.size < crop_length:
        chain_crop = get_chain_crop_indices(cifmol, valid_indices)
        return valid_indices, chain_crop
    valid_chain_list = cifmol.residues[valid_indices].chains.chain_id.value
    if chain_bias is not None:
        if chain_bias not in valid_chain_list:
            msg = f"Invalid chain: {chain_bias} \
            valid_chain_list = {valid_chain_list} at cifmol {cifmol.id}"
            raise ValueError(msg)
        pivot_chain_id = chain_bias
    else:
        pivot_chain_id = np.random.choice(valid_chain_list)
    pivot_chain_indices = np.where(cifmol.chains.chain_id == pivot_chain_id)[0]
    pivot_residue_indices = cifmol.index_table.chains_to_residues(pivot_chain_indices)
    pivot_residue_indices = np.unique(pivot_residue_indices)
    pivot_valid_residue_indices = np.intersect1d(pivot_residue_indices, valid_indices)

    pivot_residue_idx = np.random.choice(pivot_valid_residue_indices)
    pivot_atom_indices = cifmol.index_table.residues_to_atoms(
        np.array([pivot_residue_idx]),
    )
    valid_atom_indices = cifmol.index_table.residues_to_atoms(valid_indices)

    xyz = cifmol.atoms.xyz.value
    pivot_xyz = xyz[pivot_atom_indices]
    distance_map = np.linalg.norm(
        xyz[:, None, :] - pivot_xyz[None, :, :],
        axis=-1,
    )  # (N_atoms, N_pivot_atoms)
    distance_map[np.isnan(distance_map)] = float("inf")
    distance_map[~np.isin(np.arange(xyz.shape[0]), valid_atom_indices), :] = float(
        "inf",
    )
    distance_map = distance_map.min(axis=1)  # (N_atoms,)

    atom_to_res = cifmol.index_table.atom_to_res
    order = np.argsort(atom_to_res)
    res_sorted = atom_to_res[order]
    dist_sorted = distance_map[order]

    starts = np.flatnonzero(np.r_[True, res_sorted[1:] != res_sorted[:-1]])
    mins = np.minimum.reduceat(dist_sorted, starts)

    crop_indices = np.argsort(mins)[:crop_length]
    crop_indices = np.sort(crop_indices)
    return crop_indices, get_chain_crop_indices(cifmol, crop_indices)


def crop_spatial_interface(  # noqa: PLR0915
    interface_bias: tuple[str, str] | None,
    cifmol: CIFMol,
    crop_length: int,
    interface_distance_cutoff: float = 6.0,  # atom level
) -> tuple[ndarray, dict[str, ndarray]]:
    """Crop the structure and sequence into a spatial region."""
    valid_indices = get_valid_indices(cifmol)
    if valid_indices.size < crop_length:
        chain_crop = get_chain_crop_indices(cifmol, valid_indices)
        return valid_indices, chain_crop
    chain_list = cifmol.chains.chain_id.value
    valid_chain_list = cifmol.residues[valid_indices].chains.chain_id.value

    src, dst = cifmol.chains.contact.src_indices, cifmol.chains.contact.dst_indices
    edges = [(chain_list[src[i]], chain_list[dst[i]]) for i in range(len(src))]
    valid_edges = [
        (a, b) for (a, b) in edges if a in valid_chain_list and b in valid_chain_list
    ]

    if interface_bias is not None:
        if (
            interface_bias[0] not in valid_chain_list
            or interface_bias[1] not in valid_chain_list
        ):
            msg = f"Invalid chain: {interface_bias} \
            valid_chain_list = {valid_chain_list}"
            print(f"You have to check this cifmol object: {cifmol.id[0]}")
            raise NoInterfaceError(msg)
        if (interface_bias[0], interface_bias[1]) not in valid_edges and (
            interface_bias[1],
            interface_bias[0],
        ) not in valid_edges:
            msg = f"Invalid interface: {interface_bias} \
            valid_edges = {valid_edges}"
            print(f"You have to check this cifmol object: {cifmol.id[0]}")
            raise NoInterfaceError(msg)
        pivot_edge = interface_bias
    else:
        if len(valid_edges) == 0:
            msg = "No interfaces found in the biomolstructure."
            print(f"You have to check this cifmol object: {cifmol.id[0]}")
            raise NoInterfaceError(msg)
        pivot_edge = random.choice(valid_edges)
    xyz = cifmol.atoms.xyz.value
    src_chain_indices = np.where(cifmol.chains.chain_id == pivot_edge[0])[0]
    src_residue_indices = cifmol.index_table.chains_to_residues(src_chain_indices)
    src_residue_indices = np.unique(src_residue_indices)
    dst_chain_indices = np.where(cifmol.chains.chain_id == pivot_edge[1])[0]
    dst_residue_indices = cifmol.index_table.chains_to_residues(dst_chain_indices)
    dst_residue_indices = np.unique(dst_residue_indices)

    total_residue_indices = np.concatenate([src_residue_indices, dst_residue_indices])
    total_atom_indices = cifmol.index_table.residues_to_atoms(total_residue_indices)
    src_atom_indices = cifmol.index_table.residues_to_atoms(src_residue_indices)
    dst_atom_indices = cifmol.index_table.residues_to_atoms(dst_residue_indices)
    xyz_cropped = xyz[total_atom_indices]
    distance_map = pdist_clipped(
        xyz_cropped,
        d_thr=interface_distance_cutoff + 0.1,
        n_max=128,
    )

    distance_map[np.isnan(distance_map)] = float("inf")
    distance_map[np.arange(distance_map.shape[0]), np.arange(distance_map.shape[0])] = (
        float("inf")
    )

    interface_distance_map = distance_map[
        : len(src_atom_indices), len(src_atom_indices) :,
    ]  # (L_src, L_dst)

    interface_src_atoms_distance = interface_distance_map.min(axis=1)  # (L_src,)
    interface_dst_atoms_distance = interface_distance_map.min(axis=0)  # (L_dst,)

    src_atom_to_res = cifmol.index_table.atoms_to_residues(src_atom_indices)
    dst_atom_to_res = cifmol.index_table.atoms_to_residues(dst_atom_indices)

    src_order = np.argsort(src_atom_to_res)
    dst_order = np.argsort(dst_atom_to_res)
    src_res_sorted = src_atom_to_res[src_order]
    dst_res_sorted = dst_atom_to_res[dst_order]
    src_starts = np.flatnonzero(np.r_[True, src_res_sorted[1:] != src_res_sorted[:-1]])
    dst_starts = np.flatnonzero(np.r_[True, dst_res_sorted[1:] != dst_res_sorted[:-1]])

    src_dist_sorted = interface_src_atoms_distance[src_order]
    dst_dist_sorted = interface_dst_atoms_distance[dst_order]

    interface_src_residues_dist = np.minimum.reduceat(src_dist_sorted, src_starts)
    interface_dst_residues_dist = np.minimum.reduceat(dst_dist_sorted, dst_starts)

    interface_src_residues = np.where(
        interface_src_residues_dist <= interface_distance_cutoff,
    )[0]
    interface_dst_residues = np.where(
        interface_dst_residues_dist <= interface_distance_cutoff,
    )[0]

    interface_residues = np.concatenate(
        [
            src_residue_indices[interface_src_residues],
            dst_residue_indices[interface_dst_residues],
        ],
    )
    interface_residue_dist = np.concatenate(
        [
            interface_src_residues_dist[interface_src_residues],
            interface_dst_residues_dist[interface_dst_residues],
        ],
    )
    if len(interface_residues) >= crop_length:
        # sort by min distance to the interface
        order = np.argsort(interface_residue_dist)
        sorted_interface_residues = interface_residues[order]
        crop_indices = sorted_interface_residues[:crop_length]
        crop_indices = np.sort(crop_indices)
        chain_crop = get_chain_crop_indices(cifmol, crop_indices)
        return crop_indices, chain_crop

    interface_atoms = cifmol.index_table.residues_to_atoms(interface_residues)

    xyz_interface = xyz[interface_atoms]
    xyz_rest = xyz[np.setdiff1d(np.arange(xyz.shape[0]), interface_atoms)]

    xyz_interface_center = np.mean(xyz_interface, axis=0, keepdims=True)  # (1, 3)
    rest_min_distance = np.linalg.norm(xyz_rest - xyz_interface_center, axis=-1)

    rest_atom_to_res = cifmol.index_table.atoms_to_residues(
        np.setdiff1d(np.arange(xyz.shape[0]), interface_atoms),
    )
    rest_order = np.argsort(rest_atom_to_res)
    rest_res_sorted = rest_atom_to_res[rest_order]
    rest_starts = np.flatnonzero(
        np.r_[True, rest_res_sorted[1:] != rest_res_sorted[:-1]],
    )
    rest_min_distance_sorted = rest_min_distance[rest_order]
    rest_min_distance_per_res = np.minimum.reduceat(
        rest_min_distance_sorted, rest_starts,
    )
    rest_residue_indices = rest_res_sorted[rest_starts]

    remaining_length = crop_length - len(interface_residues)
    selected = np.argsort(rest_min_distance_per_res)[:remaining_length]
    rest_selected_indices = rest_residue_indices[selected]
    crop_indices = np.sort(np.concatenate([interface_residues, rest_selected_indices]))
    chain_crop = get_chain_crop_indices(cifmol, crop_indices)

    return crop_indices, chain_crop


def crop_spatial_residue(
    chain_bias: str | None,
    cifmol: CIFMol,
    crop_length: int,
) -> tuple[ndarray, dict[str, ndarray]]:
    """Crop the structure and sequence into a spatial region."""
    valid_indices = get_valid_indices(cifmol)
    if valid_indices.size < crop_length:
        chain_crop = get_chain_crop_indices(cifmol, valid_indices)
        return valid_indices, chain_crop
    valid_chain_list = cifmol.residues[valid_indices].chains.chain_id.value
    if chain_bias is not None:
        if chain_bias not in valid_chain_list:
            msg = f"Invalid chain: {chain_bias} \
            valid_chain_list = {valid_chain_list} at cifmol {cifmol.id}"
            raise ValueError(msg)
        pivot_chain_id = chain_bias
    else:
        pivot_chain_id = np.random.choice(valid_chain_list)
    pivot_chain_indices = np.where(cifmol.chains.chain_id == pivot_chain_id)[0]
    pivot_residue_indices = cifmol.index_table.chains_to_residues(pivot_chain_indices)
    pivot_residue_indices = np.unique(pivot_residue_indices)
    pivot_valid_residue_indices = np.intersect1d(pivot_residue_indices, valid_indices)
    pivot_residue_idx = np.random.choice(pivot_valid_residue_indices)

    residue_xyz = extract_residue_com(cifmol)

    pivot_xyz = residue_xyz[pivot_residue_idx]
    distance_map = np.linalg.norm(
        residue_xyz - pivot_xyz[None, :], axis=-1,
    )  # (N_residues, )
    distance_map[np.isnan(distance_map)] = float("inf")
    distance_map[~np.isin(np.arange(residue_xyz.shape[0]), valid_indices)] = float(
        "inf",
    )

    crop_indices = np.argsort(distance_map)[:crop_length]
    crop_indices = np.sort(crop_indices)
    return crop_indices, get_chain_crop_indices(cifmol, crop_indices)


def crop_spatial_interface_residue(  # noqa: PLR0915
    interface_bias: tuple[str, str] | None,
    cifmol: CIFMol,
    crop_length: int,
    interface_distance_cutoff: float = 12.0,  # residue level
) -> tuple[ndarray, dict[str, ndarray]]:
    """Crop the structure and sequence into a spatial region."""
    valid_indices = get_valid_indices(cifmol)
    if valid_indices.size < crop_length:
        chain_crop = get_chain_crop_indices(cifmol, valid_indices)
        return valid_indices, chain_crop
    chain_list = cifmol.chains.chain_id.value
    valid_chain_list = cifmol.residues[valid_indices].chains.chain_id.value

    src, dst = cifmol.chains.contact.src_indices, cifmol.chains.contact.dst_indices
    edges = [(chain_list[src[i]], chain_list[dst[i]]) for i in range(len(src))]
    valid_edges = [
        (a, b) for (a, b) in edges if a in valid_chain_list and b in valid_chain_list
    ]

    if interface_bias is not None:
        if (
            interface_bias[0] not in valid_chain_list
            or interface_bias[1] not in valid_chain_list
        ):
            msg = f"Invalid chain: {interface_bias} \
            valid_chain_list = {valid_chain_list}"
            print(f"You have to check this cifmol object: {cifmol.id[0]}")
            raise NoInterfaceError(msg)
        if (interface_bias[0], interface_bias[1]) not in valid_edges and (
            interface_bias[1],
            interface_bias[0],
        ) not in valid_edges:
            msg = f"Invalid interface: {interface_bias} \
            valid_edges = {valid_edges}"
            print(f"You have to check this cifmol object: {cifmol.id[0]}")
            raise NoInterfaceError(msg)
        pivot_edge = interface_bias
    else:
        if len(valid_edges) == 0:
            msg = "No interfaces found in the biomolstructure."
            raise NoInterfaceError(msg)
        pivot_edge = random.choice(valid_edges)
    residue_xyz = extract_residue_com(cifmol)
    src_chain_indices = np.where(cifmol.chains.chain_id == pivot_edge[0])[0]
    src_residue_indices = cifmol.index_table.chains_to_residues(src_chain_indices)
    src_residue_indices = np.unique(src_residue_indices)
    dst_chain_indices = np.where(cifmol.chains.chain_id == pivot_edge[1])[0]
    dst_residue_indices = cifmol.index_table.chains_to_residues(dst_chain_indices)
    dst_residue_indices = np.unique(dst_residue_indices)

    total_residue_indices = np.concatenate([src_residue_indices, dst_residue_indices])
    xyz_cropped = residue_xyz[total_residue_indices]
    distance_map = pdist_clipped(
        xyz_cropped,
        d_thr=interface_distance_cutoff + 0.1,
        n_max=128,
    )

    distance_map[np.isnan(distance_map)] = float("inf")
    distance_map[np.arange(distance_map.shape[0]), np.arange(distance_map.shape[0])] = (
        float("inf")
    )

    interface_distance_map = distance_map[
        : len(src_residue_indices), len(src_residue_indices) :,
    ]  # (L_src, L_dst)

    interface_src_distance = interface_distance_map.min(axis=1)  # (L_src,)
    interface_dst_distance = interface_distance_map.min(axis=0)  # (L_dst,)

    interface_src_residues = np.where(
        interface_src_distance <= interface_distance_cutoff,
    )[0]
    interface_dst_residues = np.where(
        interface_dst_distance <= interface_distance_cutoff,
    )[0]

    interface_residues = np.concatenate(
        [
            src_residue_indices[interface_src_residues],
            dst_residue_indices[interface_dst_residues],
        ],
    )
    interface_residue_dist = np.concatenate(
        [
            interface_src_distance[interface_src_residues],
            interface_dst_distance[interface_dst_residues],
        ],
    )
    if len(interface_residues) >= crop_length:
        # sort by min distance to the interface
        order = np.argsort(interface_residue_dist)
        sorted_interface_residues = interface_residues[order]
        crop_indices = sorted_interface_residues[:crop_length]
        crop_indices = np.sort(crop_indices)
        chain_crop = get_chain_crop_indices(cifmol, crop_indices)
        return crop_indices, chain_crop

    if interface_residues.size == 0:
        msg = "No interface residues found in the biomolstructure."
        raise NoInterfaceError(msg)

    xyz_interface = residue_xyz[interface_residues]
    distance_map = np.linalg.norm(
        residue_xyz[:, None, :] - xyz_interface[None, :, :], axis=-1,
    )  # (N_residues, N_interface_residues)
    distance_map[np.isnan(distance_map)] = float("inf")
    distance_map = distance_map.min(axis=1)  # (N_residues,)
    crop_indices = np.argsort(distance_map)[:crop_length]
    crop_indices = np.sort(crop_indices)
    chain_crop = get_chain_crop_indices(cifmol, crop_indices)

    return crop_indices, chain_crop


def _get_interface_residues(
    interface_bias: tuple[str, str] | None,
    cifmol: CIFMol,
    valid_indices: np.ndarray,
    interface_distance_cutoff: float,
) -> np.ndarray:
    """Return global residue indices that are interface residues for a chosen chain-pair."""
    chain_list = cifmol.chains.chain_id.value
    valid_chain_list = cifmol.residues[valid_indices].chains.chain_id.value

    src, dst = cifmol.chains.contact.src_indices, cifmol.chains.contact.dst_indices
    edges = [(chain_list[src[i]], chain_list[dst[i]]) for i in range(len(src))]
    valid_edges = [(a, b) for (a, b) in edges if a in valid_chain_list and b in valid_chain_list]

    if interface_bias is not None:
        a, b = interface_bias
        if (a not in valid_chain_list) or (b not in valid_chain_list):
            msg = f"Invalid chain: {interface_bias} valid_chain_list={valid_chain_list}"
            print(f"You have to check this cifmol object: {cifmol.id[0]}")
            raise NoInterfaceError(msg)
        if (a, b) not in valid_edges and (b, a) not in valid_edges:
            msg = f"Invalid interface: {interface_bias} valid_edges={valid_edges}"
            print(f"You have to check this cifmol object: {cifmol.id[0]}")
            raise NoInterfaceError(msg)
        pivot_edge = interface_bias
    else:
        if len(valid_edges) == 0:
            msg = "No interfaces found in the biomolstructure."
            raise NoInterfaceError(msg)
        pivot_edge = random.choice(valid_edges)

    residue_xyz = extract_residue_com(cifmol)

    src_chain_indices = np.where(cifmol.chains.chain_id == pivot_edge[0])[0]
    src_residue_indices = np.unique(cifmol.index_table.chains_to_residues(src_chain_indices))

    dst_chain_indices = np.where(cifmol.chains.chain_id == pivot_edge[1])[0]
    dst_residue_indices = np.unique(cifmol.index_table.chains_to_residues(dst_chain_indices))

    # (optional) keep only valid residues
    src_residue_indices = np.intersect1d(src_residue_indices, valid_indices)
    dst_residue_indices = np.intersect1d(dst_residue_indices, valid_indices)

    if src_residue_indices.size == 0 or dst_residue_indices.size == 0:
        msg = "No valid residues in one of the interface chains."
        raise NoInterfaceError(msg)

    total = np.concatenate([src_residue_indices, dst_residue_indices])

    xyz_cropped = residue_xyz[total]

    distance_map = pdist_clipped(
        xyz_cropped,
        d_thr=interface_distance_cutoff + 0.1,
        n_max=128,
    )
    distance_map[np.isnan(distance_map)] = float("inf")
    np.fill_diagonal(distance_map, float("inf"))

    Ls = len(src_residue_indices)
    interface_map = distance_map[:Ls, Ls:]

    src_min = interface_map.min(axis=1)  # (L_src,)
    dst_min = interface_map.min(axis=0)  # (L_dst,)

    src_iface = np.where(src_min <= interface_distance_cutoff)[0]
    dst_iface = np.where(dst_min <= interface_distance_cutoff)[0]

    interface_residues = np.concatenate(
        [src_residue_indices[src_iface], dst_residue_indices[dst_iface]],
    )

    if interface_residues.size == 0:
        msg = "No interface residues found in the biomolstructure."
        raise NoInterfaceError(msg)

    return interface_residues


def crop_spatial_interface_residue_simple(
    interface_bias: tuple[str, str] | None,
    cifmol: CIFMol,
    crop_length: int,
    interface_distance_cutoff: float = 12.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Find interface residues, pick one at random as pivot, then crop by spatial proximity (like crop_spatial_residue)."""
    valid_indices = get_valid_indices(cifmol)
    if valid_indices.size < crop_length:
        chain_crop = get_chain_crop_indices(cifmol, valid_indices)
        return valid_indices, chain_crop

    interface_residues = _get_interface_residues(
        interface_bias=interface_bias,
        cifmol=cifmol,
        valid_indices=valid_indices,
        interface_distance_cutoff=interface_distance_cutoff,
    )

    # pick a random pivot residue among interface residues
    pivot_residue_idx = np.random.choice(interface_residues)

    residue_xyz = extract_residue_com(cifmol)
    pivot_xyz = residue_xyz[pivot_residue_idx]

    # crop like crop_spatial_residue: nearest residues to pivot, restricted to valid_indices
    distance_map = np.linalg.norm(residue_xyz - pivot_xyz[None, :], axis=-1)  # (N_residues,)
    distance_map[np.isnan(distance_map)] = float("inf")
    distance_map[~np.isin(np.arange(residue_xyz.shape[0]), valid_indices)] = float("inf")

    crop_indices = np.argsort(distance_map)[:crop_length]
    crop_indices = np.sort(crop_indices)
    return crop_indices, get_chain_crop_indices(cifmol, crop_indices)



class Cropper:
    """Cropper class to crop biomolecular structures using different strategies."""

    def __init__(
        self,
        contiguous_prob: float = 0.2,
        spatial_prob: float = 0.4,
        interface_prob: float = 0.0,
        interface_simple_prob: float = 0.4,
        monomer_only: bool = False,
    ) -> None:
        self.contiguous_prob = contiguous_prob
        self.spatial_prob = spatial_prob
        self.interface_prob = interface_prob
        self.interface_simple_prob = interface_simple_prob
        self.monomer_only = monomer_only

    def crop(
        self,
        cifmol: CIFMol,
        crop_length: int,
        chain_bias: str | None = None,
        interface_bias: tuple[str, str] | None = None,
    ) -> ndarray:  # residue level indices
        """Return cropped residue indices and chain crop dict."""
        if len(cifmol.chains.chain_id) == 1:
            method = "contiguous"
        method = random.choices(
            population=["contiguous", "spatial", "spatial_interface", "spatial_interface_simple"],
            weights=[self.contiguous_prob, self.spatial_prob, self.interface_prob, self.interface_simple_prob],
            k=1,
        )[0]
        if method == "contiguous":
            return crop_contiguous_monomer(
                interface_bias,
                cifmol,
                crop_length,
            )
        if method == "spatial":
            return crop_spatial_residue(
                chain_bias,
                cifmol,
                crop_length,
            )
        if method == "spatial_interface":
            try:
                return crop_spatial_interface_residue(
                    interface_bias,
                    cifmol,
                    crop_length,
                )
            except NoInterfaceError:
                return crop_contiguous_monomer(
                    interface_bias,
                    cifmol,
                    crop_length,
                )
        if method == "spatial_interface_simple":
            try:
                return crop_spatial_interface_residue_simple(
                    interface_bias,
                    cifmol,
                    crop_length,
                )
            except NoInterfaceError:
                return crop_contiguous_monomer(
                    interface_bias,
                    cifmol,
                    crop_length,
                )
        else:
            msg = f"Invalid cropping method: {method}"
            raise ValueError(msg)
