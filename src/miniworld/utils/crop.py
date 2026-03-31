import numpy as np
from jaxtyping import Float, Int
from team_gm import typecheck

from miniworld.data.mols import CIFMolAttached


@typecheck
def crop_spatial_segment_token(
    mol: CIFMolAttached,
    center_xyz: Float[np.ndarray, "3"],
    tokens_to_res: Int[np.ndarray, " R"],
    segment_size: int,
    max_tokens: int,
    max_atoms: int | None = None,
) -> Int[np.ndarray, " L"]:
    """Select residue indices by spatial proximity with contiguous segment expansion.

    Residues are visited in order of distance to `center_xyz`.  For each visited
    residue a contiguous window of `segment_size` residues centred on it is taken
    from its chain, giving crops that are both spatially compact and sequentially
    coherent within each chain.

    Parameters
    ----------
    mol : biomol.CIFMol
        Input structure.
    center_xyz : ndarray, shape (3,)
        3-D coordinate of the crop centre.
    tokens_to_res : ndarray, shape (R,)
        Maps token indices to residue indices. Used to track the token count in the crop.
    segment_size : int
        Contiguous residues to expand per chain visit.
    max_tokens : int
        Maximum number of tokens in the crop.
    max_atoms : int | None
        Optional atom budget. The crop stops before exceeding this limit
        even if `max_tokens` has not been reached.

    Returns
    -------
    ndarray, shape (L,)
        Sorted residue indices selected for cropping.

    Notes
    -----
    When ``segment_size=1`` the result is equivalent to :func:`crop_spatial`.

    References
    ----------
    Boltz-1 unified cropping.
    https://doi.org/10.1101/2024.11.19.624167

    """
    distance_to_center = np.linalg.norm(mol.atoms.xyz.value - center_xyz, axis=-1)
    distance_to_center[~np.isfinite(distance_to_center)] = np.inf

    atom_to_res = mol.index_table.atom_to_res
    order = np.argsort(atom_to_res)
    atom_to_res_sorted = atom_to_res[order]
    dist_sorted = distance_to_center[order]

    _, starts = np.unique(atom_to_res_sorted, return_index=True)
    min_distances = np.minimum.reduceat(dist_sorted, starts)
    atoms_per_res = np.diff(np.append(starts, len(atom_to_res_sorted)))

    tokens_to_res_sorted = tokens_to_res[np.argsort(tokens_to_res)]
    _, starts = np.unique(tokens_to_res_sorted, return_index=True)
    tokens_per_res = np.diff(np.append(starts, len(tokens_to_res_sorted)))

    selected: set[int] = set()
    total_atoms = 0
    total_tokens = 0
    cif_idx = mol.residues.cif_idx.value.astype(int)

    for res_idx in np.argsort(min_distances):
        if res_idx in selected:
            continue
        chain = mol.residues[res_idx].chains
        res_indices = set(chain.residues.indices)

        if cif_idx[max(res_indices)] - cif_idx[min(res_indices)] + 1 <= segment_size:
            segment = res_indices
        else:
            min_idx = max_idx = res_idx
            segment = {int(res_idx)}
            while cif_idx[max(segment)] - cif_idx[min(segment)] + 1 < segment_size:
                min_idx -= 1
                max_idx += 1
                _segment = set(range(min_idx, max_idx + 1)) & res_indices
                if cif_idx[max(_segment)] - cif_idx[min(_segment)] + 1 > segment_size:
                    break
                if not _segment - segment:
                    break
                segment = _segment

        new = segment - selected
        if not new:
            continue

        new_atoms = int(atoms_per_res[list(new)].sum())
        if max_atoms is not None and total_atoms + new_atoms > max_atoms:
            break
        new_tokens = int(tokens_per_res[list(new)].sum())
        if total_tokens + new_tokens > max_tokens:
            break

        selected.update(new)
        total_atoms += new_atoms
        total_tokens += new_tokens

    return np.array(sorted(selected))
