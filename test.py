import pickle
from pathlib import Path

import numpy as np
from biomol.cif.mol import CIFMol


def extract_residue_com(cifmol: CIFMol) -> np.ndarray:
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



def debug_cifmol(cifmol_pickled_path: Path):

    with open(cifmol_pickled_path, "rb") as f:
        cifmol: CIFMol = pickle.load(f)

    res_com = extract_residue_com(cifmol)
    
    print("CIFMol loaded successfully.")
    print(f"Number of atoms: {len(cifmol.atoms)}")
    print(f"Number of residues: {len(cifmol.residues)}")
    print(f"Number of chains: {len(cifmol.chains)}")
    
    breakpoint()

if __name__ == "__main__":
    debug_cifmol(Path("debug_cifmol.pkl"))
