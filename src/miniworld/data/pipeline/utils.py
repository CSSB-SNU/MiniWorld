from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from pathlib import Path

    from miniworld.data.mols.cifmol_attached import CIFMolAttached, CIFResidueView


def remove_terminal_oxygen(cifmol: CIFMolAttached) -> np.ndarray:
    """Remove terminal oxygen atoms from the cifmol based on molecule type."""
    _entity_tag_to_idx_mapping = {
        "A": "OXT",  # MoleculeType.ANTIBODY,
        "P": "OXT",  # MoleculeType.PROTEIN,
        "Q": "OXT",  # MoleculeType.DPROTEIN,
        "R": "OP3",  # MoleculeType.RNA,
        "D": "OP3",  # MoleculeType.DNA,
        "N": "OP3",  # MoleculeType.NA,
        "L": None,  # MoleculeType.LIGAND,
        "B": None,  # MoleculeType.BRANCHED,
        "X": None,  # unknown molecule type treated as ligand
    }
    atom_mask = []

    for chain_id in cifmol.chains.chain_id:
        temp_cifmol = cifmol.chains[cifmol.chains.chain_id == chain_id].extract()
        entity_tag = cifmol.chains[cifmol.chains.chain_id == chain_id].seq_id[0].value
        entity_tag = str(entity_tag[0])
        terminal_oxygen_name = _entity_tag_to_idx_mapping[entity_tag]
        if terminal_oxygen_name is None:
            # not applicable
            atom_mask.append(np.array([True] * len(temp_cifmol.atoms)))
            continue
        atom_mask.append(temp_cifmol.atoms.id != terminal_oxygen_name)
    return np.concatenate(atom_mask, axis=0)


def remove_water(cifmol: CIFMolAttached) -> CIFMolAttached:
    """Remove water molecules from the cifmol."""
    residue_mask = cifmol.residues.chem_comp_id != "HOH"
    return cifmol.residues[residue_mask].extract()


def parse_signalp(signalp_path: Path) -> tuple[int, int] | None:
    """Parse the signalp output file and extract the sequence ids."""
    if not signalp_path.exists():
        return None
    with signalp_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    result = lines[1].split("\t")
    return int(result[3]) - 1, int(result[4]) - 1


def load_signalp(
    signalp_dir: Path | None,
) -> dict[str, tuple[int, int]]:
    """Load SignalP data from a directory containing GFF3 files."""
    signalp_data = {}
    if signalp_dir is not None:
        if not signalp_dir.exists():
            msg = f"SignalP directory {signalp_dir} does not exist."
            raise FileNotFoundError(msg)
        for signalp_file in signalp_dir.glob("*.gff3"):
            seqid = signalp_file.stem
            signalp_data[seqid] = parse_signalp(signalp_file)
    return signalp_data


def remove_signalp(
    cifmol: CIFMolAttached,
    signalp_dict: dict[str, tuple[int, int]],
) -> CIFMolAttached:
    """Filter instruction to remove signal peptides from CIFMol."""
    if cifmol is None:
        return None
    valid_residue_indices = []
    cursor = 0
    for ii in range(len(cifmol.chains)):
        chain_id = cifmol.chains.chain_id[ii].value
        seq_id = str(cifmol.chains.seq_id[ii].value)
        chain_cifmol = cifmol.chains[cifmol.chains.chain_id == chain_id].extract()
        if seq_id not in signalp_dict:
            valid_residue_indices.extend(
                list(range(cursor, cursor + len(chain_cifmol.residues))),
            )
            cursor += len(chain_cifmol.residues)
            continue
        _, signalp_end = signalp_dict[seq_id]
        valid_residue_indices.extend(
            list(range(cursor + signalp_end + 1, cursor + len(chain_cifmol.residues))),
        )
        cursor += len(chain_cifmol.residues)
    return cifmol.residues[valid_residue_indices].extract()


class NoInterfaceError(ValueError):
    """Raised when no interface is found in the biomolstructure."""


def find_interface_residues(
    mol: CIFMolAttached,
    chain_id1: str,
    chain_id2: str,
    cutoff: float = 6.0,
) -> CIFResidueView:
    """Find interface residues between two chains in a CIFMol."""
    src_chain = mol.chains.select(chain_id=chain_id1)
    dst_chain = mol.chains.select(chain_id=chain_id2)

    selected_atoms = (src_chain + dst_chain).atoms
    valid_indices = np.where(np.all(np.isfinite(selected_atoms.xyz), axis=-1))[0]
    pairs = KDTree(selected_atoms.xyz[valid_indices]).query_pairs(
        cutoff,
        output_type="ndarray",
    )
    pairs = valid_indices[pairs]

    atoms_chain_ids = selected_atoms.to_chains().chain_id
    chain_1_atom_indices = np.where(atoms_chain_ids == chain_id1)[0]
    chain_2_atom_indices = np.where(atoms_chain_ids == chain_id2)[0]
    pairs = pairs[
        (np.isin(pairs, chain_1_atom_indices).sum(axis=1) == 1)
        & (np.isin(pairs, chain_2_atom_indices).sum(axis=1) == 1)
    ]
    interface_residues = selected_atoms[pairs.flatten()].residues.sort()

    if len(interface_residues) == 0:
        msg = f"No interface residues found in structure {mol.id}."
        raise NoInterfaceError(msg)

    return interface_residues
