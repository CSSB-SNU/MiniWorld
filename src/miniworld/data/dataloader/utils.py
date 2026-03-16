from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from biomol.cif import CIFMol

from miniworld.data.features.batch_edge_backprop import MSAFeatures
from miniworld.data.io import load_a3m
from miniworld.data.mapping import ResidueMapping
from miniworld.data.mols.cifmol_attached import CIFMolAttached
from miniworld.data.msa import MSA, ComplexMSA

if TYPE_CHECKING:
    from pathlib import Path

CIFMOL = CIFMol | CIFMolAttached


def _get_query_sequence(
    cifmol: CIFMOL,
    chain_id: str,
) -> np.ndarray:
    rm = ResidueMapping()
    seq_id = cifmol.chains[cifmol.chains.chain_id == chain_id].seq_id[0].value
    cropped_seq_can = cifmol.chains[
        cifmol.chains.chain_id == chain_id
    ].residues.one_letter_code_can.value
    molecule_type = seq_id[0]
    if molecule_type in ["A", "P", "Q"]:
        return rm.protein.map(cropped_seq_can)
    if molecule_type == "D":
        return rm.dna.map(cropped_seq_can)
    if molecule_type == "R":
        return rm.rna.map(cropped_seq_can)
    if molecule_type == "N":
        cropped_seq = cifmol.chains[
            cifmol.chains.chain_id == chain_id
        ].residues.one_letter_code.value
        return rm.na.map(cropped_seq)
    return rm.ligand.map(cropped_seq_can)


def load_msa(
    cifmol: CIFMOL,
    chain_id_to_crop_indices: dict[str, np.ndarray],
    env_path: Path,
    missing_policy: Literal["gap", "query"] = "gap",
) -> ComplexMSA:
    """Load and crop MSAs for each chain in the cropped cifmol."""
    msa_list: list[MSA] = []
    total_length = 0
    for chain_id, crop_indices in chain_id_to_crop_indices.items():
        if len(crop_indices) == 0:
            continue
        total_length += len(crop_indices)
        seq_id = cifmol.chains[cifmol.chains.chain_id == chain_id].seq_id[0].value
        seq_id = str(seq_id)
        msa = load_a3m(
            key=seq_id,
            env_path=env_path,
        )
        if msa is None:
            # already cropped
            query_seq = _get_query_sequence(cifmol, chain_id)
            msa = MSA.from_query(
                query=query_seq,
                seq_id=seq_id,
            )
            msa_list.append(msa)
            continue
        msa = MSA.cropped(msa, crop_indices)
        msa_list.append(msa)

    return ComplexMSA(
        MSAs=msa_list,
        missing_policy=missing_policy,
    )


def sample_msa(
    msa: ComplexMSA,
    n_samples: int,
    max_msa_depth: int,
) -> MSAFeatures:
    """Sample and process MSA for model input."""
    msa_profile = msa.profile
    msa_deletion_mean = msa.deletion_mean
    msa_sequence_sampled = []
    msa_has_deletion_sampled = []
    msa_deletion_value_sampled = []
    for _ in range(n_samples):
        _, sampled_sequence, sampled_has_deletion, sampled_deletion_value = msa.sample(
            max_msa_depth,
        )
        msa_sequence_sampled.append(sampled_sequence)  # (N_seq, L)
        msa_has_deletion_sampled.append(sampled_has_deletion)  # (N_seq, L)
        msa_deletion_value_sampled.append(sampled_deletion_value)  # (N_seq, L)
    msa_sequence_sampled = np.stack(
        msa_sequence_sampled,
        axis=0,
    )  # (N_sample, N_seq, L)
    msa_mask = np.ones_like(msa_sequence_sampled, dtype=bool)  # (N_sample, N_seq, L)
    msa_has_deletion_sampled = np.stack(msa_has_deletion_sampled, axis=0)
    msa_deletion_value_sampled = np.stack(
        msa_deletion_value_sampled,
        axis=0,
    ).astype(np.float32)

    return MSAFeatures.from_sample(
        aligned_sequences=torch.from_numpy(msa_sequence_sampled),
        msa_mask=torch.from_numpy(msa_mask),
        has_deletion=torch.from_numpy(msa_has_deletion_sampled).int(),
        deletion_value=torch.from_numpy(msa_deletion_value_sampled),
        profile=torch.from_numpy(msa_profile),
        deletion_mean=torch.from_numpy(msa_deletion_mean),
    )


def remove_terminal_oxygen(cifmol: CIFMolAttached) -> CIFMolAttached:
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
    atom_mask = np.concatenate(atom_mask, axis=0)

    return cifmol.atoms[atom_mask].extract()


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
