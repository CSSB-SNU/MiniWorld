from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from biomol.cif import CIFMol

from miniworld.data.features.batch_edge_backprop import MSAFeatures
from miniworld.data.io import load_a3m
from miniworld.data.mols.cifmol_attached import CIFMolAttached
from miniworld.data.msa import MSA, ComplexMSA

if TYPE_CHECKING:
    from pathlib import Path

CIFMOL = CIFMol | CIFMolAttached


def load_msa(
    cifmol: CIFMOL,
    chain_id_to_crop_indices: dict[str, np.ndarray],
    env_path: Path,
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
            cropped_seq = cifmol.chains[
                cifmol.chains.chain_id == chain_id
            ].residues.one_letter_code_can.value
            cropped_seq = "".join(cropped_seq)
            msa = MSA.from_query(
                query=cropped_seq,
                seq_id=seq_id,
                a3m_type="protein",
            )
            msa_list.append(msa)
            continue
        msa = MSA.cropped(msa, crop_indices)
        msa_list.append(msa)

    return ComplexMSA(
        MSAs=msa_list,
    )

def sample_msa(
    msa: ComplexMSA, n_samples: int, max_msa_depth: int,
    ) -> MSAFeatures:
    """Sample and process MSA for model input."""
    msa_profile = msa.profile
    msa_deletion_mean = msa.deletion_mean
    msa_sequence_sampled = []
    msa_has_deletion_sampled = []
    msa_deletion_value_sampled = []
    for _ in range(n_samples):
        _, sampled_sequence, sampled_has_deletion, sampled_deletion_value = (
            msa.sample(max_msa_depth)
        )
        msa_sequence_sampled.append(sampled_sequence)  # (N_seq, L)
        msa_has_deletion_sampled.append(sampled_has_deletion)  # (N_seq, L)
        msa_deletion_value_sampled.append(sampled_deletion_value)  # (N_seq, L)
    msa_sequence_sampled = np.stack(
        msa_sequence_sampled,
        axis=0,
    )  # (N_sample, N_seq, L)
    msa_has_deletion_sampled = np.stack(msa_has_deletion_sampled, axis=0)
    msa_deletion_value_sampled = np.stack(
        msa_deletion_value_sampled,
        axis=0,
    ).astype(np.float32)

    return MSAFeatures.from_sample(
        aligned_sequences=torch.from_numpy(msa_sequence_sampled),
        has_deletion=torch.from_numpy(msa_has_deletion_sampled).int(),
        deletion_value=torch.from_numpy(msa_deletion_value_sampled),
        profile=torch.from_numpy(msa_profile),
        deletion_mean=torch.from_numpy(msa_deletion_mean),
    )
