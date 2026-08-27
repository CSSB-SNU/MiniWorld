from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np
import torch
from biomol.cif import CIFMol
from biomol.core.container import FeatureContainer
from biomol.core.feature import NodeFeature
from numpy import ndarray

from miniworld.data.constants import ResidueMapping
from miniworld.data.features.features import MSAFeatures
from miniworld.data.mols import CIFMolAttached

CIFMOL = CIFMol | CIFMolAttached
GAP_IDX = ResidueMapping().protein.GAP_INDEX


class MSA:
    """Multiple Sequence Alignment (MSA) class."""

    def __init__(
        self,
        seq_id: str | None,
        msa_residue_container: FeatureContainer,
        msa_chain_container: FeatureContainer,
    ) -> None:
        self.seq_id = seq_id
        self.query_sequence: ndarray = msa_residue_container[
            "query_sequence"
        ].value  # (L, )
        self.sequences: ndarray = msa_residue_container[
            "sequences"
        ].value.T  # (N_seqs, L)
        self.deletions: ndarray = msa_residue_container[
            "deletions"
        ].value.T  # (N_seqs, L)
        self.deletion_mean: ndarray = msa_residue_container[
            "deletion_mean"
        ].value  # (L, )
        self.profile: ndarray = msa_residue_container["profile"].value  # (L, 32)
        self.species: ndarray = msa_chain_container["species"].value  # (N_seqs, )
        self.species_to_idx = {
            species: np.where(self.species == species)[0].tolist()
            for species in set(self.species)
        }

        self.num_seqs = self.sequences.shape[0]
        self.length = self.sequences.shape[1]
        self.shape = (self.num_seqs, self.length)

    @classmethod
    def from_query(
        cls,
        query: np.ndarray,
        seq_id: str | None = None,
    ) -> MSA:
        """Create an MSA from a single sequence if there is no aligned sequences."""
        if len(query) == 0:
            msg = "query must be a non-empty string or list or ndarray"
            raise ValueError(msg)
        rm = ResidueMapping()
        max_idx = rm.MAX_INDEX

        sequences = query[:, np.newaxis]  # (L, 1)
        deletions = np.zeros_like(sequences, dtype=np.uint8)  # (L, 1)
        deletion_mean = np.zeros((len(query),), dtype=np.float32)  # (L,)
        profile = np.eye(max_idx + 1, dtype=np.int32)[sequences]  # for now, protein only
        profile = np.mean(profile, axis=1).astype(np.float32)

        query_sequence = NodeFeature(value=query)
        sequences = NodeFeature(value=sequences)
        deletions = NodeFeature(value=deletions)
        deletion_mean = NodeFeature(value=deletion_mean)
        profile = NodeFeature(value=profile)
        species = NodeFeature(value=np.array(["query"], dtype=object))  # (1,)

        msa_residue_container = FeatureContainer(
            {
                "query_sequence": query_sequence,
                "sequences": sequences,
                "deletions": deletions,
                "deletion_mean": deletion_mean,
                "profile": profile,
            },
        )
        msa_chain_container = FeatureContainer(
            {
                "species": species,
            },
        )

        return cls(
            seq_id=seq_id,
            msa_residue_container=msa_residue_container,
            msa_chain_container=msa_chain_container,
        )

    def __len__(self) -> int:
        """Return number of sequences in the MSA."""
        return self.num_seqs

    def __getitem__(self, idx: int) -> ndarray:
        """Return the sequence at the given index."""
        return self.sequences[idx]

    @classmethod
    def cropped(
        cls,
        msa: MSA,
        crop_idx: np.ndarray,
    ) -> MSA:
        """Return a new MSA instance cropped along residue dimension.

        Does NOT modify the original msa.

        crop_idx: 1D index array of selected residue positions.
        """
        # 1. Crop residue-level features
        query_sequence = NodeFeature(value=msa.query_sequence[crop_idx])
        profile = NodeFeature(value=msa.profile[crop_idx])
        deletion_mean = NodeFeature(value=msa.deletion_mean[crop_idx])

        # (N_seqs, L) → crop L dimension → still (N_seqs, L')
        sequences = NodeFeature(value=msa.sequences[:, crop_idx].T)
        deletions = NodeFeature(value=msa.deletions[:, crop_idx].T)

        msa_residue_container = FeatureContainer(
            {
                "query_sequence": query_sequence,
                "sequences": sequences,
                "deletions": deletions,
                "deletion_mean": deletion_mean,
                "profile": profile,
            },
        )
        # 2. chain-level features (unchanged)
        msa_chain_container = FeatureContainer(
            {"species": NodeFeature(value=msa.species.copy())},
        )

        # 3. Create a new instance
        return cls(
            seq_id=msa.seq_id,
            msa_residue_container=msa_residue_container,
            msa_chain_container=msa_chain_container,
        )

    def get_query_sequence(self) -> ndarray:
        """Return the query sequence."""
        return self.sequences[0]

    def get_profile(self) -> ndarray:
        """Return the profile."""
        return self.profile

    def get_deletion_mean(self) -> ndarray:
        """Return the deletion mean."""
        return self.deletion_mean


class ComplexMSA:
    """Complex MSA class that combines multiple MSAs."""

    def __init__(
        self,
        MSAs: list[MSA],
        missing_policy: Literal["gap", "query"] = "gap",
        max_MSA_depth: int = 16384,
        max_paired_depth: int = 8192,  # including query
    ) -> None:
        """Pair and combine multiple MSAs.

        Pairing is done based on:
        1. same rep_ID
        2. species_ID.
        """
        self.missing_policy = missing_policy
        self.num_of_MSAs = len(MSAs)
        self.max_MSA_depth = max_MSA_depth
        self.max_paired_depth = max_paired_depth
        self._prepare_MSA(MSAs)

    def _test_uniqueness(self, input_dict: dict[int, ndarray]) -> None:
        """Test the uniqueness of the values in a dictionary for each key."""
        out = True
        for _value in input_dict.values():
            # remove -1
            value = [v for v in _value if v != -1]
            out = (len(value) == len(set(value))) and out
            if not out:
                msg = "Values in the dictionary are not unique for key."
                raise ValueError(msg)

    def _pairing_MSAs(
        self,
        MSAs: dict[int, MSA],
        max_paired_depth: int = 8191,  # 8192 - 1 (query sequence is always included)
    ) -> tuple[dict[int, np.ndarray], set, int]:
        species_to_idx_dict = {ii: MSA.species_to_idx for ii, MSA in MSAs.items()}
        # gap indices per MSA (sequence entirely gaps) : (L, N) -> column-wise all-gaps
        gap_idx_dict = {
            ii: np.where((MSA.sequences == GAP_IDX).all(axis=0))[0]
            for ii, MSA in MSAs.items()
        }
        all_species = set.union(*(set(d.keys()) for d in species_to_idx_dict.values()))
        all_species.discard("N/A")

        species_to_count = Counter(
            species for s_to_idx in species_to_idx_dict.values() for species in s_to_idx
        )
        species_to_count = dict(species_to_count)
        species_to_count.pop("N/A", None)

        sorted_species = sorted(
            species_to_count.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_species = [species for species, count in sorted_species]

        msa_indices_list: dict[int, list[int]] = {key: [] for key in MSAs}
        empty = np.array([], dtype=int)
        num_of_paired = 0
        paired_species = set()
        for species in sorted_species:
            valid_idx_dict = {}
            for ii, species_map in species_to_idx_dict.items():
                idx = np.asarray(species_map.get(species, empty), dtype=int)
                idx = np.setdiff1d(idx, gap_idx_dict[ii], assume_unique=True)
                idx = idx[idx != 0]
                valid_idx_dict[ii] = idx

            num_seqs = {ii: len(idx) for ii, idx in valid_idx_dict.items()}
            # remove 0
            temp_list = [num_seqs[ii] for ii in num_seqs if num_seqs[ii] > 0]

            if len(temp_list) == 0:
                continue
            min_num_seqs = min(temp_list)
            for key, valid_idx in valid_idx_dict.items():
                if num_seqs[key] > 0:
                    msa_indices_list[key].extend(valid_idx[:min_num_seqs].tolist())
                else:
                    msa_indices_list[key].extend([-1] * min_num_seqs)

            num_of_paired += min_num_seqs
            paired_species.add(species)
            if num_of_paired >= max_paired_depth:
                break

        msa_indices = {
            key: np.array(indices, dtype=int)
            if len(indices) > 0
            else np.empty((0,), dtype=int)
            for key, indices in msa_indices_list.items()
        }

        self._test_uniqueness(msa_indices)

        # sort by sum of indices (row)  (-1을 큰 값으로 치환해서 뒤로 밀기)
        large = 1 << 30
        mat = np.stack(
            [np.where(arr == -1, large, arr) for arr in msa_indices.values()],
            axis=0,
        )
        sum_of_indices = np.sum(mat, axis=0)
        sorted_indices = np.argsort(sum_of_indices)
        num_of_seqs = min(len(sorted_indices), max_paired_depth)

        msa_indices = {
            key: indices[sorted_indices][:num_of_seqs]
            for key, indices in msa_indices.items()
        }
        self._test_uniqueness(msa_indices)

        return msa_indices, paired_species, num_of_seqs

    def _prepare_MSA(self, _MSAs: list[MSA]) -> None:  # noqa: C901, PLR0912, PLR0915
        MSAs: dict[int, MSA] = dict(enumerate(_MSAs))
        max_msa_depth = self.max_MSA_depth

        # 0. Query sequence
        query_indices = {ii: [0] for ii in MSAs}

        # 1. Simple pairing common species for all MSAs
        paired_msa_indices, _, paired_num_of_seqs = self._pairing_MSAs(MSAs)
        paired_msa_indices = {
            key: np.concatenate([query_indices[key], indices])
            for key, indices in paired_msa_indices.items()
        }
        paired_num_of_seqs += 1
        self._test_uniqueness(paired_msa_indices)

        # 3. Add extra MSAs
        final_msa_indices = {}
        for key, values in MSAs.items():
            msa_depth = len(values)
            full_indices = list(range(msa_depth))
            paired_indices = paired_msa_indices[key]

            # add missing indices at the end
            missing_indices = set(full_indices) - set(paired_indices)
            missing_indices = sorted(missing_indices)

            # if msa_depth < max_msa_depth, add -1 to the end
            if msa_depth < max_msa_depth:
                missing_indices += [-1] * (max_msa_depth - msa_depth)
            else:
                missing_indices = missing_indices[: max_msa_depth - paired_num_of_seqs]
            missing_indices = np.array(missing_indices)
            final_msa_indices[key] = np.concatenate(
                [paired_msa_indices[key], missing_indices],
            ).astype(np.int32)
        self._test_uniqueness(final_msa_indices)

        final_sequence = []
        final_deletion = []
        final_has_deletion = []
        filtered_paired_num_of_seqs = paired_num_of_seqs

        query_sequence = None
        for ii in range(max_msa_depth):
            seqs = []
            deletion = []
            indices = []
            for key, values in MSAs.items():
                idx = final_msa_indices[key][ii]
                indices.append(idx)
                msa = values
                if idx == -1:
                    if self.missing_policy == "gap":
                        seqs.append(np.full((msa.length,), GAP_IDX))
                    elif self.missing_policy == "query":
                        seqs.append(msa.get_query_sequence())
                    else:
                        msg = f"Unsupported missing_policy: {self.missing_policy}"
                        raise ValueError(msg)
                    deletion.append(np.zeros(msa.length))
                else:
                    # (L, N_seqs) -> 열 인덱싱으로 1D(L,) 가져오기
                    seqs.append(msa.sequences[idx])
                    deletion.append(msa.deletions[idx])
            seqs = np.concatenate(seqs)
            if query_sequence is None:
                query_sequence = seqs
            # 1. if all sequences are missing, skip this sequence
            # 2. if all sequences are identical to query sequence, skip this sequence
            elif np.array_equal(
                seqs,
                query_sequence,
            ) or np.array_equal(seqs, np.full_like(seqs, GAP_IDX)):
                if ii < paired_num_of_seqs:
                    filtered_paired_num_of_seqs -= 1
                continue

            deletion = np.concatenate(deletion)
            final_sequence.append(seqs)
            has_deletion = np.array(deletion > 0, dtype=np.uint8)
            final_deletion.append(deletion)
            final_has_deletion.append(has_deletion)

        # remove all gap sequences
        final_sequence = np.array(final_sequence)
        final_has_deletion = np.array(final_has_deletion)
        final_deletion = np.array(final_deletion)

        gap_idx = np.where((final_sequence == GAP_IDX).all(axis=1))[0]
        final_sequence = np.delete(final_sequence, gap_idx, axis=0)
        final_deletion = np.delete(final_deletion, gap_idx, axis=0)
        final_has_deletion = np.delete(final_has_deletion, gap_idx, axis=0)
        final_msa_indices = {
            key: np.delete(indices, gap_idx, axis=0)
            for key, indices in final_msa_indices.items()
        }

        # concat profile, deletion_mean
        profile = np.concatenate([msa.profile for msa in MSAs.values()], axis=0)
        deletion_mean = np.concatenate(
            [msa.deletion_mean for msa in MSAs.values()],
            axis=0,
        )

        self.msa_indices = final_msa_indices

        self.sequence = final_sequence
        self.has_deletion = final_has_deletion
        self.deletion_value = 2 * np.arctan(final_deletion / 3) / np.pi
        self.profile = profile
        self.deletion_mean = deletion_mean

        self.num_of_paired = filtered_paired_num_of_seqs
        self.num_of_unpaired = self.sequence.shape[0] - filtered_paired_num_of_seqs
        self.total_depth = self.num_of_paired + self.num_of_unpaired

    def sample(
        self,
        max_msa_depth: int = 256,
        ratio: tuple[float, float] = (0.5, 0.5),
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly sample sequences from the complex MSA."""
        if rng is None:
            rng = np.random.default_rng()

        max_msa_depth = min(max_msa_depth, self.total_depth)
        sampled = [int(ratio[ii] * max_msa_depth) for ii in range(2)]
        if sum(sampled) != max_msa_depth:
            sampled[0] += 1  # make sure the sum is equal to max_msa_depth

        to_be_sampled = (self.num_of_paired, self.num_of_unpaired)
        if to_be_sampled[0] < sampled[0]:
            sampled[1] += sampled[0] - to_be_sampled[0]
            sampled[0] = to_be_sampled[0]
        sampled[1] = min(sampled[1], to_be_sampled[1])

        query = np.array([0])
        paired_sampled = rng.choice(
            self.num_of_paired,
            sampled[0] - 1,
            replace=False,
        )  # -1 for query

        if sampled[1] > 0:
            unpaired_sampled = (
                rng.choice(self.num_of_unpaired, sampled[1], replace=False)
                + self.num_of_paired
            )

            sampled_indices = np.concatenate([query, paired_sampled, unpaired_sampled])
        else:
            sampled_indices = np.concatenate([query, paired_sampled])
        sampled_indices = np.sort(sampled_indices)

        sampled_sequence = self.sequence[sampled_indices]
        sampled_has_deletion = self.has_deletion[sampled_indices]
        sampled_deletion_value = self.deletion_value[sampled_indices]

        return (
            sampled_indices,
            sampled_sequence,
            sampled_has_deletion,
            sampled_deletion_value,
        )


def sample_msa(
    msa: ComplexMSA,
    max_msa_depth: int,
    rng: np.random.Generator | None = None,
) -> MSAFeatures:
    """Sample and process MSA for model input."""
    if rng is None:
        rng = np.random.default_rng()
    profile = msa.profile
    deletion_mean = msa.deletion_mean
    _, aligned_sequences, has_deletion, deletion_value = msa.sample(
        max_msa_depth,
        rng=rng,
    )
    n_seq, _ = aligned_sequences.shape
    mask = np.ones((n_seq), dtype=np.float32)

    return MSAFeatures.from_sample(
        aligned_sequences=torch.from_numpy(aligned_sequences),
        mask=torch.from_numpy(mask),
        has_deletion=torch.from_numpy(has_deletion).int(),
        deletion_value=torch.from_numpy(deletion_value),
        profile=torch.from_numpy(profile),
        deletion_mean=torch.from_numpy(deletion_mean),
    )
