from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np
import torch
from biomol.cif import CIFMol
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
        sequences: dict[str, ndarray],
        headers: dict[str, ndarray],
        species_to_idx: dict[object, list[int]] | None = None,
    ) -> None:
        self.seq_id = seq_id
        self.query_sequence: ndarray = sequences["query_sequence"]  # (L, )
        self.aligned_sequences: ndarray = sequences["aligned_sequences"]  # (N_seqs, L)
        self.deletions: ndarray = sequences["deletions"]  # (N_seqs, L)
        self.deletion_mean: ndarray = sequences["deletion_mean"]  # (L, )
        self.profile: ndarray = sequences["profile"]  # (L, 21) for now, protein only
        self.species: ndarray = headers["species"]  # (N_seqs, )
        # ``species_to_idx`` groups sequence indices by species. Reused by
        # ``_pairing_MSAs`` for every chain and rebuilt on every ``cropped``
        # (which changes residue axis, not species axis), so cropped instances
        # pass their parent's dict through instead of rebuilding. The build
        # itself uses a numpy sort + boundary scan: the equivalent CPython
        # ``dict.setdefault + list.append`` loop was fast on synthetic data
        # (~3 ms for 12k rows) but blew up to 18+ s on some AF3 distillation
        # MSAs with |S121 species arrays — presumably GC/malloc thrash from
        # the 10k+ Python bytes objects the ``.tolist()`` allocates. numpy
        # sort at C speed sidesteps that entirely and is ~5 ms worst case.
        if species_to_idx is not None:
            self.species_to_idx = species_to_idx
        else:
            species_arr = self.species
            n = species_arr.shape[0]
            if n == 0:
                self.species_to_idx = {}
            else:
                order = np.argsort(species_arr, kind="stable")
                sorted_sp = species_arr[order]
                boundaries = np.concatenate((
                    np.array([0], dtype=np.int64),
                    np.nonzero(sorted_sp[1:] != sorted_sp[:-1])[0] + 1,
                    np.array([n], dtype=np.int64),
                ))
                uniques = sorted_sp[boundaries[:-1]].tolist()
                self.species_to_idx = {
                    uniques[i]: order[boundaries[i]:boundaries[i + 1]]
                    for i in range(len(uniques))
                }

        self.num_seqs = self.aligned_sequences.shape[0]
        self.length = self.aligned_sequences.shape[1]
        self.shape = (self.num_seqs, self.length)

    @classmethod
    def from_query(
        cls,
        query_sequence: np.ndarray,
        seq_id: str | None = None,
    ) -> MSA:
        """Create an MSA from a single sequence if there is no aligned sequences."""
        if len(query_sequence) == 0:
            msg = "query_sequence must be a non-empty string or list or ndarray"
            raise ValueError(msg)
        rm = ResidueMapping()
        max_idx = rm.MAX_INDEX

        aligned_sequences = query_sequence[np.newaxis, :]  # (1, L)
        deletions = np.zeros_like(aligned_sequences, dtype=np.uint8)  # (1, L)
        deletion_mean = np.zeros((len(query_sequence),), dtype=np.float32)  # (L,)
        profile = np.eye(max_idx + 1, dtype=np.int32)[
            aligned_sequences
        ]  # for now, protein only
        profile = np.mean(profile, axis=0).astype(np.float32)

        sequences = {
            "query_sequence": query_sequence,
            "aligned_sequences": aligned_sequences,
            "deletions": deletions,
            "deletion_mean": deletion_mean,
            "profile": profile,
        }

        headers = {
            "species": np.array(["query"], dtype=object),
        }

        return cls(
            seq_id=seq_id,
            sequences=sequences,
            headers=headers,
        )

    def __len__(self) -> int:
        """Return number of sequences in the MSA."""
        return self.num_seqs

    def __getitem__(self, idx: int) -> ndarray:
        """Return the sequence at the given index."""
        return self.aligned_sequences[idx]

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
        query_sequence = msa.query_sequence[crop_idx]
        profile = msa.profile[crop_idx]
        deletion_mean = msa.deletion_mean[crop_idx]
        aligned_sequences = msa.aligned_sequences[:, crop_idx]
        deletions = msa.deletions[:, crop_idx]

        sequences = {
            "query_sequence": query_sequence,
            "aligned_sequences": aligned_sequences,
            "deletions": deletions,
            "deletion_mean": deletion_mean,
            "profile": profile,
        }

        headers = {"species": msa.species.copy()}

        # 3. Create a new instance — reuse the parent's species_to_idx since the
        # residue-axis crop does not touch the sequence axis, so species remain
        # aligned index-for-index.
        return cls(
            seq_id=msa.seq_id,
            sequences=sequences,
            headers=headers,
            species_to_idx=msa.species_to_idx,
        )

    def get_query_sequence(self) -> ndarray:
        """Return the query sequence."""
        return self.aligned_sequences[0]

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
        pairing_mode: Literal["mixed", "paired_only", "no_pairing"] = "mixed",
        condition_groups: list[list[int]] | None = None,
    ) -> None:
        """Pair and combine multiple MSAs.

        Pairing modes:
            - ``mixed`` (default): species-paired rows first, then per-chain
              unpaired rows fill the remaining slots. Matches the original
              training-time behaviour.
            - ``paired_only``: keep only species-paired rows; unpaired
              homologs are dropped entirely (the remaining slots up to
              ``max_msa_depth`` stay as gap/query rows).
            - ``no_pairing``: drop species pairing — row r past the query
              is a positional concat of each chain's r-th homolog
              (``row_r = chain_0[r] ++ chain_1[r] ++ ...``). Chains that
              run out of homologs hold gap/query for that row.

        Pairing is done based on:
        1. same rep_ID
        2. species_ID.

        Optional ``condition_groups``: a partition of chain indices that
        confines species pairing to within each group. Cross-group rows
        become gaps. Useful when two parts of the complex have no
        biological co-evolution (e.g. an antibody Fab and its non-cognate
        antigen), where species pairing across the boundary just yields
        spurious co-occurrences. Chains not listed in any group are
        treated as their own singleton group (no pairing partner). Ignored
        under ``no_pairing`` mode. The same partition is also used to gate
        template-derived geometry — see
        ``data/inference/complex_template.py``.
        """
        self.missing_policy = missing_policy
        self.num_of_MSAs = len(MSAs)
        self.max_MSA_depth = max_MSA_depth
        self.max_paired_depth = max_paired_depth
        self.pairing_mode = pairing_mode
        self.condition_groups = [list(g) for g in (condition_groups or [])]
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
            ii: np.where((MSA.aligned_sequences == GAP_IDX).all(axis=0))[0]
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

        # 1. Pairing — three modes:
        #    - mixed (default): species-paired rows + per-chain unpaired
        #      tail (positionally stacked).
        #    - paired_only: only species-paired rows survive; rest -> -1.
        #    - no_pairing: AF-Multimer/AF3-style block-diagonal — row 0 is
        #      the all-chain query, then each chain's homologs occupy
        #      their own row block while the other chains hold -1 (gap).
        if self.pairing_mode == "no_pairing":
            # Skip species pairing entirely: each chain's own a3m is the
            # only source. "paired" set = query row only, then the
            # extra-MSA fill stacks each chain's r-th homolog positionally
            # (row r = chain_0[r] ++ chain_1[r] ++ ... with chains that
            # ran out of homologs holding gap/query for that row).
            paired_msa_indices = {
                ii: np.asarray(query_indices[ii], dtype=int) for ii in MSAs
            }
            paired_num_of_seqs = 1
        elif self.condition_groups:
            # Per-group pairing: chains within a group go through species
            # pairing among themselves; chains in other groups hold -1
            # (gap) for those rows. Chains absent from every group become
            # their own singleton group (no pairing partner, so their
            # pairing run yields zero rows). The concatenated index
            # vectors per chain remain row-aligned, so the existing
            # downstream layout (one column-block per chain) just works.
            covered = {ii for g in self.condition_groups for ii in g}
            groups_with_singletons = [
                list(g) for g in self.condition_groups
            ] + [[ii] for ii in MSAs if ii not in covered]
            per_chain_indices: dict[int, list[int]] = {ii: [] for ii in MSAs}
            paired_num_of_seqs = 0
            for group in groups_with_singletons:
                budget = max(0, self.max_paired_depth - 1 - paired_num_of_seqs)
                if budget <= 0:
                    break
                group_msas = {ii: MSAs[ii] for ii in group if ii in MSAs}
                if len(group_msas) < 2:
                    # Singleton group: no species pairing possible.
                    continue
                grp_indices, _grp_species, grp_n = self._pairing_MSAs(
                    group_msas, max_paired_depth=budget,
                )
                if grp_n <= 0:
                    continue
                for ii in MSAs:
                    if ii in grp_indices:
                        per_chain_indices[ii].extend(grp_indices[ii].tolist())
                    else:
                        per_chain_indices[ii].extend([-1] * grp_n)
                paired_num_of_seqs += grp_n
            paired_msa_indices = {
                key: np.concatenate([
                    np.asarray(query_indices[key], dtype=int),
                    np.asarray(indices, dtype=int)
                    if indices else np.empty((0,), dtype=int),
                ])
                for key, indices in per_chain_indices.items()
            }
            paired_num_of_seqs += 1
        else:
            paired_msa_indices, _, paired_num_of_seqs = self._pairing_MSAs(MSAs)
            paired_msa_indices = {
                key: np.concatenate([query_indices[key], indices])
                for key, indices in paired_msa_indices.items()
            }
            paired_num_of_seqs += 1
        self._test_uniqueness(paired_msa_indices)

        # 3. Add extra MSAs (skipped when paired_only — keeps the model from
        # ever seeing an unpaired row).
        if self.condition_groups and self.pairing_mode != "paired_only":
            # Group-aware unpaired tail: within a group, the chain's r-th
            # unpaired homolog gets stacked positionally at row r (same as
            # the default 'mixed' behavior). Chains in *other* groups hold
            # -1 (gap) at that row, so the cross-group co-occurrence the
            # group-aware paired block already eliminated isn't reintroduced
            # by the tail. Groups fill in order; row indices grow until the
            # max_msa_depth budget is exhausted.
            covered = {ii for g in self.condition_groups for ii in g}
            groups_with_singletons = [
                list(g) for g in self.condition_groups
            ] + [[ii] for ii in MSAs if ii not in covered]
            tail_indices: dict[int, list[int]] = {ii: [] for ii in MSAs}
            total_budget = max(0, max_msa_depth - paired_num_of_seqs)
            # Fair-share the unpaired tail across groups so no single group
            # (e.g. a heavily-templated antibody Fab) crowds out the others.
            # Per-group budget = floor(total/n) with the remainder spread
            # over the first few groups; unused budget rolls over to the
            # next group so we still fill ``max_msa_depth`` rows whenever a
            # later group has the homologs to use them.
            n_groups = len(groups_with_singletons)
            base = total_budget // n_groups if n_groups else 0
            extra = total_budget - base * n_groups
            rollover = 0
            for gi, group in enumerate(groups_with_singletons):
                budget = base + (1 if gi < extra else 0) + rollover
                if budget <= 0:
                    continue
                group_msas = {ii: MSAs[ii] for ii in group if ii in MSAs}
                if not group_msas:
                    rollover = budget
                    continue
                pool: dict[int, list[int]] = {}
                for ii in group_msas:
                    used = set(paired_msa_indices[ii].tolist())
                    full = list(range(len(MSAs[ii])))
                    pool[ii] = [r for r in full if r not in used]
                group_rows = min(budget, max(len(p) for p in pool.values()))
                for r in range(group_rows):
                    for ii in MSAs:
                        if ii in pool and r < len(pool[ii]):
                            tail_indices[ii].append(pool[ii][r])
                        else:
                            tail_indices[ii].append(-1)
                rollover = budget - group_rows
            # Pad each chain's index vector out to ``max_msa_depth`` with
            # -1 (gap). Without this, the consumer loop below
            # (``for ii in range(max_msa_depth)``) over-runs whenever the
            # group-aware paired+tail block produces fewer rows than
            # ``max_msa_depth`` — e.g. when paired-a3m sources (colab
            # merge) cap total depth below the budget. The non-grouped
            # branch already pads in the same way (see below).
            final_msa_indices = {}
            for key in MSAs:
                combined = np.concatenate([
                    paired_msa_indices[key],
                    np.asarray(tail_indices[key], dtype=int),
                ])
                if combined.shape[0] < max_msa_depth:
                    pad = np.full(max_msa_depth - combined.shape[0], -1, dtype=int)
                    combined = np.concatenate([combined, pad])
                final_msa_indices[key] = combined.astype(np.int32)
        else:
            final_msa_indices = {}
            for key, values in MSAs.items():
                msa_depth = len(values)
                full_indices = list(range(msa_depth))
                paired_indices = paired_msa_indices[key]

                if self.pairing_mode == "paired_only":
                    # No extra rows — pad with -1 only.
                    missing_indices = [-1] * max(0, max_msa_depth - paired_num_of_seqs)
                else:
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
                    seqs.append(msa.aligned_sequences[idx])
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
        """Randomly sample sequences from the complex MSA.

        ``no_pairing`` mode skips the random draw and takes the first
        ``max_msa_depth`` rows of :attr:`sequence` straight off the top —
        i.e. each chain's top-N a3m hits, lined up positionally. No
        species pairing, no shuffling.
        """
        if rng is None:
            rng = np.random.default_rng()

        if getattr(self, "pairing_mode", "mixed") == "no_pairing":
            n = min(max_msa_depth, self.sequence.shape[0])
            idx = np.arange(n, dtype=int)
            return (
                idx,
                self.sequence[idx],
                self.has_deletion[idx],
                self.deletion_value[idx],
            )

        max_msa_depth = min(max_msa_depth, self.total_depth)
        sampled = [int(ratio[ii] * max_msa_depth) for ii in range(2)]
        if sum(sampled) != max_msa_depth:
            sampled[0] += 1  # make sure the sum is equal to max_msa_depth

        # Donate leftover slots in both directions so we always fill
        # ``max_msa_depth`` rows when ``total_depth`` allows. Without the
        # second branch, an MSA whose rows are all paired (monomer, or a
        # homo-mer with shared per-chain a3m where species pairing is
        # trivially full) wastes the unpaired half of the split and only
        # ~total_depth/2 rows survive.
        to_be_sampled = (self.num_of_paired, self.num_of_unpaired)
        if to_be_sampled[0] < sampled[0]:
            sampled[1] += sampled[0] - to_be_sampled[0]
            sampled[0] = to_be_sampled[0]
        if to_be_sampled[1] < sampled[1]:
            sampled[0] += sampled[1] - to_be_sampled[1]
            sampled[1] = to_be_sampled[1]
        sampled[0] = min(sampled[0], to_be_sampled[0])
        sampled[1] = min(sampled[1], to_be_sampled[1])

        # Query lives at row 0 of ``self.sequence`` and is always prepended
        # below — sample the rest of the paired slots from [1, num_of_paired)
        # so we never draw the query row twice.
        query = np.array([0])
        n_extra_paired = max(0, sampled[0] - 1)
        if n_extra_paired > 0:
            paired_sampled = rng.choice(
                np.arange(1, self.num_of_paired),
                n_extra_paired,
                replace=False,
            )
        else:
            paired_sampled = np.empty(0, dtype=int)

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
    sample_depth: Literal["uniform", "fixed"] = "uniform",
) -> MSAFeatures:
    """Sample and process MSA for model input.

    ``sample_depth="uniform"`` (AF3-style, default) draws the per-item depth
    k ~ Uniform[1, min(n_available, max_msa_depth)] so the model sees a range
    of MSA depths. ``sample_depth="fixed"`` always requests max_msa_depth
    (legacy behavior).
    """
    if rng is None:
        rng = np.random.default_rng()

    if sample_depth == "uniform":
        if getattr(msa, "pairing_mode", "mixed") == "no_pairing":
            n_available = int(msa.sequence.shape[0])
        else:
            n_available = int(msa.total_depth)
        upper = max(1, min(n_available, max_msa_depth))
        effective_depth = int(rng.integers(1, upper + 1))
    else:
        effective_depth = max_msa_depth

    profile = msa.profile
    deletion_mean = msa.deletion_mean
    _, aligned_sequences, has_deletion, deletion_value = msa.sample(
        effective_depth,
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
