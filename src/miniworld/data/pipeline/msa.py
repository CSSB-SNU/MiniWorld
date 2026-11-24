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
        # (N_seqs, L); ``cropped`` leaves it None and gathers it on first use (``deletion_rows`` reads only the rows that are needed)
        self._deletions: ndarray | None = sequences["deletions"]
        self._del_parent: ndarray | None = None
        self._del_crop: ndarray | None = None
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
        # built on first use: only species pairing reads it (never under ``no_pairing``, the training config), and the argsort over
        # the 12k+ species strings of every a3m entry cost ~8 ms per item
        self._species_to_idx = species_to_idx
        self._species_parent = None

        self.num_seqs = self.aligned_sequences.shape[0]
        self.length = self.aligned_sequences.shape[1]
        self.shape = (self.num_seqs, self.length)

    @property
    def species_to_idx(self):
        if self._species_to_idx is None:
            if self._species_parent is not None:
                self._species_to_idx = self._species_parent.species_to_idx
            else:
                self._species_to_idx = self._build_species_to_idx()
        return self._species_to_idx

    def _build_species_to_idx(self):
        species_arr = self.species
        n = species_arr.shape[0]
        if n == 0:
            return {}
        order = np.argsort(species_arr, kind="stable")
        sorted_sp = species_arr[order]
        boundaries = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.nonzero(sorted_sp[1:] != sorted_sp[:-1])[0] + 1,
                np.array([n], dtype=np.int64),
            )
        )
        uniques = sorted_sp[boundaries[:-1]].tolist()
        return {uniques[i]: order[boundaries[i] : boundaries[i + 1]] for i in range(len(uniques))}

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
        # deletions (the largest array of an a3m entry) are cropped lazily: see ``deletion_rows``
        if msa._deletions is None:
            del_parent, del_crop = msa._del_parent, msa._del_crop[crop_idx]
        else:
            del_parent, del_crop = msa._deletions, crop_idx

        sequences = {
            "query_sequence": query_sequence,
            "aligned_sequences": aligned_sequences,
            "deletions": None,
            "deletion_mean": deletion_mean,
            "profile": profile,
        }

        headers = {"species": msa.species.copy()}

        # 3. Create a new instance — reuse the parent's species_to_idx since the
        # residue-axis crop does not touch the sequence axis, so species remain
        # aligned index-for-index.
        new = cls(
            seq_id=msa.seq_id,
            sequences=sequences,
            headers=headers,
            species_to_idx=msa._species_to_idx,  # may still be None: the cropped copy then resolves it through its parent on first use
        )
        new._species_parent = msa
        new._del_parent, new._del_crop = del_parent, del_crop
        return new

    @property
    def deletions(self) -> ndarray:
        if self._deletions is None:
            self._deletions = self._del_parent[:, self._del_crop]
        return self._deletions

    def deletion_rows(self, rows: ndarray, lo: int, hi: int) -> ndarray:
        """``self.deletions[rows, lo:hi]`` without materialising the cropped alignment."""
        if self._deletions is not None:
            return self._deletions[rows, lo:hi]
        return self._del_parent[np.asarray(rows)[:, None], self._del_crop[lo:hi][None, :]]

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
            value = np.asarray(_value)
            value = value[value != -1]
            out = (np.unique(value).size == value.size) and out
            if not out:
                msg = "Values in the dictionary are not unique for key."
                raise ValueError(msg)

    def _pairing_MSAs(
        self,
        MSAs: dict[int, MSA],
        max_paired_depth: int = 8191,  # 8192 - 1 (query sequence is always included)
    ) -> tuple[dict[int, np.ndarray], set, int]:
        species_to_idx_dict = {ii: MSA.species_to_idx for ii, MSA in MSAs.items()}
        # Per-chain removal mask over the sequence-index value space, precomputed
        # once so the per-species loop below filters by O(1) boolean indexing
        # instead of a per-(species, chain) np.setdiff1d + `idx != 0`.
        # Behavior-identical to the original value-based removal: it drops sequence
        # indices whose value appears in the all-gap-COLUMN set (axis=0, unchanged)
        # plus the query row 0. Only values < n_seqs can match a sequence index,
        # so the mask covers [0, n_seqs) and idx (< n_seqs) never overflows it.
        remove_idx: dict[int, np.ndarray] = {}
        for ii, MSA in MSAs.items():
            n = MSA.aligned_sequences.shape[0]
            r = np.zeros(n, dtype=bool)
            if n:
                gap_cols = np.where(
                    (MSA.aligned_sequences == GAP_IDX).all(axis=0),
                )[0]
                r[gap_cols[gap_cols < n]] = True
                r[0] = True  # query row 0 (was dropped via idx[idx != 0])
            remove_idx[ii] = r
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
                r = remove_idx[ii]
                if idx.size and r.shape[0]:
                    idx = idx[~r[idx]]
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
            groups_with_singletons = [list(g) for g in self.condition_groups] + [
                [ii] for ii in MSAs if ii not in covered
            ]
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
                    group_msas,
                    max_paired_depth=budget,
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
                key: np.concatenate(
                    [
                        np.asarray(query_indices[key], dtype=int),
                        np.asarray(indices, dtype=int)
                        if indices
                        else np.empty((0,), dtype=int),
                    ]
                )
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
            groups_with_singletons = [list(g) for g in self.condition_groups] + [
                [ii] for ii in MSAs if ii not in covered
            ]
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
                combined = np.concatenate(
                    [
                        paired_msa_indices[key],
                        np.asarray(tail_indices[key], dtype=int),
                    ]
                )
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
                    # rows not already used by the pairing, ascending (a boolean mask instead of two python sets over up to 16k rows)
                    used = np.asarray(paired_indices)
                    used = used[(used >= 0) & (used < msa_depth)]
                    not_used = np.ones(msa_depth, dtype=bool)
                    not_used[used] = False
                    missing_indices = np.flatnonzero(not_used)

                    # if msa_depth < max_msa_depth, add -1 to the end
                    if msa_depth < max_msa_depth:
                        missing_indices = np.concatenate(
                            [
                                missing_indices,
                                np.full(max_msa_depth - msa_depth, -1, dtype=missing_indices.dtype),
                            ],
                        )
                    else:
                        missing_indices = missing_indices[
                            : max_msa_depth - paired_num_of_seqs
                        ]
                missing_indices = np.array(missing_indices)
                final_msa_indices[key] = np.concatenate(
                    [paired_msa_indices[key], missing_indices],
                ).astype(np.int32)
        self._test_uniqueness(final_msa_indices)

        # --- Vectorized MSA assembly (replaces the per-row Python loop) --------
        # Gather each chain's selected rows in one shot, concatenate chains on the
        # column axis, then drop rows that duplicate the query (row 0) or are
        # all-gap via boolean masks instead of per-row np.array_equal. Row 0
        # always defines the query and is kept. Behavior-identical to the loop
        # (verified against it under MINIWORLD_MSA_VERIFY).
        if self.missing_policy not in ("gap", "query"):
            msg = f"Unsupported missing_policy: {self.missing_policy}"
            raise ValueError(msg)

        seq_blocks: list[np.ndarray] = []
        del_src: list[tuple] = []  # per chain: (MSA, source rows, missing mask or None): the deletion blocks are built only for the drawn rows
        for _key, msa in MSAs.items():
            # Only the first max_msa_depth rows are consumed (the original loop
            # ran `for ii in range(max_msa_depth)`); final_msa_indices[key] can be
            # longer (mixed mode) or exactly max_msa_depth (padded).
            fmi = final_msa_indices[_key][:max_msa_depth]
            missing = fmi == -1
            safe = np.where(missing, 0, fmi)
            seq_block = msa.aligned_sequences[safe]  # advanced-index -> copy
            if missing.any():
                if self.missing_policy == "gap":
                    seq_block[missing] = GAP_IDX
                else:
                    seq_block[missing] = msa.get_query_sequence().astype(
                        seq_block.dtype,
                    )
            seq_blocks.append(seq_block)
            del_src.append((msa, safe, missing if missing.any() else None))

        # Column span of each chain inside the stacked alignment. Row filtering
        # below never touches columns, so these stay valid for self.sequence and
        # are what per-chain subsampling needs (see sample(randomize=True)).
        spans, _c = {}, 0
        for _key, _blk in zip(MSAs, seq_blocks, strict=True):
            spans[_key] = (_c, _c + _blk.shape[1])
            _c += _blk.shape[1]
        self.chain_col_spans = spans

        full_seq = np.concatenate(seq_blocks, axis=1)

        query_row = full_seq[0]
        skip = (full_seq == query_row).all(axis=1) | (full_seq == GAP_IDX).all(axis=1)
        skip[0] = False  # row 0 defines the query and is always kept
        filtered_paired_num_of_seqs = paired_num_of_seqs - int(
            skip[:paired_num_of_seqs].sum(),
        )
        keep = ~skip
        final_sequence = full_seq[keep]
        sel = np.flatnonzero(keep)  # rows of the stacked alignment that survive; deletion values are read from the sources at these rows

        gap_idx = np.where((final_sequence == GAP_IDX).all(axis=1))[0]
        if gap_idx.size:  # np.delete copies the whole array even when nothing is deleted (the all-gap rows were already dropped by ``skip``)
            final_sequence = np.delete(final_sequence, gap_idx, axis=0)
            sel = np.delete(sel, gap_idx)
        # Keep ``msa_indices`` ROW-ALIGNED with ``sequence``: apply the same
        # ``[:max_msa_depth]`` slice the blocks used, the same ``keep`` filter, then the
        # same ``gap_idx`` deletion (whose indices are relative to the kept rows).
        # Before this, only ``gap_idx`` was applied -- to the UNFILTERED indices -- so
        # every row past the first ``skip``ped one (a homolog identical to the query is
        # enough) pointed one or more rows too far, and per-chain subsampling either
        # read the wrong homolog or ran off the end (IndexError on an 8679-row stack).
        final_msa_indices = {
            key: (np.delete(indices[:max_msa_depth][keep], gap_idx, axis=0) if gap_idx.size else indices[:max_msa_depth][keep])
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
        # the (N x L) deletion count, has_deletion and arctan value matrices are NOT built for the whole alignment (up to 16384 x L):
        # ``_deletion_rows`` reads the counts of the rows that are drawn from the per-chain sources, and has_deletion / deletion_value
        # are elementwise functions of them (``deletion_value`` and ``has_deletion`` still give the full matrices on request)
        self._del_src = [(m, s[sel], None if mk is None else mk[sel]) for m, s, mk in del_src]
        self._dv_full = None
        self._hd_full = None
        self.profile = profile
        self.deletion_mean = deletion_mean

        self.num_of_paired = filtered_paired_num_of_seqs
        self.num_of_unpaired = self.sequence.shape[0] - filtered_paired_num_of_seqs
        self.total_depth = self.num_of_paired + self.num_of_unpaired

    def _deletion_rows(self, rows, lo=None, hi=None):
        """``final_deletion[rows, lo:hi]`` (uint8): the deletion counts of the given rows (column span lo:hi, default all)."""
        rows = np.atleast_1d(np.asarray(rows))
        total = self.sequence.shape[1]
        lo = 0 if lo is None else lo
        hi = total if hi is None else hi
        dt = np.result_type(*[(m._deletions if m._deletions is not None else m._del_parent).dtype for m, _, _ in self._del_src])  # as np.concatenate promoted before
        out = np.zeros((rows.shape[0], hi - lo), dtype=dt)
        for (msa, safe, missing), (clo, chi) in zip(self._del_src, self.chain_col_spans.values(), strict=True):
            a, b = max(lo, clo), min(hi, chi)
            if a >= b:
                continue
            blk = msa.deletion_rows(safe[rows], a - clo, b - clo)
            if missing is not None:
                blk = blk.copy() if not blk.flags.writeable else blk
                blk[missing[rows]] = 0
            out[:, a - lo : b - lo] = blk
        return out

    def _has_deletion_rows(self, rows, lo=None, hi=None):
        return (self._deletion_rows(rows, lo, hi) > 0).astype(np.uint8)

    @property
    def has_deletion(self):
        if self._hd_full is None:
            self._hd_full = self._has_deletion_rows(np.arange(self.sequence.shape[0]))
        return self._hd_full

    @staticmethod
    def _dv_of(deletion):
        return 2 * np.arctan(deletion / 3) / np.pi

    @property
    def deletion_value(self):
        """Deletion value of every row (computed on first use; the drawn rows are transformed directly by ``_dv_of``)."""
        if self._dv_full is None:
            self._dv_full = self._dv_of(self._deletion_rows(np.arange(self.sequence.shape[0])))
        return self._dv_full

    def _subsample_no_pairing(
        self,
        depth: int,
        rng: np.random.Generator,
        compact: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pick each chain's homologs independently, then re-stack positionally.

        Sampling rows of the stacked alignment uniformly is wrong here, and badly
        so. The stack is positional (row r = every chain's r-th homolog) and is as
        tall as the DEEPEST chain, with shallower chains holding query/gap fill
        past their own depth. A uniform draw therefore (a) keeps only
        depth/total of a shallow chain's real homologs -- on 7xld, chain 0 went
        from 2047 to ~460 of its 3665, and on 8tuz a chain with only 89 homologs
        in the entire pool kept 12-28 of them -- and (b) throws away the close
        homologs, because a3m rows are ordered best-first and a uniform sample is
        overwhelmingly remote hits (7xld chain 0: rows above 70% identity went
        4 -> 0, chain 1 mean identity 0.507 -> 0.36). Both chains' alignments
        collapse and the trunk puts the interface in the wrong place.

        AF3 shuffles uniformly too, but over a set where every row is a real
        homolog and each chain was cropped best-first to its own budget
        (msa_crop_size, features.py), so the operation is not the same one.

        Under ``no_pairing`` the row-to-row correspondence between chains is
        arbitrary by construction, so each chain can choose its own homologs
        independently. Every chain therefore keeps its full available depth while
        WHICH homologs it contributes still varies with the rng.
        """
        length = self.sequence.shape[1]
        fill = (
            self.sequence[0]
            if self.missing_policy == "query"
            else np.full(length, GAP_IDX, dtype=self.sequence.dtype)
        )
        out_seq = np.repeat(fill[None], depth, axis=0)
        # compact: int8 / float32 outputs (see sample_msa); the float32 rounding of the float64 value is the one the model applies itself
        out_hd = np.zeros((depth, length), dtype=np.int8 if compact else np.uint8)
        out_dv = np.zeros((depth, length), dtype=np.float32 if compact else np.float64)
        # Row 0 is the query and is never sampled over.
        out_seq[0] = self.sequence[0]
        row0 = self._deletion_rows([0])[0]
        out_hd[0] = (row0 > 0)
        out_dv[0] = self._dv_of(row0)

        for key, (lo, hi) in self.chain_col_spans.items():
            rows = np.flatnonzero(np.asarray(self.msa_indices[key]) != -1)
            rows = rows[rows != 0]
            if rows.size == 0:
                continue
            if rows.size > depth - 1:
                rows = rng.choice(rows, depth - 1, replace=False)
            rows = np.sort(rows)  # keep better hits first, as the trunk saw
            dst = np.arange(1, 1 + rows.size)
            out_seq[dst, lo:hi] = self.sequence[rows, lo:hi]
            del_rows = self._deletion_rows(rows, lo, hi)
            out_hd[dst, lo:hi] = del_rows > 0
            out_dv[dst, lo:hi] = self._dv_of(del_rows)

        return np.arange(depth, dtype=int), out_seq, out_hd, out_dv

    def sample(
        self,
        max_msa_depth: int = 256,
        ratio: tuple[float, float] = (0.5, 0.5),
        rng: np.random.Generator | None = None,
        randomize: bool = False,
        compact: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly sample sequences from the complex MSA.

        ``no_pairing`` mode normally skips the random draw and takes the first
        ``max_msa_depth`` rows of :attr:`sequence` straight off the top —
        i.e. each chain's top-N a3m hits, lined up positionally. No species
        pairing, no shuffling.

        ``randomize=True`` instead draws ``max_msa_depth - 1`` rows uniformly
        without replacement from rows ``[1, N)`` and keeps them in their original
        order behind the query at row 0. Used at inference to give each seed a
        different alignment: the top-N truncation is identical for every seed, so
        without this a 5-seed ensemble sees one fixed MSA and the seeds differ only
        in the reference conformers and the diffusion noise.
        """
        if rng is None:
            rng = np.random.default_rng()

        if getattr(self, "pairing_mode", "mixed") == "no_pairing":
            total = int(self.sequence.shape[0])
            n = min(max_msa_depth, total)
            if randomize and n > 1:
                return self._subsample_no_pairing(n, rng, compact)
            idx = np.arange(n, dtype=int)
            del_rows = self._deletion_rows(idx)
            return (
                idx,
                self.sequence[idx],
                (del_rows > 0).astype(np.uint8),
                self._dv_of(del_rows),
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
        del_rows = self._deletion_rows(sampled_indices)
        sampled_has_deletion = (del_rows > 0).astype(np.uint8)
        sampled_deletion_value = self._dv_of(del_rows)

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
    sample_depth: Literal["uniform", "fixed", "random", "af3"] = "uniform",
    compact: bool = False,
) -> MSAFeatures:
    """Sample and process MSA for model input.

    ``sample_depth="uniform"`` (default) draws the per-item depth
    k ~ Uniform[1, min(n_available, max_msa_depth)] and takes the FIRST k rows,
    so the model sees a range of depths but never a row past ``max_msa_depth``.
    ``sample_depth="fixed"`` always requests max_msa_depth (legacy behavior).
    ``sample_depth="random"`` also requests max_msa_depth but draws WHICH rows
    at random (query kept), so two rngs give two different alignments of the
    same depth -- the inference-side ensemble knob.

    ``sample_depth="af3"`` (v1.2.0) is the AF3 SI 2.2 data-stage rule: the depth
    is drawn over the FULL stored alignment, k ~ Uniform[1, n_available], and the
    rows are then cropped to the ``max_msa_depth`` budget -- AF3 shuffles before it
    crops, so the kept rows are a random per-chain subset, not the best-first
    prefix. Two things follow. Deep alignments saturate the budget most of the
    time (P(k >= budget) = 1 - budget/n), instead of a median depth of budget/2,
    and rows past the budget are actually seen. The budget itself is unchanged,
    so tensor shapes and MSA buckets are the same as under "uniform".
    """
    if rng is None:
        rng = np.random.default_rng()

    if getattr(msa, "pairing_mode", "mixed") == "no_pairing":
        n_available = int(msa.sequence.shape[0])
    else:
        n_available = int(msa.total_depth)

    if sample_depth == "uniform":
        upper = max(1, min(n_available, max_msa_depth))
        effective_depth = int(rng.integers(1, upper + 1))
    elif sample_depth == "af3":
        k = int(rng.integers(1, max(1, n_available) + 1))
        effective_depth = min(k, max_msa_depth)
    else:
        effective_depth = max_msa_depth

    profile = msa.profile
    deletion_mean = msa.deletion_mean
    _, aligned_sequences, has_deletion, deletion_value = msa.sample(
        effective_depth,
        rng=rng,
        randomize=sample_depth in ("random", "af3"),
        compact=compact,
    )
    n_seq, _ = aligned_sequences.shape
    mask = np.ones((n_seq), dtype=np.float32)

    if compact:
        # Host tensors at the width the values need (residue classes < 32, has_deletion 0/1, float32 deletion value). The model casts them
        # itself before use (``.long()`` one-hot, concat with float features, ``.float()``), so what it computes from them is unchanged.
        aligned_sequences = aligned_sequences.astype(np.int8, copy=False)
        has_deletion_t = torch.from_numpy(has_deletion.astype(np.int8, copy=False))
        deletion_value = deletion_value.astype(np.float32, copy=False)
    else:
        has_deletion_t = torch.from_numpy(has_deletion).int()

    return MSAFeatures.from_sample(
        aligned_sequences=torch.from_numpy(aligned_sequences),
        mask=torch.from_numpy(mask),
        has_deletion=has_deletion_t,
        deletion_value=torch.from_numpy(deletion_value),
        profile=torch.from_numpy(profile),
        deletion_mean=torch.from_numpy(deletion_mean),
    )
