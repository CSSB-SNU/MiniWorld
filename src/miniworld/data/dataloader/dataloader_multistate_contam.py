from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import torch
from numpy import ndarray
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, DistributedSampler

from miniworld.data.crop import Cropper
from miniworld.data.features.features_multistate import (
    Batch,
    ContamFeatures,
    MSAFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    SequenceFeatures,
    StructureFeatures,
)
from miniworld.data.io import extract_lmdb_keys, load_a3m, load_cifmols, load_fasta
from miniworld.data.mapping import AtomMapping
from miniworld.data.msa import MSA
from miniworld.utils.structure import SE3_oper


class BioMolSampler(DistributedSampler):
    """To be removed."""

    class Config(BaseModel):
        """To be removed."""

        # 1) let Pydantic accept any Python class as a field
        model_config = ConfigDict(arbitrary_types_allowed=True)

        # 2) annotate with the real class (not a primitive)
        dataset: BioMolMonomerData
        num_replicas: int = 1
        rank: int = 0
        shuffle: bool = True
        seed: int = 0
        drop_last: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__(
            dataset=config.dataset,
            num_replicas=config.num_replicas,
            rank=config.rank,
            shuffle=config.shuffle,
            seed=config.seed,
            drop_last=config.drop_last,
        )
        self.dataset = config.dataset

    def set_epoch(self, epoch: int) -> None:
        """To be removed."""
        self.epoch = epoch


class CropConfig(BaseModel):
    """Configuration for cropping strategy."""

    contiguous_prob: float = 0.5
    spatial_prob: float = 0.5
    interface_prob: float = 0.0
    crop_length: int = 384
    monomer: bool = True


class KmerFastAlignConfig(BaseModel):
    """Configuration for kmer fast alignment."""

    kmer_index: Path = Path("kmer_index.tsv")
    fasta: Path = Path("sequence_hashes.fasta")
    kmer_threshold: float = 0.2
    gap_split: int = 15
    max_mismatch: int = 15
    max_indel: int = 5
    align_num: int = -1  # -1 means no limit
    align_thr: float = 0.9
    seed: int = 1123  # for reproducibility


class MSAConfig(BaseModel):
    """Configuration for MSA sampling."""

    n_samples: int = 4
    max_msa_depth: int = 512


class MultistateConfig(BaseModel):
    """Configuration for multistate modeling."""

    n_prefilter: int = 128
    n_samples: int = 48
    temperatures: float = 1.0
    consensus_ratio: float = 0.9
    consensus_filter: float = 0.9


class MultiStatedbConfig(BaseModel):
    """Configuration for BioMoldb paths."""

    cif_db_path: Path = Path("cif_lmdb")
    a3m_db_path: Path = Path("a3m_lmdb")
    cluster_ids_path: Path = Path("cluster_ids.txt")
    cluster_id_to_seq_ids_path: Path = Path("cluster_id_to_seq_ids.npz")
    seq_id_to_seq: Path = Path("protein.fasta")
    tmp_dir: Path = Path("./data_tmp/monomer/")
    load_all_msa: bool = False


class BioMolMonomerData(torch.utils.data.Dataset):
    """Dataset for BioMol monomer data with contamination."""

    class BioMolConfig(BaseModel):
        """Configuration for BioMolMonomerData."""

        crop_config: CropConfig = CropConfig()
        msa_config: MSAConfig = MSAConfig()
        kmer_fast_align_config: KmerFastAlignConfig = KmerFastAlignConfig()
        multistate_config: MultistateConfig = MultistateConfig()
        preprocess_config: MultiStatedbConfig = MultiStatedbConfig()

    def __init__(
        self,
        config: BioMolConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.cropper = Cropper(
            contiguous_prob=config.crop_config.contiguous_prob,
            spatial_prob=config.crop_config.spatial_prob,
            interface_prob=config.crop_config.interface_prob,
            monomer_only=config.crop_config.monomer,
        )
        self.kmer_align_options = {
            "kmer_threshold": config.kmer_fast_align_config.kmer_threshold,
            "gap_split": config.kmer_fast_align_config.gap_split,
            "max_mismatch": config.kmer_fast_align_config.max_mismatch,
            "max_indel": config.kmer_fast_align_config.max_indel,
            "align_num": config.kmer_fast_align_config.align_num,
            "align_thr": config.kmer_fast_align_config.align_thr,
        }

        self._load_preprocessed()

    def _load_preprocessed(self) -> None:
        """Load preprocessed metadata from cache or preprocess if not available."""
        tmp_dir = self.config.preprocess_config.tmp_dir
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_cache_path = tmp_dir / "preprocessed_metadata.pkl"

        if tmp_cache_path.exists():
            with tmp_cache_path.open("rb") as f:
                data = pickle.load(f)  # noqa: S301
            self.cluster_id_to_seq_ids = data["cluster_id_to_seq_ids"]
            self.seq_id_to_cluster_id = {}
            for cluster_id, seq_ids in self.cluster_id_to_seq_ids.items():
                for seq_id in seq_ids:
                    self.seq_id_to_cluster_id[seq_id] = cluster_id
            self.cluster_id_list = data["cluster_id_list"]
            self.seq_id_to_seq = data["seq_id_to_seq"]
            return

        cluster_id_to_seq_ids_data = np.load(
            self.config.preprocess_config.cluster_id_to_seq_ids_path,
            allow_pickle=True,
        )
        cluster_id_to_seq_ids = {
            cluster_id: seq_ids[0].split(",")
            for cluster_id, seq_ids in zip(
                cluster_id_to_seq_ids_data["clusterIDs"],
                cluster_id_to_seq_ids_data["seqIDs"],
                strict=True,
            )
        }

        valid_seq_ids = extract_lmdb_keys(self.config.preprocess_config.cif_db_path)
        filtered_cluster_ids_to_seq_ids = {
            cluster_id: [seq_id for seq_id in seq_ids if seq_id in valid_seq_ids]
            for cluster_id, seq_ids in cluster_id_to_seq_ids.items()
            if any(seq_id in valid_seq_ids for seq_id in seq_ids)
        }
        self.cluster_id_to_seq_ids = filtered_cluster_ids_to_seq_ids
        self.seq_id_to_cluster_id = {}
        for cluster_id, seq_ids in self.cluster_id_to_seq_ids.items():
            for seq_id in seq_ids:
                self.seq_id_to_cluster_id[seq_id] = cluster_id

        cluster_id_list = []
        with self.config.preprocess_config.cluster_ids_path.open("r") as f:
            for line in f:
                cluster_id = line.strip()
                if cluster_id in self.cluster_id_to_seq_ids:
                    cluster_id_list.append(cluster_id)
        self.cluster_id_list = cluster_id_list

        other_type = [
            cid[1] not in ("P", "A") and cid != "Ab_cluster"
            for cid in self.cluster_id_list
        ]
        if any(other_type):
            cid_other = [
                cid for cid in self.cluster_id_list if cid[1] not in ("P", "A")
            ]
            msg = f"Non-protein cluster_ids found: {cid_other}"
            raise ValueError(msg)

        seq_id_to_seq = load_fasta(self.config.preprocess_config.seq_id_to_seq)
        self.seq_id_to_seq = {
            k: v for k, v in seq_id_to_seq.items() if k in valid_seq_ids
        }

        data = {
            "cluster_id_to_seq_ids": self.cluster_id_to_seq_ids,
            "cluster_id_list": self.cluster_id_list,
            "seq_id_to_seq": self.seq_id_to_seq,
        }
        with tmp_cache_path.open("wb") as f:
            pickle.dump(data, f)

    def __len__(self) -> int:
        return len(self.cluster_id_list)

    def _gen_kmer_query(self, query_full_seq: str, crop_indices: ndarray) -> str:
        # Step 1: Build mask
        seq_len = len(query_full_seq)
        mask = np.zeros(seq_len, dtype=bool)
        mask[crop_indices] = True

        # Step 2: Build gapped sequence
        gapped_seq = "".join(
            res if keep else "-" for res, keep in zip(query_full_seq, mask, strict=True)
        )

        # Step 3: Trim leading and trailing gaps
        trimmed_seq = gapped_seq.strip("-")
        gap_indices = [i for i, char in enumerate(trimmed_seq) if char == "-"]

        return trimmed_seq, gap_indices

    def __getitem__(self, idx: int) -> Batch:
        cluster_id = self.cluster_id_list[idx]
        seq_ids = self.cluster_id_to_seq_ids[cluster_id]
        if len(seq_ids) == 0:
            msg = f"No valid seqs found for cluster_id {cluster_id} at index {idx}."
            raise ValueError(msg)

        # uniformly sample a seq _id
        # To avoid this, we will loop until we find a valid one.
        query_id = random.choice(seq_ids)
        try:
            item = self.get_item_by_seq_id(query_id=query_id)
        except:
            msg = f"Error loading data for seq_id {query_id}."
            raise ValueError(msg)
        return item

    def get_item_by_seq_id(
        self,
        query_id: str,
        contam_query_id: str | None = None,
    ) -> Batch:
        """Get item by seq_id."""
        cluster_id = self.seq_id_to_cluster_id[query_id]
        (
            query_sequence,
            query_structure,
            query_reference,
            query_scheme,
            query_msa,
            additional,
        ) = self._load_data_by_seq_id(query_id=query_id, crop_length=None)
        query_residue_length = query_msa.aligned_sequences.shape[-1]
        query_msa_depth = query_msa.aligned_sequences.shape[0]

        # contam query _id
        if not contam_query_id:
            contam_cluster_id = random.choice(
                list(set(self.cluster_id_list) - {cluster_id}),
            )
            contam_query_id = random.choice(
                self.cluster_id_to_seq_ids[contam_cluster_id],
            )
        else:
            contam_cluster_id = self.seq_id_to_cluster_id[contam_query_id]

        _, contam_structure, _, contam_scheme, contam_msa, _ = (
            self._load_data_by_seq_id(
                query_id=contam_query_id,
                crop_length=query_residue_length,
            )
        )
        contam_msa_depth = contam_msa.aligned_sequences.shape[2]
        contam_residue_length = contam_msa.aligned_sequences.shape[-1]
        if contam_residue_length < query_residue_length:
            random_bias = random.randint(
                0,
                query_residue_length - contam_residue_length,
            )
        else:
            random_bias = 0
        contam_msa_aligned_sequences = torch.full(
            (1, contam_msa_depth, query_residue_length),
            fill_value=31,
        )  # gap token
        contam_msa_has_deletion = torch.zeros(
            (1, contam_msa_depth, query_residue_length),
            dtype=torch.int,
        )
        contam_msa_deletion_value = torch.zeros(
            (1, contam_msa_depth, query_residue_length),
            dtype=torch.float,
        )
        contam_msa_profile = query_msa.profile[0].clone()
        contam_msa_deletion_mean = query_msa.deletion_mean[0].clone()
        contam_msa_aligned_sequences[
            :,
            :,
            random_bias : random_bias + contam_residue_length,
        ] = contam_msa.aligned_sequences[0]
        contam_msa_has_deletion[
            :,
            :,
            random_bias : random_bias + contam_residue_length,
        ] = contam_msa.has_deletion[0]
        contam_msa_deletion_value[
            :,
            :,
            random_bias : random_bias + contam_residue_length,
        ] = contam_msa.deletion_value[0]
        contam_msa_profile[random_bias : random_bias + contam_residue_length] = (
            contam_msa.profile[0]
        )
        contam_msa_deletion_mean[random_bias : random_bias + contam_residue_length] = (
            contam_msa.deletion_mean[0]
        )

        # mix msa
        resample_depth = min(query_msa_depth, contam_msa_depth)

        aligned_sequences = torch.cat(
            [
                query_msa.aligned_sequences[0, :, :resample_depth],
                contam_msa_aligned_sequences[:resample_depth],
            ],
            dim=1,
        )
        has_deletion = torch.cat(
            [
                query_msa.has_deletion[0, :, :resample_depth],
                contam_msa_has_deletion[:resample_depth],
            ],
            dim=1,
        )
        deletion_value = torch.cat(
            [
                query_msa.deletion_value[0, :, :resample_depth],
                contam_msa_deletion_value[:resample_depth],
            ],
            dim=1,
        )
        profile = 0.5 * (query_msa.profile[0] + contam_msa_profile)
        deletion_mean = 0.5 * (query_msa.deletion_mean[0] + contam_msa_deletion_mean)

        msa = MSAFeatures.from_sample(
            aligned_sequences=aligned_sequences,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            profile=profile,
            deletion_mean=deletion_mean,
        )

        contam = ContamFeatures.from_sample(
            atom_pos=contam_structure.atom_pos[0],
            atom_pos_mask=contam_structure.atom_pos_mask[0],
            atom_to_residue_idx_map=contam_scheme.atom_to_residue_idx_map[0],
        )

        return Batch(
            name=[
                f"{cluster_id}_{query_id}",
                f"{contam_cluster_id}_{contam_query_id}",
            ],
            heteros=[additional[0]],
            atom_ids=[additional[1]],
            chem_comp_ids=[additional[2]],
            sequence=query_sequence,
            structure=query_structure,
            reference=query_reference,
            contam=contam,
            scheme=query_scheme,
            msa=msa,
            contam_bias=[random_bias],
        )

    def _load_data_by_seq_id(  # noqa: PLR0915
        self,
        query_id: str,
        crop_length: int | None = None,
    ) -> tuple[
        SequenceFeatures,
        StructureFeatures,
        ReferenceFeatures,
        SchemeFeatures,
        MSAFeatures,
        tuple,
    ]:
        """Get item by seq_id."""
        cifmols = load_cifmols(
            db_path=self.config.preprocess_config.cif_db_path,
            seq_id=query_id,
        )
        atom_length_list = [cifmol.atoms.xyz.shape[0] for cifmol in cifmols]
        atom_length = random.choice(list(set(atom_length_list)))
        indices = [
            i for i, length in enumerate(atom_length_list) if length == atom_length
        ]
        cifmols = [cifmols[i] for i in indices]

        # remove fully invalid atoms

        # for cropping we have to sample one _id
        query_cifmol = random.choice(cifmols)

        if crop_length is None:
            crop_length = self.config.crop_config.crop_length
        crop_indices, chain_id_to_crop_indices = self.cropper.crop(
            cifmol=query_cifmol,
            crop_length=crop_length,
        )
        cifmols = [cifmol.residues[crop_indices].extract() for cifmol in cifmols]
        query_cifmol = query_cifmol.residues[crop_indices].extract()
        query_atom_length = query_cifmol.atoms.xyz.shape[0]

        atom_pos_mask = [
            ~np.isnan(cifmol.atoms.xyz.value).any(axis=1) for cifmol in cifmols
        ]
        atom_len_filter = [
            cifmol.atoms.xyz.shape[0] == query_atom_length for cifmol in cifmols
        ]
        valid_indices = [
            i
            for i, (mask, length_match) in enumerate(
                zip(atom_pos_mask, atom_len_filter, strict=True),
            )
            if mask.any() and length_match
        ]
        if len(valid_indices) == 0:
            msg = f"No valid atoms found for seq_id {query_id} after cropping."
            raise ValueError(msg)
        cifmols = [cifmols[i] for i in valid_indices]

        # Assume only protein chains are handled here
        query_full_seq = self.seq_id_to_seq[query_id]

        if (
            len(query_full_seq) < crop_indices.min()
            or len(query_full_seq) < crop_indices.max() + 1
        ):
            msg = f"Crop indices out of range for seq_id {query_id}."
            raise ValueError(msg)

        atom_pos = [cifmol.atoms.xyz.value for cifmol in cifmols]
        atom_pos_mask = [
            ~np.isnan(cifmol.atoms.xyz.value).any(axis=1) for cifmol in cifmols
        ]

        atom_pos = np.array(atom_pos)  # (N, L_atom, 3)
        atom_pos_mask = np.array(atom_pos_mask)  # (N, L_atom)
        atom_mask = np.ones_like(atom_pos_mask[0], dtype=bool)

        msa = load_a3m(key=query_id, env_path=self.config.preprocess_config.a3m_db_path)
        if msa is None:
            msa = MSA.from_query(
                query=query_full_seq,
                seq_id=query_id,
                a3m_type="protein",
            )

        msa = MSA.cropped(msa, crop_indices)

        # sample MSA
        msa_profile = msa.profile
        msa_deletion_mean = msa.deletion_mean
        msa_sequence_sampled = []
        msa_has_deletion_sampled = []
        msa_deletion_value_sampled = []
        for _ in range(self.config.msa_config.n_samples):
            sampled_sequence, sampled_deletions, sampled_species = msa.sample(
                self.config.msa_config.max_msa_depth,
            )
            msa_sequence_sampled.append(sampled_sequence)  # (N_seq, L)
            msa_has_deletion_sampled.append(sampled_deletions > 1)  # (N_seq, L)
            msa_deletion_value_sampled.append(sampled_deletions)  # (N_seq, L)
        msa_sequence_sampled = np.stack(
            msa_sequence_sampled,
            axis=0,
        )  # (N_sample, N_seq, L)
        msa_has_deletion_sampled = np.stack(msa_has_deletion_sampled, axis=0)
        msa_deletion_value_sampled = np.stack(
            msa_deletion_value_sampled,
            axis=0,
        ).astype(np.float32)

        # Tensor of residue xyz, mask, bond
        residue_length = len(query_cifmol.residues)

        # Now convert biomol to batch
        atom_bond_type = query_cifmol.atoms.bond_type.value  # (n_atom_bond, )
        atom_bond_stereo = query_cifmol.atoms.bond_stereo.value  # (n_atom_bond, )
        atom_bond_aromatic = query_cifmol.atoms.bond_aromatic.value  # (n_atom_bond, )
        atom_bond = np.stack(
            [atom_bond_type, atom_bond_stereo, atom_bond_aromatic],
            axis=1,
        )  # (n_atom_bond, 3)
        atom_bond = np.zeros_like(atom_bond, dtype=np.int64)
        residue_bond = query_cifmol.residues.bond
        value, src, dst = (
            residue_bond.value,
            residue_bond.src_indices,
            residue_bond.dst_indices,
        )
        residue_bond = np.stack([src, dst, value], axis=1)  # (n_residue_bond, 3)

        # idx map
        atom_to_residue_idx_map = query_cifmol.index_table.atom_to_res

        # ids
        chain_num = query_cifmol.chains.chain_id.shape[0]
        chain_asym_id = np.arange(chain_num).astype(np.int64)
        chain_entity_id = query_cifmol.chains.entity_id.value
        same_entity = chain_entity_id[:, None] == chain_entity_id[None, :]
        chain_sym_id = np.triu(same_entity, k=0).sum(axis=0) - 1

        residue_idx = query_cifmol.residues.cif_idx.value
        residue_idx_mono = np.arange(residue_length, dtype=np.int64)
        residue_to_chain = query_cifmol.index_table.res_to_chain
        residue_asym_id = np.take(chain_asym_id, residue_to_chain)
        residue_entity_id = np.take(chain_entity_id, residue_to_chain)
        residue_sym_id = np.take(chain_sym_id, residue_to_chain)

        residue_type = msa_sequence_sampled[0, 0]
        ref_pos = query_cifmol.atoms.model_xyz.value
        ref_pos = np.array(ref_pos, dtype=object)

        mask = (ref_pos == "?") | (ref_pos == ".")
        ref_pos[mask] = 0.0
        ref_pos = ref_pos.astype(np.float32, copy=False)

        ref_mask = ~np.isnan(ref_pos).any(axis=1)
        ref_element = query_cifmol.atoms.element.value
        ref_charge = query_cifmol.atoms.charge.value
        ref_space_uid = query_cifmol.index_table.atom_to_res

        N_res = ref_space_uid.max() + 1
        res_to_atoms = [np.where(ref_space_uid == i)[0] for i in range(N_res)]

        Rs, Ts = SE3_oper(residue_length)
        random_ref_pos = []
        for ii, atom_indices in enumerate(res_to_atoms):
            R, T = Rs[ii], Ts[ii]
            _ref_pos = ref_pos[atom_indices]
            _ref_pos = (
                _ref_pos - _ref_pos.mean(axis=0)
            ) @ R + T  # random SE(3) operation
            random_ref_pos.append(_ref_pos)
        ref_pos = np.vstack(random_ref_pos)

        if ref_pos.shape[0] != atom_pos[0].shape[0]:
            msg = f"Mismatch in atom numbers for seqID {query_id}: ref_pos has {ref_pos.shape[0]} atoms, but atom_pos has {atom_pos[0].shape[0]} atoms."
            raise ValueError(msg)

        # convert str to int
        ref_element = AtomMapping().atom_to_index(ref_element)

        sequence = SequenceFeatures.from_sample(
            residue_type=torch.from_numpy(residue_type.astype(np.int64)),
        )

        # test
        atom_pos = np.where(atom_pos_mask[..., None], atom_pos, 0.0)
        structure = StructureFeatures.from_sample(
            atom_pos=torch.from_numpy(atom_pos.astype(np.float32)),
            atom_pos_mask=torch.from_numpy(atom_pos_mask.astype(np.bool)),
            atom_mask=torch.from_numpy(atom_mask.astype(np.bool)),
            residue_mask=torch.ones((residue_length,), dtype=torch.bool),  # all ones
            atom_bond=torch.from_numpy(atom_bond.astype(np.int64)),
            residue_bond=torch.from_numpy(residue_bond.astype(np.int64)),
        )
        reference = ReferenceFeatures.from_sample(
            pos=torch.from_numpy(ref_pos.astype(np.float32)),
            mask=torch.from_numpy(ref_mask.astype(np.bool)),
            element=torch.from_numpy(ref_element.astype(np.int64)),
            charge=torch.from_numpy(ref_charge.astype(np.float32)),
            space_uid=torch.from_numpy(ref_space_uid.astype(np.int64)),
        )
        scheme = SchemeFeatures.from_sample(
            crop_indices=torch.from_numpy(crop_indices.astype(np.int64)),
            residue_idx=torch.from_numpy(residue_idx.astype(np.int64)),
            residue_idx_mono=torch.from_numpy(residue_idx_mono.astype(np.int64)),
            residue_asym_id=torch.from_numpy(residue_asym_id.astype(np.int64)),
            residue_entity_id=torch.from_numpy(residue_entity_id.astype(np.int64)),
            residue_sym_id=torch.from_numpy(residue_sym_id.astype(np.int64)),
            atom_to_residue_idx_map=torch.from_numpy(
                atom_to_residue_idx_map.astype(np.int64),
            ),
        )

        msa = MSAFeatures.from_sample(
            aligned_sequences=torch.from_numpy(msa_sequence_sampled),
            has_deletion=torch.from_numpy(msa_has_deletion_sampled).int(),
            deletion_value=torch.from_numpy(msa_deletion_value_sampled),
            profile=torch.from_numpy(msa_profile),
            deletion_mean=torch.from_numpy(msa_deletion_mean),
        )

        # ids : to make cif file from batch
        hetero = query_cifmol.residues.hetero
        atom_ids = query_cifmol.atoms.atom_id
        chem_comp_ids = query_cifmol.residues.chem_comp
        additional = (hetero, atom_ids, chem_comp_ids)

        return sequence, structure, reference, scheme, msa, additional

    def create_ddp_dataloader(
        self,
        rank: int,
        drop_last: bool = False,
        **kwargs: object,
    ) -> DataLoader:
        """Create a distributed DataLoader with AdaptiveEdgeSampler."""
        sampler = BioMolSampler(
            BioMolSampler.Config(
                dataset=self,
                num_replicas=kwargs.get("world_size", 1),
                rank=rank,
                shuffle=kwargs.get("shuffle", True),
                seed=kwargs.get("seed", 0),
                drop_last=drop_last,
            ),
        )

        kwargs.pop("shuffle", None)
        kwargs.pop("world_size", None)
        kwargs.update({"sampler": sampler})

        params = {
            "shuffle": False,  # leave False when using a sampler
            "drop_last": False,  # override to True for train
            "num_workers": kwargs.get("num_workers", 0),
            "pin_memory": True,
            "multiprocessing_context": (
                "spawn" if kwargs.get("num_workers", 0) > 0 else None
            ),
            "collate_fn": Batch.collate_fn,
        }
        params.update(kwargs)
        return DataLoader(self, **params)


if __name__ == "__main__":
    db_path = Path("/home/psk6950/data/BioMoldbv2_204Oct21/")
    config = BioMolMonomerData.BioMolConfig(
        crop_config=CropConfig(
            contiguous_prob=1.0,
            spatial_prob=0.0,
            interface_prob=0.0,
            crop_length=384,
        ),
        kmer_fast_align_config=KmerFastAlignConfig(
            kmer_index=db_path / "kmer_align" / "kmer_index.tsv",
            fasta=db_path / "fasta" / "protein_wo_signalp.fasta",
            kmer_threshold=0.2,
            gap_split=15,
            max_mismatch=15,
            max_indel=5,
            align_num=-1,  # -1 means no limit
            align_thr=0.9,
            seed=1123,  # for reproducibility
        ),
        msa_config=MSAConfig(
            n_samples=1,
            max_msa_depth=512,
        ),
        multistate_config=MultistateConfig(
            n_prefilter=48,
            n_samples=48,
            temperatures=1.0,
            consensus_ratio=0.9,
            consensus_filter=0.9,
        ),
        preprocess_config=MultiStatedbConfig(
            cif_db_path=Path(
                "/home/psk6950/data/MiniWorld/multistate/seqID_to_cifmols_filtered.lmdb",
            ),
            a3m_db_path=Path("/home/psk6950/data/BioMolDBv2_2024Oct21/slim_a3m.lmdb"),
            tmp_dir=Path(
                "/home/psk6950/data/MiniWorld/multistate/data_tmp/monomer/train/",
            ),
            cluster_ids_path=Path(
                "/home/psk6950/data/MiniWorld/multistate/train_valid_split/train_clusters.txt",
            ),
            cluster_id_to_seq_ids_path=Path(
                "/home/psk6950/data/MiniWorld/multistate/clusterID_to_seqIDs.npz",
            ),
            seq_id_to_seq=db_path / "fasta" / "protein_wo_signalp.fasta",
        ),
    )
    dataset = BioMolMonomerData(config)
    for _ in range(1000):
        dataset._load_data_by_seq_id("P0205891")
        dataset._load_data_by_seq_id("P0205892")
        dataset._load_data_by_seq_id("P0205893")
    breakpoint()
