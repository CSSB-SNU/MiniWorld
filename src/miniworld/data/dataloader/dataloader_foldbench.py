from __future__ import annotations

import re
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader, DistributedSampler

from miniworld.data.dataloader.configs import (
    MSAConfig,
)
from miniworld.data.dataloader.utils import (
    load_msa,
    load_signalp,
    remove_signalp,
    remove_terminal_oxygen,
    remove_water,
    sample_msa,
)
from miniworld.data.features.batch_foldbench import (
    Batch,
    ChainFeatures,
    MSAFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    SequenceFeatures,
    StructureFeatures,
)
from miniworld.data.io import (
    load_bytes,
    load_raw_data,
)
from miniworld.data.mapping import AtomMapping, EntityMapping
from miniworld.data.mols import CIFMolAttached
from miniworld.utils.structure import SE3_oper

if TYPE_CHECKING:
    from biomol.core.types import BioMolDict


def make_batch(  # noqa: PLR0915
    cifmol: CIFMolAttached,
    msa: MSAFeatures,
) -> tuple[
    SequenceFeatures,
    StructureFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    ChainFeatures,
]:
    """Convert CIFMol and MSA to batch features."""
    # Now convert biomol to batch
    atom_bond_type = cifmol.atoms.bond_type.value  # (n_atom_bond, )
    atom_bond_stereo = cifmol.atoms.bond_stereo.value  # (n_atom_bond, )
    atom_bond_aromatic = cifmol.atoms.bond_aromatic.value  # (n_atom_bond, )
    atom_bond = np.stack(
        [atom_bond_type, atom_bond_stereo, atom_bond_aromatic],
        axis=1,
    )  # (n_atom_bond, 3)
    atom_bond = np.zeros_like(atom_bond, dtype=np.int64)  # placeholder
    residue_bond = cifmol.residues.bond
    value, src, dst = (
        residue_bond.value,
        residue_bond.src_indices,
        residue_bond.dst_indices,
    )
    residue_bond = np.stack([src, dst, value], axis=1)  # (n_residue_bond, 3)

    # Tensor of residue xyz, mask, bond
    cropped_len = len(cifmol.residues)

    # idx map
    atom_to_residue_idx_map = cifmol.index_table.atom_to_res

    # ids
    chain_num = cifmol.chains.chain_id.shape[0]
    chain_asym_id = np.arange(chain_num).astype(np.int64)
    chain_entity_id = cifmol.chains.entity_id.value
    same_entity = chain_entity_id[:, None] == chain_entity_id[None, :]
    chain_sym_id = np.triu(same_entity, k=0).sum(axis=0) - 1

    residue_idx = cifmol.residues.cif_idx.value
    residue_idx_mono = np.arange(cropped_len, dtype=np.int64)
    residue_to_chain = cifmol.index_table.res_to_chain
    residue_asym_id = np.take(chain_asym_id, residue_to_chain)
    residue_entity_id = np.take(chain_entity_id, residue_to_chain)
    residue_sym_id = np.take(chain_sym_id, residue_to_chain)

    residue_type = msa.aligned_sequences[0, 0, 0]

    ref_pos = cifmol.atoms.model_xyz.value
    ref_pos = np.array(ref_pos, dtype=object)

    mask = (ref_pos == "?") | (ref_pos == ".")
    ref_pos[mask] = 0.0
    ref_pos = ref_pos.astype(np.float32, copy=False)

    ref_mask = ~np.isnan(ref_pos).any(axis=1)
    ref_element = cifmol.atoms.element.value
    ref_charge = cifmol.atoms.charge.value
    ref_charge = np.array(
        [float(c) if c not in {"?", "."} else 0.0 for c in ref_charge],
    )
    ref_space_uid = cifmol.index_table.atom_to_res

    N_res = ref_space_uid.max() + 1
    res_to_atoms = [np.where(ref_space_uid == i)[0] for i in range(N_res)]

    Rs, Ts = SE3_oper(cropped_len)
    random_ref_pos = []
    for ii, atom_indices in enumerate(res_to_atoms):
        R, T = Rs[ii], Ts[ii]
        _ref_pos = ref_pos[atom_indices]
        _ref_pos = (_ref_pos - _ref_pos.mean(axis=0)) @ R + T  # random SE(3) operation
        random_ref_pos.append(_ref_pos)
    ref_pos = np.vstack(random_ref_pos)

    # convert str to int
    ref_element = AtomMapping().atom_to_index(ref_element)

    # entity type mapping
    entity_mapping = EntityMapping()
    seq_id_list = cifmol.chains.seq_id.value.tolist()
    entity_id_list = [seq_id[0] for seq_id in seq_id_list]
    entity_types = entity_mapping.tag_to_idx(entity_id_list)
    contact = cifmol.chains.contact
    src, dst = contact.src_indices, contact.dst_indices
    contact_edges = list(zip(src, dst, strict=True))

    sequence = SequenceFeatures.from_sample(
        residue_type=residue_type,
    )
    atom_pos = cifmol.atoms.xyz.value
    atom_pos_mask = np.isfinite(atom_pos).all(axis=1)
    atom_mask = np.ones_like(atom_pos_mask, dtype=bool)

    # centering atom_pos
    valid_pos = atom_pos[atom_pos_mask]  # (N_valid, 3)

    mean_vector = valid_pos.mean(axis=0, keepdims=True)
    atom_pos = atom_pos - mean_vector

    atom_pos = np.where(atom_pos_mask.astype(bool)[:, None], atom_pos, 0.0)

    structure = StructureFeatures.from_sample(
        atom_pos=torch.from_numpy(atom_pos.astype(np.float32)),
        atom_pos_mask=torch.from_numpy(atom_pos_mask.astype(np.bool)),
        atom_mask=torch.from_numpy(atom_mask.astype(np.bool)),
        atom_bond=torch.from_numpy(atom_bond.astype(np.int8)),
        residue_mask=torch.ones((cropped_len,), dtype=torch.bool),  # all ones
        residue_bond=torch.from_numpy(residue_bond.astype(np.int8)),
    )
    reference = ReferenceFeatures.from_sample(
        pos=torch.from_numpy(ref_pos.astype(np.float32)),
        mask=torch.from_numpy(ref_mask.astype(np.bool)),
        element=torch.from_numpy(ref_element.astype(np.int64)),
        charge=torch.from_numpy(ref_charge.astype(np.float32)),
        space_uid=torch.from_numpy(ref_space_uid.astype(np.int64)),
    )
    scheme = SchemeFeatures.from_sample(
        residue_idx=torch.from_numpy(residue_idx.astype(np.int64)),
        residue_idx_mono=torch.from_numpy(residue_idx_mono.astype(np.int64)),
        residue_asym_id=torch.from_numpy(residue_asym_id.astype(np.int64)),
        residue_entity_id=torch.from_numpy(residue_entity_id.astype(np.int64)),
        residue_sym_id=torch.from_numpy(residue_sym_id.astype(np.int64)),
        atom_to_residue_idx_map=torch.from_numpy(
            atom_to_residue_idx_map.astype(np.int64),
        ),
    )

    chain = ChainFeatures.from_sample(
        entity_type=torch.from_numpy(entity_types.astype(np.int64)),
        contact_edges=torch.from_numpy(np.array(contact_edges, dtype=np.int64)),
    )

    return (
        sequence,
        structure,
        reference,
        scheme,
        chain,
    )


def load_cifmol(db_path: Path, cif_id: str) -> tuple[dict, dict]:
    """Load CIFMolAttached from LMDB by cif_id."""
    pdb_id, assembly_id, model_id, alt_id = re.findall(
        r"\([^)]*\)|[^_]+",
        cif_id,
    )
    value = load_raw_data(pdb_id, db_path)

    if value is None:
        msg = f"Key '{pdb_id}' not found in LMDB database at '{db_path}'."
        raise KeyError(msg)

    value = load_bytes(value)
    metadata = value["metadata_dict"]
    item = value["assembly_dict"].get(f"{assembly_id}_{model_id}_{alt_id}")
    if item is None:
        msg = f"CIFMol '{cif_id}' not found in LMDB database at '{db_path}'."
        raise KeyError(msg)
    return item, metadata


def parse_foldbench_csv(csv_path: Path) -> list[str]:
    """Parse FoldBench CSV file to get a list of cif_ids."""
    cif_ids = []
    lines = csv_path.read_text().splitlines()
    for line in lines[1:]:  # skip header
        if line.strip() == "":
            continue
        parts = line.split(",")
        cif_id = parts[0].strip()
        pdb_id, assembly_id = cif_id.split("-assembly")
        cif_ids.append(f"{pdb_id}_{assembly_id}_1_.")
    return cif_ids


class FoldBenchDataConfig(BaseModel):
    """Configuration for BioMolDB paths."""

    fold_bench_csv_dir: Path
    fasta_path: Path
    seq_hash_map_path: Path
    signalp_dir: Path | None
    cif_db_path: Path
    a3m_db_path: Path
    residue_max: int = 2560
    atom_max: int = 18000


class FoldBenchData(torch.utils.data.Dataset):
    """Dataset for FoldBench data."""

    class FoldBenchConfig(BaseModel):
        """Configuration for BioMolData."""

        msa_config: MSAConfig = MSAConfig()
        data_config: FoldBenchDataConfig

    def __init__(
        self,
        config: FoldBenchConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self._load_metadata()
        self._load_fold_bench_cif_ids()

    def _load_metadata(self) -> None:
        # load chain id -> sequence hash mapping
        # 1. load fasta file to get chain id -> sequence mapping
        chain_id_to_sequence = {}
        with self.config.data_config.fasta_path.open() as f:
            for line in f:
                if line.startswith(">"):
                    chain_id = line[1:].strip().split(" | ")[0].strip()
                    sequence = next(f).strip()
                    chain_id_to_sequence[chain_id] = sequence
        # 2. load seq_hash_map to get sequence -> sequence hash mapping
        seq_hash_map = {}
        with self.config.data_config.seq_hash_map_path.open() as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.split("\t")
                seq_hash = parts[0].strip()
                sequence = parts[1].strip()
                seq_hash_map[sequence] = seq_hash  # BUG
        # 3. combine to get chain id -> sequence hash mapping
        self.chain_id_to_seq_hash = {}
        for chain_id, sequence in chain_id_to_sequence.items():
            seq_hash = seq_hash_map.get(sequence)
            if seq_hash is None:
                msg = f"Sequence for chain_id '{chain_id}' not found in seq_hash_map."
                raise KeyError(msg)
            self.chain_id_to_seq_hash[chain_id] = seq_hash

        if self.config.data_config.signalp_dir is not None:
            self.signalp_data = load_signalp(self.config.data_config.signalp_dir)
        else:
            self.signalp_data = {}

    def _load_cifmol(self, cif_id: str) -> CIFMolAttached:
        pdb_id = cif_id.split("_")[0]
        data_dict, metadata = load_cifmol(
            db_path=self.config.data_config.cif_db_path,
            cif_id=cif_id,
        )
        chain_ids = data_dict["chains"]["nodes"]["chain_id"]["value"]
        seq_ids = []
        for cid in chain_ids:
            chain_id = cid.split("_")[0]
            key = f"{pdb_id}_{chain_id}".upper()
            seq_id = self.chain_id_to_seq_hash.get(key)
            if seq_id is None:
                msg = f"Sequence hash for chain_id '{key}' not found in chain_id_to_seq_hash."
                raise KeyError(msg)
            seq_ids.append(seq_id)
        cluster_ids = ["none" for _ in chain_ids]
        data_dict["chains"]["nodes"]["seq_id"] = {
            "value": np.array(seq_ids, dtype=str),
        }
        data_dict["chains"]["nodes"]["cluster_id"] = {
            "value": np.array(cluster_ids, dtype=str),
        }
        data_dict["metadata"] = metadata
        data_dict = cast("BioMolDict", data_dict)
        cifmol = CIFMolAttached.from_dict(data_dict)
        cifmol = remove_terminal_oxygen(cifmol)
        cifmol = remove_water(cifmol)
        if self.signalp_data:
            cifmol = remove_signalp(
                cifmol,
                self.signalp_data,
            )
        return cifmol

    def _load_fold_bench_cif_ids(self) -> None:
        # load edge_id to cif_ids mapping
        csv_dir = self.config.data_config.fold_bench_csv_dir
        csv_files = list(csv_dir.glob("*.csv"))
        cif_ids = []
        for csv_file in csv_files:
            cif_ids.extend(parse_foldbench_csv(csv_file))

        # Surprisingly, there are no missing cif_ids in the LMDB, so the filtering step does not remove any cif_id.
        filtered_cif_ids = []
        for cif_id in cif_ids:
            cifmol = self._load_cifmol(cif_id)
            residue_len = len(cifmol.residues)
            atom_len = len(cifmol.atoms)
            if residue_len > self.config.data_config.residue_max:
                continue
            if atom_len > self.config.data_config.atom_max:
                continue
            filtered_cif_ids.append(cif_id)

        # filtered_cif_ids = cif_ids  # no filtering needed

        filtered_cif_ids = list(set(filtered_cif_ids))

        self.cif_ids = filtered_cif_ids

    def __len__(self) -> int:
        """Return the number of edges in the dataset."""
        return len(self.cif_ids)

    def __getitem__(self, idx: int) -> Batch:
        """Get a data sample by index."""
        cif_id = self.cif_ids[idx]

        return self.get_item_by_id(
            cif_id=cif_id,
        )

    def get_item_by_id(
        self,
        cif_id: str,
    ) -> Batch:
        """Get a data sample by cif_id."""
        cifmol = self._load_cifmol(cif_id)

        # Load MSA
        chain_id_to_crop_indices = {}
        chain_ids = cifmol.chains.chain_id.value
        for chain_id in chain_ids:
            chain_id_to_crop_indices[chain_id] = np.arange(
                len(cifmol.chains[cifmol.chains.chain_id == chain_id].residues),
            )

        complex_msa = load_msa(
            cifmol=cifmol,
            chain_id_to_crop_indices=chain_id_to_crop_indices,  # pyright: ignore[reportPossiblyUnboundVariable]
            env_path=self.config.data_config.a3m_db_path,
        )
        msa = sample_msa(
            msa=complex_msa,
            n_samples=self.config.msa_config.n_samples,
            max_msa_depth=self.config.msa_config.max_msa_depth,
        )

        sequence, structure, reference, scheme, chain = make_batch(
            cifmol=cifmol,
            msa=msa,
        )

        # ids : to make cif file from batch
        hetero = cifmol.residues.hetero
        atom_ids = cifmol.atoms.id
        chem_comp_ids = cifmol.residues.chem_comp_id

        return Batch(
            name=[f"{cif_id}"],
            heteros=[hetero],
            atom_ids=[atom_ids],
            chem_comp_ids=[chem_comp_ids],
            sequence=sequence,
            structure=structure,
            reference=reference,
            scheme=scheme,
            msa=msa,
            chain=chain,
        )

    def create_ddp_dataloader(
        self,
        rank: int,
        *,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_workers: int = 0,
        **kwargs: object,
    ) -> DataLoader:
        """Create a distributed DataLoader with AdaptiveEdgeSampler."""
        sampler = DistributedSampler(
            dataset=self,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        kwargs.pop("shuffle", None)
        kwargs.pop("world_size", None)
        kwargs.update({"sampler": sampler})
        if num_workers == 0:
            # remove prefetch_factor
            kwargs.pop("prefetch_factor", None)

        params = {
            "shuffle": False,  # leave False when using a sampler
            "drop_last": False,  # override to True for train
            "num_workers": num_workers,
            "pin_memory": False,
            "multiprocessing_context": ("spawn" if num_workers > 0 else None),
            "collate_fn": Batch.collate_fn,
        }
        params.update(kwargs)
        return DataLoader(self, **params)
