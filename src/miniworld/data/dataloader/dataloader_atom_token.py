from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader, DistributedSampler

from miniworld.data.crop import Cropper, get_chain_crop_indices
from miniworld.data.dataloader.configs import (
    BioMolDBConfig,
    CropConfig,
    EdgeWeightConfig,
    MSAConfig,
)
from miniworld.data.dataloader.utils import load_msa, remove_terminal_oxygen, sample_msa
from miniworld.data.features.batch_edge_backprop import (
    Batch,
    ChainFeatures,
    MSAFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    SequenceFeatures,
    StructureFeatures,
)
from miniworld.data.io import (
    extract_lmdb_keys,
    load_all_raw_data,
    load_cifmol,
)
from miniworld.data.mapping import CANONICAL_CHEMCOMPS, AtomMapping, EntityMapping
from miniworld.utils.structure import SE3_oper

if TYPE_CHECKING:
    from miniworld.data.mols import CIFMolAttached


def tokenize(cifmol: CIFMolAttached) -> tuple[np.ndarray, np.ndarray]:
    """Build mapping from atom indices to token indices and from token indices to residue indices."""
    # tokens are an intermediate grouping between atoms and residues
    atom_to_residue_idx_map = cifmol.index_table.atom_to_res
    canonical_residue_mask = np.isin(
        cifmol.residues.chem_comp_id.value,
        np.array(list(CANONICAL_CHEMCOMPS)),
    )
    canonical = canonical_residue_mask.astype(bool)
    atom_res = atom_to_residue_idx_map
    n_res = canonical.shape[0]

    counts = np.bincount(atom_res, minlength=n_res)
    token_counts = np.where(canonical, 1, counts)

    token_starts = np.cumsum(
        np.concatenate(([0], token_counts[:-1])),
    )

    order = np.argsort(atom_res, kind="stable")
    sorted_res = atom_res[order]

    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_res)) + 1]
    group_sizes = np.diff(np.r_[group_starts, len(atom_res)])

    offsets_sorted = np.arange(len(atom_res)) - np.repeat(group_starts, group_sizes)

    offsets = np.empty_like(offsets_sorted)
    offsets[order] = offsets_sorted

    atom_to_token_idx_map = token_starts[atom_res] + np.where(
        canonical[atom_res],
        0,
        offsets,
    )

    token_residue_idx = np.repeat(np.arange(n_res), token_counts)

    return atom_to_token_idx_map, token_residue_idx


def atom_bonds_to_token_bonds(
    atom_bond_src: np.ndarray,
    atom_bond_dst: np.ndarray,
    atom_bond_value: np.ndarray,  # (n_bonds, 3) e.g., type/stereo/aromatic
    atom_to_token_idx_map: np.ndarray,  # (n_atoms,)
    atom_to_residue_idx_map: np.ndarray,  # (n_atoms,)
    canonical_residue_mask: np.ndarray,  # (n_res,)
) -> tuple[np.ndarray, np.ndarray]:
    """Convert atom-level bonds to token-level bonds, keeping only inter-residue bonds between non-canonical residues."""
    canonical = canonical_residue_mask.astype(bool)

    src = atom_bond_src.astype(np.int64, copy=False)
    dst = atom_bond_dst.astype(np.int64, copy=False)

    src_res = atom_to_residue_idx_map[src]
    dst_res = atom_to_residue_idx_map[dst]

    keep = (src_res == dst_res) & (~canonical[src_res])
    src = src[keep]
    dst = dst[keep]
    value = atom_bond_value[keep]

    t_src = atom_to_token_idx_map[src]
    t_dst = atom_to_token_idx_map[dst]

    a = np.minimum(t_src, t_dst)
    b = np.maximum(t_src, t_dst)
    pairs = np.stack([a, b], axis=1)

    uniq_pairs, uniq_idx = np.unique(pairs, axis=0, return_index=True)
    uniq_value = value[uniq_idx]

    return uniq_pairs.astype(np.int64, copy=False), uniq_value


def make_batch(  # noqa: PLR0915
    cif_id: str,
    cifmol: CIFMolAttached,
    msa: MSAFeatures,
    edge_index: list[int],
) -> Batch:
    """Convert CIFMol and MSA to batch features."""
    atom_to_residue_idx_map = cifmol.index_table.atom_to_res
    canonical_residue_mask = np.isin(
        cifmol.residues.chem_comp_id.value,
        np.array(list(CANONICAL_CHEMCOMPS)),
    )

    atom_to_token_idx_map, token_residue_idx = tokenize(
        cifmol,
    )  # atom level tokenization

    cropped_residue_len = len(cifmol.residues)
    cropped_token_len = token_residue_idx.shape[0]

    """Scheme features"""
    chain_num = cifmol.chains.chain_id.shape[0]
    chain_asym_id = np.arange(chain_num).astype(np.int64)
    chain_entity_id = cifmol.chains.entity_id.value
    same_entity = chain_entity_id[:, None] == chain_entity_id[None, :]
    chain_sym_id = np.triu(same_entity, k=0).sum(axis=0) - 1

    token_idx = np.arange(cropped_token_len, dtype=np.int64)
    res_to_chain = cifmol.index_table.res_to_chain
    token_to_chain = np.take(res_to_chain, token_residue_idx)
    token_asym_id = np.take(chain_asym_id, token_to_chain)
    token_entity_id = np.take(chain_entity_id, token_to_chain)
    token_sym_id = np.take(chain_sym_id, token_to_chain)

    scheme = SchemeFeatures.from_sample(
        token_residue_idx=torch.from_numpy(token_residue_idx.astype(np.int64)),
        token_idx=torch.from_numpy(token_idx.astype(np.int64)),
        token_asym_id=torch.from_numpy(token_asym_id.astype(np.int64)),
        token_entity_id=torch.from_numpy(token_entity_id.astype(np.int64)),
        token_sym_id=torch.from_numpy(token_sym_id.astype(np.int64)),
        atom_to_token_idx_map=torch.from_numpy(
            atom_to_token_idx_map.astype(np.int64),
        ),
        edge_index=torch.from_numpy(np.array(edge_index, dtype=np.int64)),
    )

    """MSA & Sequence features"""
    msa_token = MSAFeatures.from_sample(
        aligned_sequences=msa.aligned_sequences[0, :, :, token_residue_idx],
        has_deletion=msa.has_deletion[0, :, :, token_residue_idx],
        deletion_value=msa.deletion_value[0, :, :, token_residue_idx],
        profile=msa.profile[0, token_residue_idx, :],
        deletion_mean=msa.deletion_mean[0, token_residue_idx],
    )  # msa token mapping
    token_type = msa_token.aligned_sequences[0, 0, 0]
    sequence = SequenceFeatures.from_sample(
        token_type=token_type,
    )

    """Reference features"""
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

    Rs, Ts = SE3_oper(cropped_residue_len)
    random_ref_pos = []
    for ii, atom_indices in enumerate(res_to_atoms):
        R, T = Rs[ii], Ts[ii]
        _ref_pos = ref_pos[atom_indices]
        _ref_pos = (_ref_pos - _ref_pos.mean(axis=0)) @ R + T  # random SE(3) operation
        random_ref_pos.append(_ref_pos)
    ref_pos = np.vstack(random_ref_pos)
    ref_element = AtomMapping().atom_to_index(ref_element)  # convert str to int
    reference = ReferenceFeatures.from_sample(
        pos=torch.from_numpy(ref_pos.astype(np.float32)),
        mask=torch.from_numpy(ref_mask.astype(np.bool)),
        element=torch.from_numpy(ref_element.astype(np.int64)),
        charge=torch.from_numpy(ref_charge.astype(np.float32)),
        space_uid=torch.from_numpy(ref_space_uid.astype(np.int64)),
    )

    """Structure features"""
    atom_pos = cifmol.atoms.xyz.value
    atom_pos_mask = np.isfinite(atom_pos).all(axis=1)
    atom_mask = np.ones_like(atom_pos_mask, dtype=bool)

    # centering atom_pos
    valid_pos = atom_pos[atom_pos_mask]  # (N_valid, 3)
    mean_vector = valid_pos.mean(axis=0, keepdims=True)
    atom_pos = atom_pos - mean_vector
    atom_pos = np.where(atom_pos_mask.astype(bool)[:, None], atom_pos, 0.0)

    # generate token-level bond from residue-level bond and atom-level bond
    residue_bond = cifmol.residues.bond  # canonical bond + branch bond
    residue_src, residue_dst = (
        residue_bond.src,
        residue_bond.dst,
    )
    token_src, token_dst = token_residue_idx[residue_src], token_residue_idx[residue_dst]
    token_canonical_bond = np.stack(
        [token_src, token_dst],
        axis=1,
    )  # (n_residue_bond, 3)
    token_atom_bond, token_atom_bond_value = atom_bonds_to_token_bonds(
        atom_bond_src=cifmol.atoms.bond_type.src,
        atom_bond_dst=cifmol.atoms.bond_type.dst,
        atom_bond_value=cifmol.atoms.bond_type.value,
        atom_to_token_idx_map=atom_to_token_idx_map,
        atom_to_residue_idx_map=atom_to_residue_idx_map,
        canonical_residue_mask=canonical_residue_mask,
    )
    token_bond = np.concatenate([token_canonical_bond, token_atom_bond], axis=0)

    atom_bond_type = cifmol.atoms.bond_type.value  # (n_atom_bond, )
    atom_bond_stereo = cifmol.atoms.bond_stereo.value  # (n_atom_bond, )
    atom_bond_aromatic = cifmol.atoms.bond_aromatic.value  # (n_atom_bond, )
    atom_bond = np.stack(
        [atom_bond_type, atom_bond_stereo, atom_bond_aromatic],
        axis=1,
    )  # (n_atom_bond, 3)
    atom_bond = np.zeros_like(atom_bond, dtype=np.int64)  # placeholder

    structure = StructureFeatures.from_sample(
        atom_pos=torch.from_numpy(atom_pos.astype(np.float32)),
        atom_pos_mask=torch.from_numpy(atom_pos_mask.astype(np.bool)),
        atom_mask=torch.from_numpy(atom_mask.astype(np.bool)),
        atom_bond=torch.from_numpy(atom_bond.astype(np.int64)),
        token_mask=torch.ones((cropped_token_len,), dtype=torch.bool),  # all ones
        token_bond=torch.from_numpy(token_bond.astype(np.int64)),
    )

    """Chain features"""
    entity_mapping = EntityMapping()
    seq_id_list = cifmol.chains.seq_id.value.tolist()
    entity_id_list = [seq_id[0] for seq_id in seq_id_list]
    entity_types = entity_mapping.tag_to_idx(entity_id_list)
    contact = cifmol.chains.contact
    src, dst = contact.src_indices, contact.dst_indices
    contact_edges = list(zip(src, dst, strict=True))

    chain = ChainFeatures.from_sample(
        entity_type=torch.from_numpy(entity_types.astype(np.int64)),
        contact_edges=torch.from_numpy(np.array(contact_edges, dtype=np.int64)),
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
        msa=msa_token,
        chain=chain,
    )


class BioMolData(torch.utils.data.Dataset):
    """Dataset for biomolecular complexes based on BioMolDB."""

    class BioMolConfig(BaseModel):
        """Configuration for BioMolData."""

        crop_config: CropConfig = CropConfig()
        msa_config: MSAConfig = MSAConfig()
        DB_config: BioMolDBConfig = BioMolDBConfig()
        edge_weight_config: EdgeWeightConfig = EdgeWeightConfig()

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
            interface_simple_prob=config.crop_config.interface_simple_prob,
        )

        self._load_edge_to_cif_ids()
        if config.DB_config.load_all_msa:
            self.msa_bytes_dict = load_all_raw_data(
                env_path=self.config.DB_config.a3m_db_path,
            )
        else:
            self.msa_bytes_dict = None

    def _load_edge_to_cif_ids(self) -> None:
        # load edge_id to cif_ids mapping
        pdb_id_list = extract_lmdb_keys(self.config.DB_config.cif_db_path)
        pdb_ids = set(pdb_id_list)
        self.edge_id_to_cif_ids: dict[str, list[str]] = {}
        with self.config.DB_config.edge_id_to_cif_ids_path.open("r") as f:
            for _line in f:
                line = _line.strip()
                if line == "":
                    continue
                key1, key2, value = line.split("\t")
                cif_ids = value.split(",")
                cif_ids = [
                    cif_id for cif_id in cif_ids if cif_id.split("_")[0] in pdb_ids
                ]
                edge_id = f"{key1}_{key2}"
                self.edge_id_to_cif_ids[edge_id] = cif_ids

        self.edge_id_list = list(self.edge_id_to_cif_ids.keys())

    def __len__(self) -> int:
        """Return the number of edges in the dataset."""
        return len(self.edge_id_list)

    def __getitem__(self, idx: int) -> Batch:
        """Get a data sample by index."""
        edge_id = self.edge_id_list[idx]
        cif_ids = self.edge_id_to_cif_ids[edge_id]
        cif_id = random.choice(cif_ids)

        item = self.get_item_by_id(
            cif_id=cif_id,
            chain_bias=None,
            remain_invalid_tokens=self.config.crop_config.remain_invalid_tokens,
        )

        if item.sequence.token_type.shape[1] < 16:
            # too small, resample
            while True:
                idx = random.randint(0, len(self) - 1)
                edge_id = self.edge_id_list[idx]
                cif_ids = self.edge_id_to_cif_ids[edge_id]
                cif_id = random.choice(cif_ids)
                item = self.get_item_by_id(
                    cif_id=cif_id,
                    chain_bias=None,
                    remain_invalid_tokens=self.config.crop_config.remain_invalid_tokens,
                )
                if item.sequence.token_type.shape[1] >= 16:
                    break

        return item

    def get_item_by_id(
        self,
        cif_id: str,
        chain_bias: str | None = None,
        interface_bias: tuple[str, str] | None = None,
        remain_invalid_tokens: bool = False,
        crop_indices: np.ndarray | None = None,
    ) -> Batch:
        """Get a data sample by cif_id."""
        cifmol = load_cifmol(db_path=self.config.DB_config.cif_db_path, cif_id=cif_id)
        chain_id1, chain_id2 = re.findall(r"\([^)]*\)|[^_]+", cif_id)[-2:]
        chain_id1 = chain_id1.strip("()")
        chain_id2 = chain_id2.strip("()")
        if chain_bias is None:
            chain_bias = random.choice([chain_id1, chain_id2])
        if interface_bias is None:
            interface_bias = (chain_id1, chain_id2)

        # Crop
        if crop_indices is None:
            crop_length = self.config.crop_config.residue_crop_length
            if crop_length < 0:
                crop_length = len(cifmol.residues)
            while crop_length > 0:
                crop_indices, chain_id_to_crop_indices = self.cropper.crop(
                    cifmol=cifmol,
                    crop_length=crop_length,
                    chain_bias=chain_bias,
                    interface_bias=interface_bias,
                    remain_invalid_tokens=remain_invalid_tokens,
                )
                token_length = tokenize(cifmol.residues[crop_indices].extract())[
                    1
                ].shape[0]
                if token_length <= crop_length and (
                    cifmol.residues[crop_indices].atoms.element.shape[0]
                    < self.config.crop_config.atom_crop_length
                ):
                    break
                crop_length = int(crop_length * 0.8)  # reduce crop length if too large
        else:
            chain_id_to_crop_indices = get_chain_crop_indices(
                cifmol=cifmol,
                crop_indices=crop_indices,
            )
        crop_indices = cast("np.ndarray", crop_indices)
        cifmol: CIFMolAttached = cifmol.residues[crop_indices].extract()

        cifmol = remove_terminal_oxygen(cifmol)

        contact = cifmol.chains.contact
        src, dst = contact.src_indices, contact.dst_indices
        edge_index = []
        cluster_id_list = cifmol.chains.cluster_id.value.tolist()
        for ss, dd in zip(src, dst, strict=False):
            chain_ss = cluster_id_list[ss]
            chain_dd = cluster_id_list[dd]
            if chain_ss < chain_dd:
                e_id = f"{chain_ss}_{chain_dd}"
            else:
                e_id = f"{chain_dd}_{chain_ss}"
            if e_id not in self.edge_id_list:
                # Ligand-Ligand or SO4, H20, etc.
                continue
            edge_index.append(self.edge_id_list.index(e_id))

        # Load MSA
        complex_msa = load_msa(
            cifmol=cifmol,
            chain_id_to_crop_indices=chain_id_to_crop_indices,  # pyright: ignore[reportPossiblyUnboundVariable]
            env_path=self.config.DB_config.a3m_db_path,
        )
        msa = sample_msa(
            msa=complex_msa,
            n_samples=self.config.msa_config.n_samples,
            max_msa_depth=self.config.msa_config.max_msa_depth,
        )

        return make_batch(
            cif_id=cif_id,
            cifmol=cifmol,
            msa=msa,
            edge_index=edge_index,
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
        # default distributed sampler
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
