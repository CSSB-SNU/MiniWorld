from __future__ import annotations
import json
import functools
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import biomol
import numpy as np
import torch
from pydantic import BaseModel
from team_gm.utils.crop import crop_spatial_segment
from torch.utils.data import DataLoader, DistributedSampler, get_worker_info

from miniworld.configs.data_explicit import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TokenizerConfig,
)
from miniworld.data.constants import AtomMapping, EntityMapping
from miniworld.data.features import (
    Batch,
    ChainFeatures,
    MSAFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    SequenceFeatures,
    StructureFeatures,
)
from miniworld.data.io import (
    load_cifmol,
    load_msa,
)
from miniworld.data.pipeline import (
    Tokenizer,
    get_chain_crop_indices,
    sample_msa,
)
from miniworld.data.pipeline.utils import (
    NoInterfaceError,
    find_interface_residues,
    remove_terminal_oxygen,
)
from miniworld.utils.structure import SE3_oper

if TYPE_CHECKING:
    from collections.abc import Iterator

    from miniworld.data.mols import CIFMolAttached


def atom_bonds_to_token_bonds(
    cifmol: CIFMolAttached,
    atom_to_token_idx_map: np.ndarray,  # (n_atoms,)
) -> np.ndarray:
    """Convert atom-level bonds to token-level bonds, keeping only inter-residue bonds between non-canonical residues."""
    # add covale struct_conn info as token bond
    try:
        sc_value = cifmol.atoms.struct_conn.value[:, 0]
    except biomol.exceptions.FeatureKeyError:
        # to avoid empty bond, add [0,0]
        return np.array([[0, 0]])
    sc_keep = sc_value == "covale"
    sc_src, sc_dst = cifmol.atoms.struct_conn.src, cifmol.atoms.struct_conn.dst
    sc_src = sc_src[sc_keep].astype(np.int64, copy=False)
    sc_dst = sc_dst[sc_keep].astype(np.int64, copy=False)
    sc_t_src = atom_to_token_idx_map[sc_src]
    sc_t_dst = atom_to_token_idx_map[sc_dst]
    sc_a = np.minimum(sc_t_src, sc_t_dst)
    sc_b = np.maximum(sc_t_src, sc_t_dst)
    sc_pairs = np.stack([sc_a, sc_b], axis=1)

    return sc_pairs.astype(np.int64, copy=False)


def make_feature(  # noqa: PLR0915
    cifmol: CIFMolAttached,
    msa: MSAFeatures,
    atom_to_token_idx_map: np.ndarray,
    token_to_residue_idx_map: np.ndarray,
    token_type_override: np.ndarray | None = None,
) -> tuple[
    SequenceFeatures,
    StructureFeatures,
    MSAFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    ChainFeatures,
]:
    """Convert CIFMol and MSA to batch features."""
    cropped_residue_len = len(cifmol.residues)
    cropped_token_len = token_to_residue_idx_map.shape[0]

    """Scheme features"""
    chain_num = cifmol.chains.chain_id.shape[0]
    chain_asym_id = np.arange(chain_num).astype(np.int64)
    chain_entity_id = cifmol.chains.entity_id.value
    same_entity = chain_entity_id[:, None] == chain_entity_id[None, :]
    chain_sym_id = np.triu(same_entity, k=0).sum(axis=0) - 1

    token_idx = np.arange(cropped_token_len, dtype=np.int64)
    res_to_chain = cifmol.index_table.res_to_chain

    token_residue_idx = np.take(cifmol.residues.cif_idx.value, token_to_residue_idx_map)
    token_to_chain = np.take(res_to_chain, token_to_residue_idx_map)
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
    )

    """MSA & Sequence features"""
    msa_token = MSAFeatures.from_sample(
        aligned_sequences=msa.aligned_sequences[0, :, :, token_to_residue_idx_map],
        msa_mask=msa.msa_mask[0, :],
        has_deletion=msa.has_deletion[0, :, :, token_to_residue_idx_map],
        deletion_value=msa.deletion_value[0, :, :, token_to_residue_idx_map],
        profile=msa.profile[0, token_to_residue_idx_map, :],
        deletion_mean=msa.deletion_mean[0, token_to_residue_idx_map],
    )  # msa token mapping
    if token_type_override is None:
        token_type = msa_token.aligned_sequences[0, 0, 0]
    else:
        token_type = torch.from_numpy(token_type_override.astype(np.int64))

    if not isinstance(token_type, torch.Tensor):
        token_type = torch.as_tensor(token_type, dtype=torch.long)
    else:
        token_type = token_type.long()

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

    token_bond = atom_bonds_to_token_bonds(
        cifmol=cifmol,
        atom_to_token_idx_map=atom_to_token_idx_map,
    )

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

    chain = ChainFeatures.from_sample(
        entity_type=torch.from_numpy(entity_types.astype(np.int64)),
    )

    return sequence, structure, msa_token, reference, scheme, chain


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _bucketed_collate(
    batch_list: list[Batch],
    bucket_msa_multiple: int | None = None,
    bucket_token_multiple: int | None = None,
    bucket_atom_multiple: int | None = None,
) -> Batch:
    """Collate with shape bucketing for torch.compile cache efficiency.

    Collates the batch normally, then pads dimensions to bucket boundaries by
    collating with a dummy empty batch of the bucketed size and discarding it.
    """
    batch = Batch.collate_fn(batch_list)

    if bucket_token_multiple is None and bucket_atom_multiple is None:
        return batch

    n_msa = batch.msa_number
    msa_depth = batch.msa_depth
    n_tokens = batch.token_length
    n_atoms = batch.atom_length

    bucketed_msa = (
        _ceil_to_multiple(msa_depth, bucket_msa_multiple)
        if bucket_msa_multiple
        else msa_depth
    )

    bucketed_tokens = (
        _ceil_to_multiple(n_tokens, bucket_token_multiple)
        if bucket_token_multiple
        else n_tokens
    )
    bucketed_atoms = (
        _ceil_to_multiple(n_atoms, bucket_atom_multiple)
        if bucket_atom_multiple
        else n_atoms
    )

    if (
        bucket_msa_multiple
        and bucketed_msa == n_msa
        and bucketed_tokens == n_tokens
        and bucketed_atoms == n_atoms
    ):
        return batch

    dummy = Batch.empty(
        n_msa=n_msa,
        msa_depth=bucketed_msa,
        n_tokens=bucketed_tokens,
        n_atoms=bucketed_atoms,
    )
    padded = Batch.collate_fn([batch, dummy])
    return padded[0 : batch.batch_size]


class PDBWeightedSampler(DistributedSampler):
    """Sampler that samples indices according to given weights."""

    def __init__(self, dataset: BioMolData, **kwargs: Any) -> None:
        super().__init__(dataset, **kwargs)
        self.edge_id_list = dataset.edge_id_list
        self.num_samples = len(dataset.edge_id_list)
        self._load_weights(dataset.config.sampler_config)

    def _load_weights(self, config: SamplerConfig) -> None:  # noqa: C901
        """Load weights from config and edge_id_list."""

        def _get_weight(edge_id: str) -> float:  # noqa: C901, PLR0911
            """Get weight for a given edge_id based on its type."""
            if "_" not in edge_id:
                return config.sole
            parse = set(re.findall(r"c([A-Z])", edge_id))

            # Antibody
            if parse == {"A"}:
                return config.antibody_antibody
            if parse <= {"A", "D", "R"} and "A" in parse:
                return config.antibody_nucleic_acid
            if parse <= {"A", "P"} and "A" in parse:
                return config.antibody_protein

            # Nucleic acid only
            if parse == {"D"}:
                return config.DNA_DNA
            if parse == {"R"}:
                return config.RNA_RNA
            if parse == {"D", "R"}:
                return config.DNA_RNA
            if parse <= {"D", "R", "N"} and "N" in parse:
                return config.NA_NA

            # Protein 관련
            if parse <= {"P", "D", "R", "N"} and "P" in parse and len(parse) > 1:
                return config.protein_nucleic_acid
            if parse == {"P"}:
                return config.protein_protein
            if parse <= {"P", "L"} and "P" in parse:
                return config.protein_ligand

            # Ligand
            if parse == {"L"}:
                return config.ligand_ligand

            # fallback
            return config.etc_interface

        initial_weights = np.array(
            [_get_weight(edge_id) for edge_id in self.edge_id_list],
            dtype=np.float32,
        )
        weights = initial_weights / initial_weights.sum()
        self.weights = torch.from_numpy(weights)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # weighted sampling으로 교체, 나머지 DDP 로직은 DistributedSampler에 위임
        all_indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=g,
        ).tolist()

        return iter(all_indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples


class BioMolData(torch.utils.data.Dataset):
    """Dataset for biomolecular complexes based on BioMolDB."""

    class BioMolConfig(BaseModel):
        """Configuration for BioMolData."""

        crop_config: CropConfig = CropConfig()
        msa_config: MSAConfig = MSAConfig()
        DB_config: BioMolDBConfig = BioMolDBConfig()
        tokenizer_config: TokenizerConfig = TokenizerConfig()
        sampler_config: SamplerConfig = SamplerConfig()

    def __init__(
        self,
        config: BioMolConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = Tokenizer(config=config.tokenizer_config)

        self._load_edge_to_cif_ids()
        self._crop_calls = 0
        self._crop_zero = 0
        self._crop_empty_selected_atoms = 0
        self._crop_report_every = 10
        self.chemcomp_to_fp_idx: dict[str, int] | None = None
        self.fp_unk_idx: int | None = None

        vocab_path = self.config.DB_config.fingerprint_vocab_path
        if vocab_path is not None:
            with vocab_path.open("r") as f:
                self.chemcomp_to_fp_idx = json.load(f)
                self.fp_unk_idx = int(self.chemcomp_to_fp_idx.get("UNK", 0))
            
    def _load_edge_to_cif_ids(self) -> None:
        # load edge_id to cif_ids mapping
        self.edge_id_to_bias: dict[str, list[str]] = {}
        with self.config.DB_config.edge_id_to_bias_path.open("r") as f:
            for _line in f:
                line = _line.strip()
                if line == "":
                    continue
                key1, key2, value = line.split("\t")
                edge_id = key1 if key2 == "None" else f"{key1}_{key2}"
                self.edge_id_to_bias[edge_id] = value.split(",")

        self.edge_id_list = list(self.edge_id_to_bias.keys())

    def __len__(self) -> int:
        """Return the number of edges in the dataset."""
        return len(self.edge_id_list)

    def _report_crop_stats(self) -> None:
        if self._crop_calls == 0:
            return
        if self._crop_calls % self._crop_report_every != 0:
            return

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else "main"

        zero_rate = self._crop_zero / self._crop_calls
        empty_sel_rate = self._crop_empty_selected_atoms / self._crop_calls

        print(
        f"[crop-monitor][worker={worker_id}] "
        f"calls={self._crop_calls} "
        f"zero_crop={self._crop_zero} ({zero_rate:.6f}) "
        f"empty_selected_atoms={self._crop_empty_selected_atoms}"# ({empty_sel_rate:.6f})"
        )

    def get_crop_indices(
        self,
        cifmol: CIFMolAttached,
        crop_indices: np.ndarray | None,
        bias: list[str],
        max_tokens: int,
        max_atoms: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Get crop indices for a given cifmol, either by cropping or using provided indices."""
        self._crop_calls += 1
        
        if crop_indices is not None:
            crop_indices = np.asarray(crop_indices, dtype=np.int64)
            chain_id_to_crop_indices = get_chain_crop_indices(
                cifmol=cifmol,
                crop_indices=crop_indices,
            )
            self._report_crop_stats()
            return crop_indices, chain_id_to_crop_indices
        match bias:
            case [chain_id]:
                selected_atoms = cifmol.chains.select(chain_id=chain_id).atoms
            case [chain_id1, chain_id2]:
                try:
                    selected_atoms = find_interface_residues(
                        cifmol,
                        chain_id1,
                        chain_id2,
                    ).atoms
                except NoInterfaceError:
                    chain_id = np.random.choice([chain_id1, chain_id2])
                    selected_atoms = cifmol.chains.select(chain_id=chain_id).atoms
            case _:
                msg = f"Unexpected chain_ids: {bias}"
                raise ValueError(msg)

        valid = np.all(np.isfinite(selected_atoms.xyz), axis=-1)
        selected_atoms = selected_atoms[valid]
        # center_xyz = selected_atoms.xyz[np.random.randint(len(selected_atoms))]
        if len(selected_atoms) == 0:
            # Fallback: use any finite atom in the structure.
            self._crop_empty_selected_atoms += 1
            all_atoms = cifmol.atoms
            all_valid = np.all(np.isfinite(all_atoms.xyz), axis=-1)
            selected_atoms = all_atoms[all_valid]

        if len(selected_atoms) == 0:
            # Extremely defensive fallback when no finite coordinates exist.
            center_xyz = np.zeros((3,), dtype=np.float32)
        else:
            center_xyz = selected_atoms.xyz[
                np.random.randint(len(selected_atoms))
            ]

        segment_size = np.random.randint(
            self.config.crop_config.min_segment_size,
            self.config.crop_config.max_segment_size + 1,
        )

        # crop_indices = crop_spatial_segment(
        #     cifmol,  # pyright: ignore[reportArgumentType]
        #     np.asarray(center_xyz),
        #     segment_size=segment_size,
        #     max_tokens=max_tokens,
        #     max_atoms=max_atoms,
        # )
        crop_indices = np.asarray(
            crop_spatial_segment(
                cifmol,  # pyright: ignore[reportArgumentType]
                np.asarray(center_xyz),
                segment_size=segment_size,
                max_tokens=max_tokens,
                max_atoms=max_atoms,
            ),
            dtype=np.int64,
         )

        if crop_indices.size == 0:
            # Guard empty crop: choose nearest finite atom's residue as fallback.
            self._crop_zero += 1
            atom_xyz = cifmol.atoms.xyz
            finite = np.all(np.isfinite(atom_xyz), axis=-1)
            if np.any(finite):
                finite_atom_idx = np.flatnonzero(finite)
                d = np.linalg.norm(
                    atom_xyz[finite] - np.asarray(center_xyz),
                    axis=-1,
                )
                nearest_atom_idx = int(finite_atom_idx[int(np.argmin(d))])
                fallback_res_idx = int(cifmol.index_table.atom_to_res[nearest_atom_idx])
            else:
                fallback_res_idx = int(np.random.randint(len(cifmol.residues)))
            crop_indices = np.asarray([fallback_res_idx], dtype=np.int64)


        chain_id_to_crop_indices = get_chain_crop_indices(
            cifmol=cifmol,
            crop_indices=crop_indices,
        )

        # crop_indices = cast("np.ndarray", crop_indices)
        crop_indices = np.asarray(
            cast("np.ndarray", crop_indices), dtype=np.int64
        )
        self._report_crop_stats()
        return crop_indices, chain_id_to_crop_indices  # pyright: ignore[reportPossiblyUnboundVariable]

    def __getitem__(self, idx: int) -> Batch:
        """Get a data sample by index."""
        edge_id = self.edge_id_list[idx]
        biases = self.edge_id_to_bias[edge_id]
        bias = random.choice(biases)
        pdb_id, assembly_id, model_id, alt_id = bias.split("_")[:4]
        bias = re.findall(r"\(([^)]+)\)", bias)

        item = self.get_item_by_id(
            pdb_id=pdb_id.lower(),
            assembly_id=assembly_id,
            model_id=model_id,
            alt_id=alt_id,
            bias=bias,
        )

        if item.sequence.token_type.shape[1] < 16:
            # too small, resample
            while True:
                idx = random.randint(0, len(self) - 1)
                edge_id = self.edge_id_list[idx]
                biases = self.edge_id_to_bias[edge_id]
                bias = random.choice(biases)
                pdb_id, assembly_id, model_id, alt_id = bias.split("_")[:4]
                bias = re.findall(r"\(([^)]+)\)", bias)
                item = self.get_item_by_id(
                    pdb_id=pdb_id.lower(),
                    assembly_id=assembly_id,
                    model_id=model_id,
                    alt_id=alt_id,
                    bias=bias,
                )
                if item.sequence.token_type.shape[1] >= 16:
                    break

        return item

    def get_item_by_id(
        self,
        pdb_id: str,
        assembly_id: str | None = None,
        model_id: str | None = None,
        alt_id: str | None = None,
        bias: list[str] | None = None,
        crop_indices: np.ndarray | None = None,
    ) -> Batch:
        """Get a data sample by cif_id."""
        cifmol = load_cifmol(
            db_path=self.config.DB_config.cif_db_path,
            pdb_id=pdb_id,
            assembly_id=assembly_id,
            model_id=model_id,
            alt_id=alt_id,
        )

        cifmol = remove_terminal_oxygen(cifmol)
        if bias is None:
            # randoml sample chain_id
            chain_id = np.random.choice(cifmol.chains.chain_id.value)
            bias = [chain_id]
        # TODO
        max_tokens = self.config.crop_config.residue_crop_length
        max_atoms = self.config.crop_config.atom_crop_length
        if crop_indices is None:
            while True:
                crop_indices, chain_id_to_crop_indices = self.get_crop_indices(
                    cifmol=cifmol,
                    crop_indices=crop_indices,
                    bias=bias,
                    max_tokens=max_tokens,
                    max_atoms=max_atoms,
                )
                cifmol_test: CIFMolAttached = cifmol.residues[crop_indices].extract()
                atom_to_token_idx_map, token_to_residue_idx_map = (
                    self.tokenizer.tokenize(
                        cifmol_test,
                    )
                )
                token_num = token_to_residue_idx_map.shape[0]
                if token_num <= self.config.crop_config.residue_crop_length:
                    break
                max_tokens = int(max_tokens * 0.9)
                max_atoms = int(max_atoms * 0.9)
                crop_indices = None
        else:
            crop_indices, chain_id_to_crop_indices = self.get_crop_indices(
                cifmol=cifmol,
                crop_indices=None,
                bias=bias,
                max_tokens=max_tokens,
                max_atoms=max_atoms,
            )
        cifmol: CIFMolAttached = cifmol.residues[crop_indices].extract()
        atom_to_token_idx_map, token_to_residue_idx_map = self.tokenizer.tokenize(
            cifmol,
        )

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
        
        token_type_override = None
        if self.chemcomp_to_fp_idx is not None:
            token_chem_comp = np.take(
                cifmol.residues.chem_comp_id.value,
                token_to_residue_idx_map,
            )
            unk = 0 if self.fp_unk_idx is None else self.fp_unk_idx
            token_type_override = np.array(
                [self.chemcomp_to_fp_idx.get(str(x), unk) for x in token_chem_comp],
                dtype=np.int64,
            )
        sequence, structure, msa_token, reference, scheme, chain = make_feature(
            cifmol=cifmol,
            msa=msa,
            atom_to_token_idx_map=atom_to_token_idx_map,
            token_to_residue_idx_map=token_to_residue_idx_map,
            token_type_override=token_type_override,
        )

        # ids : to make cif file from batch
        hetero = cifmol.residues.hetero
        atom_ids = cifmol.atoms.id
        chem_comp_ids = cifmol.residues.chem_comp_id

        return Batch(
            name=[f"{pdb_id}_{assembly_id}_{model_id}_{alt_id} with bias {bias}"],
            heteros=[hetero],
            atom_ids=[atom_ids],
            chem_comp_ids=[chem_comp_ids],
            sequence=sequence,
            structure=structure,
            msa=msa_token,
            reference=reference,
            scheme=scheme,
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
        bucket_msa_multiple: int | None = None,
        bucket_token_multiple: int | None = None,
        bucket_atom_multiple: int | None = None,
        **kwargs: object,
    ) -> DataLoader:
        """Create a distributed DataLoader with WeightedSampler."""
        # default distributed sampler
        sampler = PDBWeightedSampler(
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
            "collate_fn": functools.partial(
                _bucketed_collate,
                bucket_msa_multiple=bucket_msa_multiple,
                bucket_token_multiple=bucket_token_multiple,
                bucket_atom_multiple=bucket_atom_multiple,
            ),
        }
        params.update(kwargs)
        return DataLoader(self, **params)

from collections import Counter
from team_gm.constants import ResidueType

def audit_chem_comp_vs_mapped_restype(
    dataset: BioMolData,
    n_samples: int = 2000,
    seed: int = 0,
) -> None:
    rng = random.Random(seed)

    pair_counts: Counter[tuple[str, str]] = Counter()
    mismatch_counts: Counter[tuple[str, str]] = Counter()

    total_residues = 0
    mapped_diff_count = 0
    failed_samples = 0

    for _ in range(n_samples):
        try:
            idx = rng.randrange(len(dataset))
            edge_id = dataset.edge_id_list[idx]
            bias_str = rng.choice(dataset.edge_id_to_bias[edge_id])

            pdb_id, assembly_id, model_id, alt_id = bias_str.split("_")[:4]
            bias = re.findall(r"\(([^)]+)\)", bias_str)

            cifmol = load_cifmol(
                db_path=dataset.config.DB_config.cif_db_path,
                pdb_id=pdb_id.lower(),
                assembly_id=assembly_id,
                model_id=model_id,
                alt_id=alt_id,
            )
            cifmol = remove_terminal_oxygen(cifmol)

            max_tokens = dataset.config.crop_config.residue_crop_length
            max_atoms = dataset.config.crop_config.atom_crop_length
            crop_indices = None

            while True:
                crop_indices, _ = dataset.get_crop_indices(
                    cifmol=cifmol,
                    crop_indices=crop_indices,
                    bias=bias,
                    max_tokens=max_tokens,
                    max_atoms=max_atoms,
                )
                cropped = cifmol.residues[crop_indices].extract()
                _, token_to_residue_idx_map = dataset.tokenizer.tokenize(cropped)

                if token_to_residue_idx_map.shape[0] <= dataset.config.crop_config.residue_crop_length:
                    cifmol = cropped
                    break

                max_tokens = int(max_tokens * 0.9)
                max_atoms = int(max_atoms * 0.9)
                crop_indices = None

            chem_comp_ids = np.asarray(cifmol.residues.chem_comp_id.value, dtype=object)
            canonical_one_letter = np.asarray(cifmol.residues.one_letter_code_can.value, dtype=object)
            res_to_chain = np.asarray(cifmol.index_table.res_to_chain, dtype=np.int64)
            chain_entity_types = np.asarray(cifmol.chains.entity_type.value, dtype=object)

            for r in range(len(cifmol.residues)):
                chem = str(chem_comp_ids[r])
                entity_type = str(chain_entity_types[res_to_chain[r]])
                can1 = str(canonical_one_letter[r])

                mapped = ResidueType.resolve(
                    chem_comp_id=chem,
                    entity_type=entity_type,
                    canonical_one_letter=can1,
                ).name

                pair_counts[(chem, mapped)] += 1
                total_residues += 1

                if chem != mapped:
                    mapped_diff_count += 1
                    mismatch_counts[(chem, mapped)] += 1

        except Exception:
            failed_samples += 1

    print("\n=== chem_comp_id vs mapped canonical residue type ===")
    print(f"samples_checked={n_samples}")
    print(f"samples_failed={failed_samples}")
    print(f"total_cropped_residues={total_residues}")
    if total_residues > 0:
        print(
            f"mapped_different={mapped_diff_count} "
            f"({mapped_diff_count / total_residues:.6f})"
        )

    print("\nTop 30 mismatches: chem_comp_id -> mapped_restype")
    for (chem, mapped), cnt in mismatch_counts.most_common(30):
        print(f"{chem:>8} -> {mapped:<12} : {cnt}")

    print("\nTop 30 overall pairs: (chem_comp_id, mapped_restype)")
    for (chem, mapped), cnt in pair_counts.most_common(30):
        print(f"({chem:>8}, {mapped:<12}) : {cnt}")


if __name__ == "__main__":
    # test dataloader
    config = BioMolData.BioMolConfig(
        crop_config=CropConfig(
            residue_crop_length=512,
            atom_crop_length=4096,
            remain_invalid_tokens=False,
        ),
        msa_config=MSAConfig(
            n_samples=4,
            max_msa_depth=256,
            missing_policy="gap",
        ),
        DB_config=BioMolDBConfig(
            cif_db_path=Path(
                "/public_data/BioMolDB_20260224/cif_attached_train.lmdb",
            ),
            a3m_db_path=Path("/public_data/BioMolDB_20260224/a3m.lmdb"),
            edge_id_to_bias_path=(
                Path(
                    "/public_data/BioMolDB_20260224/train_edge_node.tsv",
                )
            ),
        ),
    )
    dataset = BioMolData(config)
    dataloader = dataset.create_ddp_dataloader(
        rank=0,
        world_size=1,
        shuffle=True,
        seed=42,
        drop_last=False,
        num_workers=0,
        bucket_token_multiple=128,
        bucket_atom_multiple=1024,
    )

    # wo dataloader
    # test data 1hcu
    cif_id = "3ni0_1_1_._(A_1)_(C_1)"
    data = dataset.get_item_by_id(pdb_id="3ni0", assembly_id="1", model_id="1", alt_id=".", bias=["A", "C"])
    print(data.chem_comp_ids)
    # for _ in range(10):
    #     batch = dataset[174]

    # for idx in range(len(dataset)):
    #     print(f"Testing dataset idx {idx}/{len(dataset)}")
    #     batch = dataset[idx]

    # test dataloader    for i, batch in enumerate(dataloader):

    # for i, batch in enumerate(dataloader):
    #     print(f"Batch {i}:")
    
    audit_chem_comp_vs_mapped_restype(
    dataset,
    n_samples=500,
    seed=0,
    )