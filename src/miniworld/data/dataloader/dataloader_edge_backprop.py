from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from biomol.core.container import FeatureContainer
from biomol.core.utils import load_bytes
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, DistributedSampler

from miniworld.data.crop import Cropper
from miniworld.data.features.features_biomol import (
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
    load_a3m,
    load_all_raw_data,
    load_cifmol,
)
from miniworld.data.mapping import AtomMapping, EntityMapping
from miniworld.data.msa import MSA, ComplexMSA
from miniworld.utils.structure import SE3_oper

if TYPE_CHECKING:
    from miniworld.data.mols import CIFMolAttached


class EdgeWeightConfig(BaseModel):
    """Configuration for edge weighting scheme."""

    PP_edge: float = 1.0  # protein-protein edge
    PN_edge: float = 1.0  # protein-nucleic acid edge
    PL_edge: float = 2 / 3  # protein-ligand edge
    NN_edge: float = 1.0  # nucleic acid-nucleic acid edge
    NL_edge: float = 2 / 3  # nucleic acid-ligand edge
    LL_edge: float = 0.0  # ligand-ligand edge

    # params
    eta: float = 0.9
    decay: float = 0.999
    temperature: float = 1.0
    init_score: float = 1.0
    init_freq: float = 0.0
    device: str = "cpu"
    use_freq: bool = True


class EdgeScoreStore:
    """Maintains per-edge statistics such as frequency and score.

    Score is updated via EMA. The sampler reads these values to compute
    sampling weights every epoch.
    """

    def __init__(
        self,
        config: EdgeWeightConfig,
        edges: list[str],
    ) -> None:
        self.edges = edges
        num_edges = len(edges)
        self.eta = config.eta
        self.decay = config.decay
        self.temperature = config.temperature
        self.device = config.device

        # upweight by edge type
        edge_type_weight = [self._edge_type_weight(e_id, config) for e_id in edges]
        self.edge_upweight = torch.tensor(
            [w for w, _ in edge_type_weight],
            dtype=torch.float32,
            device=config.device,
        )
        self.edge_type = [t for _, t in edge_type_weight]

        # EMA score for each edge
        # score: 1 means the model performs badly on this edge
        # score: 0 means the model performs well on this edge
        self.score = torch.full(
            (num_edges,),
            float(config.init_score),
            dtype=torch.float32,
            device=config.device,
        )

        # Visit frequency for each edge
        self.freq = torch.full(
            (num_edges,),
            float(config.init_freq),
            dtype=torch.float32,
            device=config.device,
        )

        self.use_freq = config.use_freq

    def _map_type(self, cluster_type: str) -> str:
        if cluster_type == "P":
            return "P"
        if cluster_type in {"D", "R", "N"}:
            return "N"
        if cluster_type in {"L", "B"}:
            return "L"
        return None

    def _edge_type_weight(
        self,
        e_id: str,
        config: EdgeWeightConfig,
    ) -> tuple[float, str]:
        cluster1, cluster2 = e_id.split("_")
        c1, c2 = cluster1[1], cluster2[1]
        c1, c2 = self._map_type(c1), self._map_type(c2)

        if c1 == "P" and c2 == "P":
            return config.PP_edge, "PP"
        if {c1, c2} == {"P", "N"}:
            return config.PN_edge, "PN"
        if {c1, c2} == {"P", "L"}:
            return config.PL_edge, "PL"
        if c1 == "N" and c2 == "N":
            return config.NN_edge, "NN"
        if {c1, c2} == {"N", "L"}:
            return config.NL_edge, "NL"
        return config.LL_edge, "LL"

    @torch.no_grad()
    def update(self, edge_idx: torch.Tensor, loss: torch.Tensor) -> None:
        """Update edge scores based on observed losses."""
        if edge_idx.shape[0] == 0:
            # for contiguous cropping, may have no edges in the crop
            return
        # assume batch size is 1
        edge_idx = edge_idx.to(self.device)
        loss = loss.to(self.device)
        edge_idx = edge_idx.squeeze(0)
        loss = loss.squeeze(0).repeat_interleave(edge_idx.shape[0])

        # deduplicate indices
        unique_idx, inv = torch.unique(edge_idx, return_inverse=True)

        # visit_incr[i] = how many times unique_idx[i] appeared in edge_idx
        visit_incr = torch.bincount(inv, minlength=unique_idx.numel()).float()
        self.freq.index_add_(0, unique_idx, visit_incr)

        # decay scores for not-updated edges
        not_updated = torch.ones_like(self.score, dtype=torch.bool)
        not_updated[unique_idx] = False
        self.score[not_updated] = 1 - (1 - self.score[not_updated]) * self.decay

        # aggregate loss per unique edge (mean)
        loss_sum = torch.bincount(inv, weights=loss, minlength=unique_idx.numel())
        mean_loss = loss_sum / torch.clamp_min(visit_incr, 1.0)

        new = 1.0 - torch.exp(-mean_loss / self.temperature)
        old_s = self.score[unique_idx]
        self.score[unique_idx] = (1.0 - self.eta) * old_s + self.eta * new

        # clamp only on updated + decayed entries
        self.score.clamp_(0.0, 1.0)

    @torch.no_grad()
    def get_weights(self, min_prob: float = 0.0) -> torch.Tensor:
        """Get sampling weights for all edges."""
        if self.use_freq:
            # v1: inverse freq weighting
            priority = self.score / torch.sqrt(self.freq + 1.0)
        else:
            # v2: only score weighting
            priority = self.score
        priority = priority * self.edge_upweight
        priority = torch.clamp(priority, min=min_prob)

        total = priority.sum()
        if total <= 0:
            # fallback: uniform sampling
            return torch.full_like(priority, 1.0 / priority.numel())

        return priority / total


class AdaptiveEdgeSampler(DistributedSampler):
    """Distributed sampler that performs weighted sampling based on per-edge scores from EdgeScoreStore."""

    class Config(BaseModel):
        """Configuration for AdaptiveEdgeSampler."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        dataset: BioMolData
        stats: EdgeScoreStore | None = None
        num_replicas: int = 1
        rank: int = 0
        shuffle: bool = True
        seed: int = 0
        drop_last: bool = False
        device: str = "cpu"

        temperature: float = 1.0
        min_prob: float = 1e-6

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
        self.stats: EdgeScoreStore = config.stats
        self.device = config.device
        self.temperature = config.temperature
        self.min_prob = config.min_prob
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self) -> iter[int]:
        """Return a generator that yields sampled indices for this rank."""
        # Optional per-epoch RNG for reproducibility across nodes

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        num_samples = self.num_samples  # per-rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        for _ in range(num_samples):
            valid_weights = self.stats.get_weights(min_prob=self.min_prob)
            weights = torch.zeros_like(valid_weights)
            weights[indices] = valid_weights[indices]
            weights = weights / weights.sum()
            cdf = torch.cumsum(weights, dim=0)
            cdf[-1] = 1.0  # To prevent possible numerical issues
            r = torch.rand(1, device=self.device)
            idx = torch.searchsorted(cdf, r).item()
            yield idx


class CropConfig(BaseModel):
    """Configuration for cropping strategy."""

    contiguous_prob: float = 0.5
    spatial_prob: float = 0.5
    interface_prob: float = 0.0
    crop_length: int = 384


class MSAConfig(BaseModel):
    """Configuration for MSA sampling."""

    n_samples: int = 4
    max_msa_depth: int = 512


class BioMolDBConfig(BaseModel):
    """Configuration for BioMolDB paths."""

    cif_db_path: Path = Path("cif_lmdb")
    a3m_db_path: Path = Path("a3m_lmdb")
    edge_id_to_cif_ids_path: Path = Path("edge_id_to_cif_ids.tsv")
    load_all_msa: bool = False


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

        # gen stats
        self.stats = EdgeScoreStore(
            config=self.config.edge_weight_config,
            edges=self.edge_id_list,
        )

    def __len__(self) -> int:
        """Return the number of edges in the dataset."""
        return len(self.edge_id_list)

    def load_msa(
        self,
        cifmol: CIFMolAttached,
        chain_id_to_crop_indices: dict[str, np.ndarray],
    ) -> ComplexMSA:
        """Load and crop MSAs for each chain in the cropped cifmol."""
        msa_list: list[MSA] = []
        total_length = 0
        for chain_id, crop_indices in chain_id_to_crop_indices.items():
            if len(crop_indices) == 0:
                continue
            total_length += len(crop_indices)
            seq_id = cifmol.chains[cifmol.chains.chain_id == chain_id].seq_id[0].value
            if self.msa_bytes_dict is not None:
                msa_bytes = self.msa_bytes_dict.get(seq_id, None)
                if msa_bytes is None:
                    break
                msa_container = load_bytes(bytes(msa_bytes))["msa_container"]
                msa_residue_container = msa_container["residue_container"]
                msa_chain_container = msa_container["chain_container"]
                msa_residue_container = FeatureContainer.from_dict(
                    msa_residue_container,
                )
                msa_chain_container = FeatureContainer.from_dict(msa_chain_container)
                msa = MSA(
                    seq_id=seq_id,
                    msa_residue_container=msa_residue_container,
                    msa_chain_container=msa_chain_container,
                )
            else:
                msa = load_a3m(
                    key=seq_id,
                    env_path=self.config.DB_config.a3m_db_path,
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

    def __getitem__(self, idx: int) -> Batch:
        """Get a data sample by index."""
        edge_id = self.edge_id_list[idx]

        return self.get_item_by_id(edge_id=edge_id)

    def get_item_by_id(self, edge_id: str) -> Batch:  # noqa: PLR0915
        """Get a data sample by edge_id."""
        cif_ids = self.edge_id_to_cif_ids[edge_id]
        cif_id = random.choice(cif_ids)

        cifmol = load_cifmol(db_path=self.config.DB_config.cif_db_path, cif_id=cif_id)

        chain_id1, chain_id2 = re.findall(r"\([^)]*\)|[^_]+", cif_id)[-2:]
        chain_id1 = chain_id1.strip("()")
        chain_id2 = chain_id2.strip("()")
        chain_bias = random.choice([chain_id1, chain_id2])
        interface_bias = (chain_id1, chain_id2)

        # Crop
        crop_length = self.config.crop_config.crop_length
        while crop_length > 0:
            crop_indices, chain_id_to_crop_indices = self.cropper.crop(
                cifmol=cifmol,
                crop_length=crop_length,
                chain_bias=chain_bias,
                interface_bias=interface_bias,
            )
            if (
                cifmol.residues[crop_indices].atoms.element.shape[0]
                < self.config.crop_config.crop_length * 12
            ):
                break
            crop_length = int(crop_length * 0.8)  # reduce crop length if too large

        cifmol: CIFMolAttached = cifmol.residues[crop_indices].extract()
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
        msa = self.load_msa(
            cifmol=cifmol,
            chain_id_to_crop_indices=chain_id_to_crop_indices,
        )

        # sample MSAs
        msa_profile = msa.profile
        msa_deletion_mean = msa.deletion_mean
        msa_sequence_sampled = []
        msa_has_deletion_sampled = []
        msa_deletion_value_sampled = []
        for _ in range(self.config.msa_config.n_samples):
            _, sampled_sequence, sampled_has_deletion, sampled_deletion_value = (
                msa.sample(
                    self.config.msa_config.max_msa_depth,
                )
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
        cropped_len = len(crop_indices)

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

        residue_type = msa_sequence_sampled[0, 0]
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
            _ref_pos = (
                _ref_pos - _ref_pos.mean(axis=0)
            ) @ R + T  # random SE(3) operation
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
            residue_type=torch.from_numpy(residue_type.astype(np.int64)),
        )
        atom_pos = cifmol.atoms.xyz.value
        atom_pos_mask = np.isfinite(atom_pos).all(axis=1)
        atom_mask = np.ones_like(atom_pos_mask, dtype=bool)

        # centering atom_pos
        valid_pos = atom_pos[atom_pos_mask]          # (N_valid, 3)

        mean_vector = valid_pos.mean(axis=0, keepdims=True)
        atom_pos = atom_pos - mean_vector

        atom_pos = np.where(atom_pos_mask[:, None], atom_pos, 0.0)

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
            crop_indices=torch.from_numpy(crop_indices.astype(np.int64)),
            residue_idx=torch.from_numpy(residue_idx.astype(np.int64)),
            residue_idx_mono=torch.from_numpy(residue_idx_mono.astype(np.int64)),
            residue_asym_id=torch.from_numpy(residue_asym_id.astype(np.int64)),
            residue_entity_id=torch.from_numpy(residue_entity_id.astype(np.int64)),
            residue_sym_id=torch.from_numpy(residue_sym_id.astype(np.int64)),
            atom_to_residue_idx_map=torch.from_numpy(
                atom_to_residue_idx_map.astype(np.int64),
            ),
            edge_index=torch.from_numpy(np.array(edge_index, dtype=np.int64)),
        )
        msa = MSAFeatures.from_sample(
            aligned_sequences=torch.from_numpy(msa_sequence_sampled).to(torch.int8),
            has_deletion=torch.from_numpy(msa_has_deletion_sampled).to(torch.int8),
            deletion_value=torch.from_numpy(msa_deletion_value_sampled).half(),
            profile=torch.from_numpy(msa_profile).half(),
            deletion_mean=torch.from_numpy(msa_deletion_mean).half(),
        )

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
            msa=msa,
            chain=chain,
        )

    def create_ddp_dataloader(
        self,
        rank: int,
        drop_last: bool = False,
        use_adaptive_sampler: bool = True,  # train only
        **kwargs: object,
    ) -> DataLoader:
        """Create a distributed DataLoader with AdaptiveEdgeSampler."""
        if use_adaptive_sampler:
            sampler = AdaptiveEdgeSampler(
                AdaptiveEdgeSampler.Config(
                    dataset=self,
                    num_replicas=kwargs.get("world_size", 1),
                    stats=self.stats,
                    rank=rank,
                    shuffle=kwargs.get("shuffle", True),
                    seed=kwargs.get("seed", 0),
                    drop_last=drop_last,
                ),
            )
        else:
            # default distributed sampler
            sampler = DistributedSampler(
                dataset=self,
                num_replicas=kwargs.get("world_size", 1),
                rank=rank,
                shuffle=kwargs.get("shuffle", False),
                seed=kwargs.get("seed", 0),
                drop_last=drop_last,
            )

        kwargs.pop("shuffle", None)
        kwargs.pop("world_size", None)
        kwargs.update({"sampler": sampler})

        params = {
            "shuffle": False,  # leave False when using a sampler
            "drop_last": False,  # override to True for train
            "num_workers": kwargs.get("num_workers", 0),
            "pin_memory": False,
            "multiprocessing_context": (
                "spawn" if kwargs.get("num_workers", 0) > 0 else None
            ),
            "collate_fn": Batch.collate_fn,
        }
        params.update(kwargs)
        return DataLoader(self, **params)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    crop_config = CropConfig(
        contiguous_prob=0.0,
        spatial_prob=1.0,
        interface_prob=0.0,
        crop_length=384,
    )
    msa_config = MSAConfig(
        n_samples=4,
        max_msa_depth=512,
    )
    DB_config = BioMolDBConfig(
        cif_db_path=Path(
            "/home/psk6950/data/BioMolDBv2_2024Oct21/cif_20210930_res9.lmdb",
        ),
        a3m_db_path=Path("/home/psk6950/data/BioMolDBv2_2024Oct21/slim_a3m.lmdb"),
        edge_id_to_cif_ids_path=Path(
            "/home/psk6950/data/BioMolDBv2_2024Oct21/metadata/graph_split_20210930_res9/train_edges.tsv",
        ),
    )
    config = BioMolData.BioMolConfig(
        crop_config=crop_config,
        msa_config=msa_config,
        DB_config=DB_config,
    )
    set_seed(42)
    dataset = BioMolData(config=config)
    dataset[0]

    for ii in range(len(dataset)):
        dataset[ii]
        try:
            dataset[ii]
        except Exception as e:  # noqa: BLE001
            print(f"Error at index {ii}: {e}")  # noqa: T201
            continue
        print(f"Processed {ii} samples.")  # noqa: T201
