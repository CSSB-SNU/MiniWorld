import torch
from math import pi
import numpy as np
import random
import pickle
from datetime import datetime as date
from joblib import Parallel, delayed
import json
from typing import Literal

from torch.utils.data import DataLoader, DistributedSampler
from abc import ABC, abstractmethod
from pathlib import PosixPath, Path
from pydantic import BaseModel, ConfigDict


from team_gm.data.features_BioMol import (
    Batch,
    SequenceFeatures,
    StructureFeatures,
    ReferenceFeatures,
    SchemeFeatures,
    MSAFeatures,
    ChainFeatures,
)
from BioMol.BioMol import BioMol
from BioMol import DB_PATH, SEQ_TO_HASH_PATH
from BioMol.constant.chemical import AA2num


with open(SEQ_TO_HASH_PATH, "rb") as f:
    seq_to_hash = pickle.load(f)  # dict of {seq: seq_hash}

seq_to_cluster_path = f"{DB_PATH}/cluster/seq_clust/seq_to_cluster.pkl"
with open(seq_to_cluster_path, "rb") as f:
    seq_to_cluster = pickle.load(f)  # dict of {seq: seq_cluster}

chain_ID_to_cluster_path = f"{DB_PATH}/cluster/seq_clust/chain_ID_to_cluster.pkl"
with open(chain_ID_to_cluster_path, "rb") as f:
    chain_ID_to_cluster = pickle.load(f)  # dict of {chain_ID: seq_cluster}

hash_to_seq = {v: k for k, v in seq_to_hash.items()}  # dict of {seq_hash: seq}


def to_mmcif(
    batch: Batch,
    denoised_atom_pos: torch.Tensor,
    true_mmcif_path: PosixPath,
    denoised_mmcif_path: PosixPath,
    mol_types: list[str] = ["protein"],
):
    pdb_id, assembly_id, model_id, alt_id = batch.name[0].split("_")
    crop_indices = batch.scheme.crop_indices[0]
    biomol = BioMol(pdb_ID=pdb_id, mol_types=mol_types)
    biomol.choose(assembly_id, model_id, alt_id)
    biomol.crop(crop_indices=crop_indices.cpu(), crop_MSA=True)

    if true_mmcif_path is not None:
        biomol.structure.to_mmcif(
            true_mmcif_path,
        )

    biomol.structure.atom_tensor[:, 5:8] = denoised_atom_pos
    biomol.structure.to_mmcif(
        denoised_mmcif_path,
    )


class BaseData(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

    def create_dataloader(self, **kwargs) -> DataLoader:
        params = {
            "shuffle": False,  # leave False when using a sampler
            "drop_last": False,  # override to True for train
            "num_workers": 4,
            "pin_memory": True,
            "collate_fn": Batch.collate_fn,
        }
        params.update(kwargs)
        return DataLoader(self, **params)

    def create_ddp_dataloader(
        self,
        rank: int,
        world_size: int,
        drop_last: bool = False,
        **kwargs,
    ):
        sampler = DistributedSampler(
            self,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=drop_last,
        )
        # ensure shuffle=False since sampler does the shuffling
        kwargs.pop("shuffle", None)
        kwargs.update({"sampler": sampler})
        return self.create_dataloader(**kwargs)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Batch:
        pass

class MolTypeConfig(BaseModel):
    protein: bool = True
    nucleic_acid: bool = True
    ligand: bool = True

    @property
    def mol_types(self) -> list[str]:
        mol_types = []
        if self.protein:
            mol_types.append("protein")
        if self.nucleic_acid:
            mol_types.append("nucleic_acid")
        if self.ligand:
            mol_types.append("ligand")
        return mol_types

class BioMolPreProcessing:
    class MetaConfig(BaseModel):
        graph_cluster_metadata_path: PosixPath | str = "graph_cluster_metadata.json"
        chainID_to_cluster_path: PosixPath | str = "chainID_to_cluster.pkl"
        node_score_path: PosixPath | str = "node_score.pkl"
        edge_score_path: PosixPath | str = "edge_score.pkl"
        ideal_ligand_path: PosixPath | str = "ideal_ligand.pkl"
        unique_graphs_path: PosixPath | str = "unique_graphs.pkl"
        metadata_path: PosixPath | str

    class PipelineConfig(BaseModel):
        graph_hash_path: PosixPath | str
        thread_num: int = 8
        filter_date: str = "2024-10-21"
        filter_mask_ratio: float = 0.5
        filter_resolution: float = 9.0
        filter_chain_num: int = 40
        filtered_item_path: PosixPath | str | None = None
        data_tmp_dir: PosixPath | str = "./data_tmp/"

    class Config(BaseModel):
        meta: "BioMolPreProcessing.MetaConfig"
        pipeline: "BioMolPreProcessing.PipelineConfig"
        mol_types: MolTypeConfig

    def __init__(self, config: Config):
        self.config = config

        if isinstance(self.config.pipeline.data_tmp_dir, str):
            self.config.pipeline.data_tmp_dir = Path(self.config.pipeline.data_tmp_dir)
        if not self.config.pipeline.data_tmp_dir.exists():
            self.config.pipeline.data_tmp_dir.mkdir(parents=True, exist_ok=True)

        self.chainID_to_cluster = self._load_pickle_file(
            self.config.meta.chainID_to_cluster_path
        )
        self.mol_types = self.config.mol_types.mol_types  # list of mol types
        self._load_items_and_filter(self.config.pipeline.filtered_item_path)
        self._load_node_edge_score()
        self.ideal_ligand = self._load_pickle_file(self.config.meta.ideal_ligand_path)
        self.ideal_ligand = np.array(self.ideal_ligand)

    def _read_large_text_file(self, file_path: PosixPath | str) -> list[str]:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")

        def _compute_chunks(path: Path, n_chunks: int):
            """
            Return a list of (start, end) byte-offsets for splitting `path`.
            """
            file_size = path.stat().st_size
            chunk_size = file_size // n_chunks
            boundaries = []
            for i in range(n_chunks):
                start = i * chunk_size
                end = file_size if i == n_chunks - 1 else (i + 1) * chunk_size
                boundaries.append((start, end))
            return boundaries

        def read_chunk(path: Path, start: int, end: int):
            """
            Open `path`, seek to `start`, skip a partial first line if needed,
            then read lines until byte-offset ≥ end.
            """
            lines = []
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                f.seek(start)
                # Skip a partial first line when start > 0
                if start > 0:
                    f.readline()
                pos = f.tell()
                while pos < end:
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line.rstrip("\n"))
                    pos = f.tell()
            return lines

        n_jobs = self.config.pipeline.thread_num
        chunks = _compute_chunks(file_path, n_jobs)

        results = Parallel(n_jobs=n_jobs)(
            delayed(read_chunk)(file_path, start, end) for start, end in chunks
        )

        # Flatten into one list of lines
        all_lines = [line for chunk in results for line in chunk]
        return all_lines

    def _load_pickle_file(self, file_path: PosixPath | str):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def _load_json_file(self, file_path: PosixPath | str):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with open(file_path) as f:
            return json.load(f)

    def _load_items_and_filter(self, filtered_item_path: PosixPath | str = None):
        if isinstance(filtered_item_path, str):
            filtered_item_path = Path(filtered_item_path)
        if filtered_item_path is not None and filtered_item_path.exists():
            data = self._load_pickle_file(filtered_item_path)
            self.items = data["items"]
            self.graph_hashes = data["graph_hashes"]
            self.pdb_ids = data["pdb_ids"]
            return
        else:
            if self.config.pipeline.filter_date is not None:
                date_cutoff = date.strptime(
                    self.config.pipeline.filter_date, "%Y-%m-%d"
                )
            else:
                date_cutoff = date.strptime("2099-01-01", "%Y-%m-%d")

            if self.config.pipeline.filter_resolution is not None:
                resolution_cutoff = float(self.config.pipeline.filter_resolution)
            else:
                resolution_cutoff = float("inf")

            if self.config.pipeline.filter_chain_num is not None:
                chain_num_cutoff = self.config.pipeline.filter_chain_num
            else:
                chain_num_cutoff = float("inf")

            graph_cluster_metadata = self._load_json_file(
                self.config.meta.graph_cluster_metadata_path
            )  # cluster : {hash : [IDs]}

            graph_hash = self._read_large_text_file(
                self.config.pipeline.graph_hash_path
            )

            hash_to_graph = self._load_pickle_file(self.config.meta.unique_graphs_path)
            # filter graph_hash by chain_num
            # and if graph has no edges but has one more nodes, remove it
            filtered_graph_hash = []
            for _hash in graph_hash:
                if int(_hash) in hash_to_graph:
                    graph = hash_to_graph[int(_hash)]
                    if len(graph.nodes) > 1 and len(graph.edges) == 0:
                        continue
                    if len(graph.nodes) <= chain_num_cutoff:
                        filtered_graph_hash.append(_hash)

        items = {_hash: graph_cluster_metadata[_hash] for _hash in filtered_graph_hash}

        metadata = self._load_pickle_file(self.config.meta.metadata_path)

        # filter metadata
        filtered_IDs = []
        for ID, md in metadata.items():
            deposition_str, resolution_str = md[0], md[1]
            deposition = date.strptime(deposition_str, "%Y-%m-%d")
            resolution_str = (
                float("inf") if resolution_str == "None" else float(resolution_str)
            )
            if deposition <= date_cutoff and resolution_str <= resolution_cutoff:
                filtered_IDs.append(ID)
        filtered_IDs = set(filtered_IDs)

        def chunked_items(items, n_chunks):
            items_list = list(items.items())
            q, r = divmod(len(items_list), n_chunks)
            chunks = []
            start = 0
            for i in range(n_chunks):
                size = q + (1 if i < r else 0)
                chunks.append(items_list[start : start + size])
                start += size
            return chunks

        def _filter_chunk(chunk):
            out = []
            # for gh, full_ids in chunk:
            #     matches = [fid for fid in full_ids if fid[:4] in filtered_IDs]
            #     if matches:
            #         out.append((gh, matches, [m[:4] for m in matches]))
            # return out

            for gh, hash_dict in chunk:
                filtered_hash_dict = {}
                filtered_pdb_ids = []
                for _hash, full_ids in hash_dict.items():
                    filtered_ids = []
                    for full_id in full_ids:
                        pdb_ID, assembly_ID, model_ID, alt_ID = full_id.split("_")
                        if pdb_ID not in filtered_IDs:
                            continue

                        biomol = BioMol(pdb_ID=pdb_ID)
                        biomol.choose(assembly_ID, model_ID, alt_ID)
                        biomol.structure.filter_by_type(self.mol_types)

                        valid_residue_indices = torch.where((biomol.structure.residue_tensor[:, 4] == 1) 
                                                            & (biomol.structure.residue_tensor[:, 0] != AA2num["X"]))[0]
                        valid_residue_num = valid_residue_indices.size(0)
                        if valid_residue_num < 5 :
                            continue
                        filtered_ids.append(full_id)
                    if filtered_ids:
                        filtered_hash_dict[_hash] = filtered_ids
                        filtered_pdb_ids.extend([m[:4] for m in filtered_ids])
                if filtered_hash_dict:
                    out.append((gh, filtered_hash_dict, filtered_pdb_ids))
            return out

        n = self.config.pipeline.thread_num

        chunks = chunked_items(items, n)
        results = Parallel(n_jobs=n, verbose=10)(
            delayed(_filter_chunk)(chunk) for chunk in chunks
        )

        # collect only the passed ones
        filtered_items = {}
        filtered_pdb_ids = []
        for out_list in results:
            for out in out_list:
                gh, filtered_hash_dict, pdb_ids = out
                filtered_items[gh] = filtered_hash_dict
                filtered_pdb_ids.extend(pdb_ids)

        self.graph_hashes = list(filtered_items.keys())  # list of graph hashes
        self.graph_hashes.sort()
        self.items = [filtered_items[_hash] for _hash in self.graph_hashes]
        self.pdb_ids = list(set(filtered_pdb_ids))  # list of unique PDB IDs

        # save the items and etc at data_tmp_dir/filtered_items.pkl
        with open(filtered_item_path, "wb") as f:
            pickle.dump(
                {
                    "items": self.items,
                    "graph_hashes": self.graph_hashes,
                    "pdb_ids": self.pdb_ids,
                },
                f,
            )

    def _load_node_edge_score(self):
        self.node_score = self._load_pickle_file(
            self.config.meta.node_score_path
        )  # dict of {seq_hash: node_score}
        self.edge_score = self._load_pickle_file(
            self.config.meta.edge_score_path
        )  # dict of {(seq_hash, seq_hash): edge_score}


class BioMolSampler(DistributedSampler):
    class Config(BaseModel):
        # 1) let Pydantic accept any Python class as a field
        model_config = ConfigDict(arbitrary_types_allowed=True)

        # 2) annotate with the real class (not a primitive)
        dataset: "BioMolData"
        num_replicas: int = 1
        rank: int = 0
        shuffle: bool = True
        seed: int = 0
        drop_last: bool = False

    def __init__(self, config: Config):
        super().__init__(
            dataset=config.dataset,
            num_replicas=config.num_replicas,
            rank=config.rank,
            shuffle=config.shuffle,
            seed=config.seed,
            drop_last=config.drop_last,
        )
        self.dataset = config.dataset

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class CropConfig(BaseModel):
    contiguous_prob: float = 0.2
    spatial_prob: float = 0.4
    interface_prob: float = 0.4
    crop_length: int = 384
    level: Literal["residue", "atom"] = "residue"

    @property
    def crop_method_prob(self) -> list[float]:
        return [
            self.contiguous_prob,
            self.spatial_prob,
            self.interface_prob,
        ]




class MSAConfig(BaseModel):
    n_samples: int = 4
    max_msa_depth: int = 512


class BioMolData(BaseData):
    class BioMolConfig(BaseModel):
        crop_config: CropConfig = CropConfig()
        msa_config: MSAConfig = MSAConfig()
        data_preprocessing_config: BioMolPreProcessing.Config = None
        mol_types: MolTypeConfig = MolTypeConfig()

    def __init__(
        self,
        config: BioMolConfig,
    ):
        super().__init__(transform=None)
        self.config = config
        self.preprocessing = BioMolPreProcessing(config.data_preprocessing_config)
        self.crop_method_prob = [
            config.crop_config.contiguous_prob,
            config.crop_config.spatial_prob,
            config.crop_config.interface_prob,
        ]
        self.crop_length = config.crop_config.crop_length
        self.crop_level = config.crop_config.level
        self.chainID_to_cluster = self.preprocessing.chainID_to_cluster
        self.mol_types = config.mol_types.mol_types  # list of mol types

    def __len__(self, level: str = "graph"):
        if level == "graph":
            return len(self.preprocessing.items)
        elif level == "pdb":
            return len(self.preprocessing.pdb_ids)
        elif level == "item":
            return sum(len(v) for v in self.preprocessing.items.values())

    def _get_prob_from_score(self, score_dict: dict) -> dict:
        """
        Convert scores to probabilities using
            prob ∝ 1 / score
        """
        min_score = min(score_dict.values())
        if min_score <= 0:
            raise ValueError("Scores must be positive to compute probabilities.")
        prob_dict = {k: 1 / v for k, v in score_dict.items()}
        total_prob = sum(prob_dict.values())
        return {k: v / total_prob for k, v in prob_dict.items()}

    def SE3_oper(self, length, noise_level=5.0):
        u1, u2, u3 = torch.rand(3, length)
        r1, r2 = torch.sqrt(1 - u1), torch.sqrt(u1)
        th1, th2 = 2 * pi * u2, 2 * pi * u3
        qw = r2 * torch.cos(th2)
        qx = r1 * torch.sin(th1)
        qy = r1 * torch.cos(th1)
        qz = r2 * torch.sin(th2)
        ww, xx, yy, zz = qw * qw, qx * qx, qy * qy, qz * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz

        R = torch.empty(length, 3, 3)
        R[:, 0, 0] = ww + xx - yy - zz
        R[:, 0, 1] = 2 * (xy - wz)
        R[:, 0, 2] = 2 * (xz + wy)
        R[:, 1, 0] = 2 * (xy + wz)
        R[:, 1, 1] = ww - xx + yy - zz
        R[:, 1, 2] = 2 * (yz - wx)
        R[:, 2, 0] = 2 * (xz - wy)
        R[:, 2, 1] = 2 * (yz + wx)
        R[:, 2, 2] = ww - xx - yy + zz

        # random gaussian noise
        noise = torch.randn(length, 3) * noise_level
        return R, noise

    def __getitem__(self, idx: int) -> Batch:
        hash_dict = self.preprocessing.items[idx]

        # uniformly sample a hash level graph from the list
        hash_list = list(hash_dict.keys())
        if len(hash_list) == 0:
            raise ValueError(f"No valid hash found for index {idx}.")
        hash_key = random.choice(hash_list)
        IDs = hash_dict[hash_key]  # list of IDs in the hash
        if len(IDs) == 0:
            raise ValueError(f"No valid IDs found for hash {hash_key} at index {idx}.")

        # uniformly sample a PDB ID from the list

        # 20250601 psk
        # there are invalid IDs in the dataset, so we need to catch the error

        ID = random.choice(IDs)

        pdb_ID, assembly_ID, model_ID, alt_ID = ID.split("_")

        biomol = BioMol(pdb_ID=pdb_ID)  # warning, mol type is not filtered out.
        biomol.choose(assembly_ID, model_ID, alt_ID)

        chains = list(biomol.structure.residue_chain_break.keys())
        ID = biomol.ID
        chainID_to_cluster = {
            chain: str(self.chainID_to_cluster[ID + "_" + chain.split("_")[0]])
            for chain in chains
        }
        contact_edges = biomol.structure.contact_graph.graphs[
            (assembly_ID, model_ID, alt_ID)
        ]["edges"]  # list of tuples (chain_idx1, chain_idx2)
        contact_edges = {
            (chains[edge[0]], chains[edge[1]]): tuple(
                sorted(
                    (
                        chainID_to_cluster[chains[edge[0]]],
                        chainID_to_cluster[chains[edge[1]]],
                    )
                )
            )
            for edge in contact_edges
        }
        node_scores = {
            chain_ID: self.preprocessing.node_score[seq_cluster]
            for chain_ID, seq_cluster in chainID_to_cluster.items()
        }
        cluster_pairs = [
            (
                int(chainID_to_cluster[chain_pair[0]]),
                int(chainID_to_cluster[chain_pair[1]]),
            )
            for chain_pair in contact_edges.keys()
        ]
        # sort cluster_pairs
        cluster_pairs = [str(tuple(sorted(pair))) for pair in cluster_pairs]
        edge_scores = {
            chain_pair: self.preprocessing.edge_score[cluster_pair]
            for chain_pair, cluster_pair in zip(contact_edges.keys(), cluster_pairs)
        }

        # filter out chain by type
        biomol.structure.filter_by_type(self.mol_types)
        filtered_chain_IDs = biomol.structure.residue_chain_break.keys()
        if len(filtered_chain_IDs) == 0:
            raise ValueError(
                "After filtering by mol type, no chains left in the biomolecule. You have to filter train/valid items correspondingly."
            )
        def is_valid_chain(chain_id: str, min_residue_length=1) -> bool:
            chain_start, chain_end = biomol.structure.residue_chain_break[chain_id]
            residue_tensor = biomol.structure.residue_tensor[chain_start : chain_end + 1]
            valid_residue_indices = torch.where((residue_tensor[:, 4] == 1)
                                                & (residue_tensor[:, 0] != AA2num["X"]))[0]
            return valid_residue_indices.size(0) >= min_residue_length

        filtered_chain_IDs = [
            chain for chain in filtered_chain_IDs if is_valid_chain(chain)
        ]
        for chain in list(node_scores.keys()):
            if chain not in filtered_chain_IDs:
                del node_scores[chain]
        for edge in list(edge_scores.keys()):
            if edge[0] not in filtered_chain_IDs or edge[1] not in filtered_chain_IDs:
                del edge_scores[edge]

        try:
            node_probs = self._get_prob_from_score(node_scores)
        except:
            print(f"idx : {idx}")
            print(f"biomol structure {biomol.structure}")
            breakpoint()

        # sample node bias and edge bias
        chain_bias = random.choices(
            list(node_probs.keys()),
            weights=list(node_probs.values()),
            k=1,
        )[0]
        if len(edge_scores) > 0:
            edge_probs = self._get_prob_from_score(edge_scores)

            interface_bias = random.choices(
                list(edge_probs.keys()),
                weights=list(edge_probs.values()),
                k=1,
            )[0]
            # shuffle edge_bias
            interface_bias = tuple(random.sample(interface_bias, len(interface_bias)))
        else:
            interface_bias = None

        crop_length = self.crop_length

        while crop_length > 0:
            crop_indices, seq_hash_to_crop_indices = biomol.get_crop_indices(
                chain_bias=chain_bias,
                interface_bias=interface_bias,
                contiguous_crop_weight=self.crop_method_prob[0],
                spatial_crop_weight=self.crop_method_prob[1],
                interface_crop_weight=self.crop_method_prob[2],
                crop_length=crop_length,
                level=self.crop_level,
            )
            atom_to_residue_idx_map = biomol.structure.atom_tensor[:, 2]
            mask = torch.isin(atom_to_residue_idx_map, crop_indices)

            if (
                mask.sum().item() < self.crop_length * 12
            ):  # 12 is a heuristic for atom length
                biomol.crop(crop_indices=crop_indices, crop_MSA=True)
                break
            else:
                crop_length = int(crop_length * 0.8)  # reduce crop length if too large

        # sample MSA
        msa_profile = torch.tensor(biomol.MSA.profile)
        msa_deletion_mean = torch.tensor(biomol.MSA.deletion_mean)
        msa_sequence_sampled = []
        msa_has_deletion_sampled = []
        msa_deletion_value_sampled = []
        for _ in range(self.config.msa_config.n_samples):
            _, sampled_msa, sampled_has_deletion, sampled_deletion_value = (
                biomol.MSA.sample(
                    max_msa_depth=self.config.msa_config.max_msa_depth,
                )
            )
            msa_sequence_sampled.append(torch.tensor(sampled_msa))
            msa_has_deletion_sampled.append(torch.tensor(sampled_has_deletion))
            msa_deletion_value_sampled.append(torch.tensor(sampled_deletion_value))
        msa_sequence_sampled = torch.stack(msa_sequence_sampled, dim=0)
        msa_has_deletion_sampled = torch.stack(msa_has_deletion_sampled, dim=0)
        msa_deletion_value_sampled = torch.stack(
            msa_deletion_value_sampled, dim=0
        ).float()

        # Now convert biomol to batch
        atom_residue_type = biomol.structure.atom_tensor[:, 0]  # (L_atom)
        atom_pos = biomol.structure.atom_tensor[:, 5:8]  # (L_atom, 3)
        atom_mask = biomol.structure.atom_tensor[:, 4].bool()  # (L_atom)
        atom_bond = biomol.structure.atom_bond  # (n_atom_bond, 6)

        # Tensor of residue xyz, mask, bond
        cropped_len = biomol.structure.residue_tensor.shape[0]
        residue_pos = biomol.structure.residue_tensor[:, 5:8]
        residue_mask = biomol.structure.residue_tensor[:, 4].bool()
        residue_bond = biomol.structure.residue_bond  # (n_residue_bond, 3)
        if residue_bond.shape == torch.Size([0]):
            residue_bond = torch.zeros((0, 3), dtype=torch.int32)

        # idx map
        _, atom_to_residue_idx_map = torch.unique(
            biomol.structure.atom_tensor[:, 2], sorted=True, return_inverse=True
        )

        # ids
        residue_chain_break = biomol.structure.residue_chain_break
        chain_num = len(residue_chain_break)
        chain_asym_id = torch.arange(chain_num, dtype=torch.long)
        same_entity = biomol.structure.same_entity  # (n_chain, n_chain)
        _, chain_entity_id = torch.unique(
            ~same_entity,
            dim=0,
            return_inverse=True,
        )
        chain_sym_id = (torch.triu(same_entity, diagonal=0).long()).sum(dim=0) - 1

        residue_idx = biomol.structure.residue_tensor[:, 2]
        residue_idx_mono = torch.arange(cropped_len, dtype=torch.long)
        residue_asym_id = [
            _id.repeat(end - start + 1)
            for _id, (start, end) in zip(chain_asym_id, residue_chain_break.values())
        ]
        residue_entity_id = [
            _id.repeat(end - start + 1)
            for _id, (start, end) in zip(chain_entity_id, residue_chain_break.values())
        ]
        residue_sym_id = [
            _id.repeat(end - start + 1)
            for _id, (start, end) in zip(chain_sym_id, residue_chain_break.values())
        ]
        residue_asym_id = torch.cat(residue_asym_id, dim=0)  # (L_res)
        residue_entity_id = torch.cat(residue_entity_id, dim=0)  # (L_res)
        residue_sym_id = torch.cat(residue_sym_id, dim=0)  # (L_res)

        # I used CCD idx for now
        residue_type = msa_sequence_sampled[0, 0]

        residue_ccd_idx = biomol.structure.residue_tensor[:, 1].long()
        residue_ccd_list = self.preprocessing.ideal_ligand[residue_ccd_idx.numpy()]
        ref_pos = []
        ref_mask = []
        ref_element = []
        ref_charge = []
        ref_space_uid = []

        R, T = self.SE3_oper(cropped_len)
        for ii, residue_ccd in enumerate(residue_ccd_list):
            chem_comp = biomol.structure.chem_comp_dict[residue_ccd]
            ideal_coords = chem_comp.get_ideal_coords()
            _charge = chem_comp.get_charges()
            _element, _pos, _mask = (
                ideal_coords[:, 0],
                ideal_coords[:, 1:4],
                ideal_coords[:, 4],
            )
            _pos = (_pos - _pos.mean(axis=0)) @ R[ii] + T[ii]  # random SE(3) operation
            ref_pos.append(_pos)
            ref_mask.append(_mask)
            ref_element.append(_element)
            ref_charge.append(_charge[:, 1])
            ref_space_uid.extend(
                [ii] * len(_element)
            )  # space uid is the index of residue
        ref_pos = torch.cat(ref_pos, dim=0).to(torch.float32)  # (L_atom, 3)
        ref_mask = torch.cat(ref_mask, dim=0).to(torch.float32)  # (L_atom)
        ref_element = torch.cat(ref_element, dim=0).to(torch.float32)  # (L_atom)
        ref_charge = torch.cat(ref_charge, dim=0).to(torch.float32)  # (L_atom)
        ref_space_uid = torch.tensor(ref_space_uid, dtype=torch.long)  # (L_atom)

        # centering atom_pos and residue_pos
        mean_vector = atom_pos.mean(dim=0, keepdim=True)
        atom_pos = atom_pos - mean_vector  # (L_atom, 3)
        residue_pos = residue_pos - mean_vector  # (L_res, 3)

        sequence = SequenceFeatures.from_sample(
            atom_residue_type=atom_residue_type.long(),
            residue_type=residue_type.long(),
            residue_ccd_idx=residue_ccd_idx.long(),
        )
        structure = StructureFeatures.from_sample(
            atom_pos=atom_pos,
            atom_mask=atom_mask,
            atom_bond=atom_bond,
            residue_pos=residue_pos,
            residue_mask=residue_mask,
            residue_bond=residue_bond,
        )
        reference = ReferenceFeatures.from_sample(
            pos=ref_pos,
            mask=ref_mask,
            element=ref_element,
            charge=ref_charge,
            space_uid=ref_space_uid,
        )
        scheme = SchemeFeatures.from_sample(
            crop_indices=crop_indices,
            residue_idx=residue_idx.long(),
            residue_idx_mono=residue_idx_mono,
            residue_asym_id=residue_asym_id,
            residue_entity_id=residue_entity_id,
            residue_sym_id=residue_sym_id,
            atom_to_residue_idx_map=atom_to_residue_idx_map,
            atom_chain_break=biomol.structure.atom_chain_break,
            residue_chain_break=residue_chain_break,
        )

        msa = MSAFeatures.from_sample(
            aligned_sequences=msa_sequence_sampled,
            has_deletion=msa_has_deletion_sampled.long(),
            deletion_value=msa_deletion_value_sampled,
            profile=msa_profile,
            deletion_mean=msa_deletion_mean,
        )

        contact_graph = biomol.structure.contact_graph.graphs[(assembly_ID,model_ID,alt_ID)]
        same_entity = biomol.structure.same_entity
        entity_list = biomol.structure.entity_list
        chain = ChainFeatures.from_sample(
            contact_graph=contact_graph,
            same_entity=same_entity,
            entity_list=entity_list,
        )

        return Batch(
            name=[f"{pdb_ID}_{assembly_ID}_{model_ID}_{alt_ID}"],
            sequence=sequence,
            structure=structure,
            reference=reference,
            scheme=scheme,
            msa=msa,
            chain=chain,
        )

    def create_ddp_dataloader(self, **kwargs):
        sampler = BioMolSampler(
            BioMolSampler.Config(
                dataset=self,
                num_replicas=kwargs.get("world_size", 1),
                rank=kwargs.get("rank", 0),
                shuffle=kwargs.get("shuffle", True),
                seed=kwargs.get("seed", 0),
                drop_last=kwargs.get("drop_last", False),
            )
        )

        kwargs.update({"sampler": sampler})
        return super().create_ddp_dataloader(**kwargs)


if __name__ == "__main__":
    data_preprocessing_config = BioMolPreProcessing.Config(
        meta=BioMolPreProcessing.MetaConfig(
            metadata_path=f"{DB_PATH}/metadata/ID_to_deposition.pkl",
            graph_cluster_metadata_path=f"{DB_PATH}/cluster/graph_cluster/graph_cluster_metadata.json",
            chainID_to_cluster_path=f"{DB_PATH}/cluster/seq_clust/chain_ID_to_cluster.pkl",
            node_score_path=f"{DB_PATH}/statistics/node_score.pkl",
            edge_score_path=f"{DB_PATH}/statistics/edge_score.pkl",
            unique_graphs_path=f"{DB_PATH}/cluster/graph_cluster/cluster_level_unique_graphs.pkl",
            ideal_ligand_path=f"{DB_PATH}/metadata/ideal_ligand_list.pkl",
        ),
        pipeline=BioMolPreProcessing.PipelineConfig(
            graph_hash_path=f"{DB_PATH}/cluster/graph_cluster/train_graph_hash.txt",
            thread_num=64,
            filter_date="2024-10-21",
            filter_resolution=9.0,
            filter_chain_num=40,
            filtered_item_path="./data_tmp/train/filtered_item.pkl",
            data_tmp_dir="./data_tmp/",
        ),
        mol_types=MolTypeConfig(
            protein=True,
            nucleic_acid=True,
            ligand=True
        )
    )
    config = BioMolData.BioMolConfig(
        crop_config=CropConfig(
            contiguous_prob=0.2,
            spatial_prob=0.4,
            interface_prob=0.4,
            crop_length=384,
            level="residue",
        ),
        data_preprocessing_config=data_preprocessing_config,
    )
    dataset = BioMolData(config)

    # # find 8aug
    # for ii, item in enumerate(dataset.preprocessing.items):
    #     ID_list = list(item.values())
    #     # flatten embedded list
    #     ID_list = [ID for sublist in ID_list for ID in sublist]
    #     pdb_ID_list = [ID.split("_")[0] for ID in ID_list]
    #     if "8aug" in pdb_ID_list:
    #         breakpoint()


    for _ in range(1000):
        print(f"Sampling data... {_}")
        test_data = dataset[161471]
    breakpoint()

    # for ii in range(len(dataset)):
    #     out = f"Processing item {ii}..."
    #     if ii % 100 == 0:
    #         print(f"{ii} / {len(dataset)}")
    #     try:
    #         batch = dataset[ii]
    #         residue_length = batch.structure.residue_pos.shape[1]
    #         atom_length = batch.structure.atom_pos.shape[1]
    #         print(
    #             f"{out} {batch.name} | residue length: {residue_length}, atom length: {atom_length}"
    #         )
    #     except Exception as e:
    #         out += f" Failed with error: {e}"
    #         print(out)
    #         continue
    # test_data = dataset[601]
    # breakpoint()

    # for _ in range(100):
    #     try:
    #         test_data = dataset[601]
    #     except:
    #         breakpoint()

    # dataloader = dataset.create_dataloader(
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=4,
    #     pin_memory=True,
    # )
    # ii = 0
    # for batch in dataloader:
    #     print(
    #         f"({ii}) Batch {batch.name[0]} | residue length: {batch.residue_pos.shape[1]}, atom length: {batch.atom_pos.shape[1]}"
    #     )
    #     ii += 1
