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


from MiniWorld.data.features_multistate import Batch, SequenceFeatures, StructureFeatures, ReferenceFeatures, SchemeFeatures, MSAFeatures
from BioMol.BioMol import BioMol
from BioMol.constant.chemical import AA2num, num2AA
from BioMol import DB_PATH
from kmer_fast_align import Searcher

import lmdb
atom_db_env = f"{DB_PATH}/seq_to_str/atom.lmdb"
residue_db_env = f"{DB_PATH}/seq_to_str/residue.lmdb"
def read_seq_lmdb(key: str, level: str = "atom"):
    """
    Read a sequence from the LMDB database.
    """
    if level not in ["atom", "residue"]:
        raise ValueError("level must be either 'atom' or 'residue'.")
    db_env = atom_db_env if level == "atom" else residue_db_env
    env = lmdb.open(db_env, readonly=True)
    with env.begin() as txn:
        data = txn.get(key.encode())
        if data is None:
            raise ValueError(f"Key {key} not found in the database.")
        data = pickle.loads(data)

    env.close()
    return data

def fix_padding(idx: torch.Tensor, pad_val: int = -1):
    L = idx.size(1)
    d = idx[:, 1:] - idx[:, :-1]
    drop = d < 0
    any_drop = drop.any(dim=1)
    first_drop = drop.int().argmax(dim=1)
    last_valid = torch.where(any_drop, first_drop, torch.full_like(first_drop, L-1))

    pos = torch.arange(L, device=idx.device).unsqueeze(0)  # (1, L)
    mask = pos > last_valid.unsqueeze(1)
    return idx.masked_fill(mask, pad_val)

def to_mmcif(
    batch: Batch,
    denoised_atom_pos: torch.Tensor,
    true_mmcif_path: PosixPath,
    denoised_mmcif_path: PosixPath,
    mol_types: list[str] = ["protein"],
):
    pdb_id, assembly_id, model_id, alt_id = batch.name[0].split("_")
    crop_indices = batch.crop_indices[0]
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


def kabsch_rmsd(
    coords: torch.Tensor,
    mask: torch.Tensor,          # shape (N, L, 1), dtype=bool or {0,1}
) -> torch.Tensor:
    """
    Calculates the NxN RMSD matrix for N sets of coordinates using the Kabsch algorithm,
    correctly handling masks.
    """
    N, L, _ = coords.shape
    device = coords.device
    dtype  = torch.float32

    # 1) Cast coords & mask
    coords = coords.to(device, dtype)
    mask_b = mask.to(device, torch.bool).squeeze(-1) # (N, L), boolean

    # 2) Per-structure centroid over valid atoms
    counts    = mask_b.sum(dim=1, keepdim=True).clamp(min=1)     # (N,1)
    centroid  = torch.sum(coords * mask_b.unsqueeze(-1), dim=1, keepdim=True) / counts.unsqueeze(-1)  # (N,1,3)

    # 3) Center & zero invalid
    coords = (coords - centroid) * mask_b.unsqueeze(-1)                        # (N,L,3)

    # 4) Broadcast for all pairs
    P = coords.unsqueeze(1)  # (N,1,L,3) -> (N,N,L,3)
    Q = coords.unsqueeze(0)  # (1,N,L,3) -> (N,N,L,3)
    
    # Create a pairwise mask to find common valid atoms
    mask_p = mask_b.unsqueeze(1) # (N, 1, L)
    mask_q = mask_b.unsqueeze(0) # (1, N, L)
    pairwise_mask = (mask_p & mask_q) # (N, N, L)

    # Number of common atoms for each pair
    # Add a small epsilon to avoid division by zero
    L_pairwise = pairwise_mask.sum(dim=-1).clamp(min=1e-8) # (N, N)

    # Apply pairwise mask to coordinates
    P_masked = P * pairwise_mask.unsqueeze(-1)
    Q_masked = Q * pairwise_mask.unsqueeze(-1)

    # 5) Squared norms over common valid points
    P_norm_sq = torch.sum(P_masked**2, dim=(-1, -2))  # (N,N)
    Q_norm_sq = torch.sum(Q_masked**2, dim=(-1, -2))  # (N,N)

    # 6) Covariance over common valid points
    H = P_masked.transpose(-1, -2) @ Q_masked                # (N,N,3,3)
    H = H + torch.eye(3, device=device, dtype=dtype).view(1,1,3,3) * 1e-2

    # 7) Batched SVD
    Hf = H.reshape(-1, 3, 3)                   # (N*N,3,3)
    U, S, Vt = torch.linalg.svd(Hf)
    V = Vt.transpose(-1, -2)
    det_sign = torch.sign(torch.det(V @ U.transpose(-1, -2)))
    S[..., -1] *= det_sign
    trace_S = S.sum(dim=-1).reshape(N, N)      # (N,N)

    # 8) Assemble RMSD with correct pairwise normalization
    rmsd_sq = (P_norm_sq + Q_norm_sq - 2 * trace_S) / L_pairwise
    rmsd_sq = torch.clamp(rmsd_sq, min=0.0)
    
    return torch.sqrt(rmsd_sq)                 # (N,N)
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


class BioMolMonomerPreProcessing:
    class MetaConfig(BaseModel):
        seq_to_cluster_path: PosixPath | str = "seq_to_cluster.pkl"
        seq_to_hash_path: PosixPath | str = "seq_to_hash.pkl"
        ideal_ligand_path: PosixPath | str = "ideal_ligand.pkl"
        metadata_path: PosixPath | str

    class PipelineConfig(BaseModel):
        thread_num: int = 8
        filter_date: str = "2024-10-21"
        filter_min_length: int = 32
        filter_mask_ratio: float = 0.5
        filter_resolution: float = 9.0
        filtered_item_path: PosixPath | str | None = None
        cluster_path: PosixPath | str = "cluster.pkl"
        data_tmp_dir: PosixPath | str = "./data_tmp/monomer/"

    class Config(BaseModel):
        meta: "BioMolMonomerPreProcessing.MetaConfig"
        pipeline: "BioMolMonomerPreProcessing.PipelineConfig"

    def __init__(self, config: Config):
        self.config = config

        if isinstance(self.config.pipeline.data_tmp_dir, str):
            self.config.pipeline.data_tmp_dir = Path(self.config.pipeline.data_tmp_dir)
        if not self.config.pipeline.data_tmp_dir.exists():
            self.config.pipeline.data_tmp_dir.mkdir(parents=True, exist_ok=True)

        self.seq_to_cluster = self._load_pickle_file(self.config.meta.seq_to_cluster_path)
        self._load_seq_to_hash(self.config.meta.seq_to_hash_path)
        self.ideal_ligand = self._load_pickle_file(self.config.meta.ideal_ligand_path)
        self.ideal_ligand = np.array(self.ideal_ligand)
        self._load_items_and_filter(self.config.pipeline.filtered_item_path)

    def _load_seq_to_hash(self, seq_to_hash_path: PosixPath | str):
        """
        Load the seq_to_hash mapping from a file.
        """
        if isinstance(seq_to_hash_path, str):
            seq_to_hash_path = Path(seq_to_hash_path)
        if not seq_to_hash_path.exists():
            raise FileNotFoundError(f"File {seq_to_hash_path} does not exist.")

        with open(seq_to_hash_path) as f:
            fasta_lines = f.readlines()
        hash_list = []
        seq_list = []
        for line in fasta_lines:
            if line.startswith('>'):
                seq_hash = line[1:].strip()
                hash_list.append(seq_hash)
            else :
                seq = line.strip()
                seq_list.append(seq)
        seq_to_hash = {seq: seq_hash for seq, seq_hash in zip(seq_list, hash_list)}
        hash_to_seq = {seq_hash: seq for seq, seq_hash in zip(seq_list, hash_list)}
        self.seq_to_hash = seq_to_hash
        self.hash_to_seq = hash_to_seq
        print(f"Loaded {len(seq_to_hash)} sequences from {seq_to_hash_path}.")


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
            self.items = data
            return
        else:
            if self.config.pipeline.filter_date is not None:
                date_cutoff = date.strptime(self.config.pipeline.filter_date, "%Y-%m-%d")
            else:
                date_cutoff = date.strptime("2099-01-01", "%Y-%m-%d")

            if self.config.pipeline.filter_resolution is not None:
                resolution_cutoff = float(self.config.pipeline.filter_resolution)
            else:
                resolution_cutoff = float("inf")

            clusters = self._load_pickle_file(self.config.pipeline.cluster_path)

            items = {} # dict of {seq_cluster: {seq_hash: [pdb_IDs]}}
            for seq, cluster in self.seq_to_cluster.items():
                if cluster not in clusters:
                    continue
                if '[PROTEIN]' not in seq:
                    continue
                seq = seq.split(":")[-1]

                # remove seq if seq is 'XXX...'
                unique_chars = set(seq)
                if len(unique_chars) == 1 and unique_chars.pop() == 'X':
                    continue

                if len(seq) < self.config.pipeline.filter_min_length:
                    continue
                seq_hash = self.seq_to_hash[seq]
                if cluster not in items:
                    items[cluster] = {}
                items[cluster][seq_hash] = []

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

            for cluster_ID, hash_dict in chunk:
                filtered_hash_dict = {}
                for seq_hash in hash_dict:
                    seq_hash = str(seq_hash).zfill(6)
                    data = read_seq_lmdb(seq_hash, level="residue")
                    residue_tensor = data['tensors'] # (B, L ,10)
                    cif_IDs = data['cif_IDs']  # list of IDs
                    mask = (residue_tensor[:, :, 4] == 1) & (residue_tensor[:, :, 0] != 21) # TODO bug for seq2str db
                    valid_residue_ratio = mask.sum(dim=-1) / mask.shape[-1] # (B, 1)
                    valid_residue_num = mask.sum(dim=-1) # (B, 1)

                    filtered_indices = torch.where(
                        (valid_residue_ratio >= self.config.pipeline.filter_mask_ratio) & (valid_residue_num >= self.config.pipeline.filter_min_length)
                    )[0]
                    filtered_cif_IDs = [
                        cif_IDs[i] for i in filtered_indices.tolist() if cif_IDs[i].split("_")[0] in filtered_IDs
                    ]
                    filtered_cif_IDs = [
                        cif_IDs.split(".pt")[0] for cif_IDs in filtered_cif_IDs
                    ]
                    if len(filtered_cif_IDs) == 0:
                        continue
                    filtered_hash_dict[seq_hash] = filtered_cif_IDs
                if len(filtered_hash_dict) == 0:
                    continue
                out.append(
                    (cluster_ID, filtered_hash_dict)
                )
            return out

        n = self.config.pipeline.thread_num

        chunks = chunked_items(items, n)
        results = Parallel(n_jobs=n, verbose=10)(
            delayed(_filter_chunk)(chunk) for chunk in chunks
        )

        # collect only the passed ones
        filtered_items = {}
        filtered_cif_IDs = set()
        for out_list in results:
            for out in out_list:
                cluster_ID, filtered_hash_dict = out
                filtered_items[cluster_ID] = filtered_hash_dict
                for cif_IDs in filtered_hash_dict.values():
                    filtered_cif_IDs.update(cif_IDs)

        self.items = [[cluster_ID, filtered_hash_dict] for cluster_ID, filtered_hash_dict in filtered_items.items()]

        # save the items and etc at data_tmp_dir/filtered_items.pkl
        with open(filtered_item_path, "wb") as f:
            pickle.dump(self.items,f)


class BioMolSampler(DistributedSampler):
    class Config(BaseModel):
        # 1) let Pydantic accept any Python class as a field
        model_config = ConfigDict(arbitrary_types_allowed=True)

        # 2) annotate with the real class (not a primitive)
        dataset: "BioMolMonomerData"
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
    contiguous_prob: float = 0.5
    spatial_prob: float = 0.5
    crop_length: int = 384
    level: Literal["residue", "atom"] = "residue"

    @property
    def crop_method_prob(self) -> list[float]:
        return [
            self.contiguous_prob,
            self.spatial_prob,
            0.0,
        ]

class KmerFastAlignConfig(BaseModel):
    kmer_index: str = "kmer_index.tsv"
    fasta: str = "sequence_hashes.fasta"
    kmer_threshold: float = 0.2
    gap_split: int = 15
    max_mismatch: int = 15
    max_indel: int = 5
    align_num: int = -1  # -1 means no limit
    align_thr: float = 0.9
    seed: int = 1123  # for reproducibility


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


class MSAConfig(BaseModel):
    n_samples: int = 4
    max_msa_depth: int = 512

class MultistateConfig(BaseModel):
    n_prefilter: int = 128
    n_samples: int = 48
    temperatures: float = 1.0
    consensus_ratio : float = 0.9
    consensus_filter: float = 0.9



class BioMolMonomerData(BaseData):
    class BioMolConfig(BaseModel):
        crop_config: CropConfig = CropConfig()
        msa_config: MSAConfig = MSAConfig()
        data_preprocessing_config: BioMolMonomerPreProcessing.Config = None
        mol_types: MolTypeConfig = MolTypeConfig()
        kmer_fast_align_config: KmerFastAlignConfig = KmerFastAlignConfig()
        multistate_config: MultistateConfig = MultistateConfig()

    def __init__(
        self,
        config: BioMolConfig,
    ):
        super().__init__(transform=None)
        self.config = config
        self.preprocessing = BioMolMonomerPreProcessing(config.data_preprocessing_config)
        self.crop_method_prob = [
            config.crop_config.contiguous_prob,
            config.crop_config.spatial_prob,
        ]
        self.crop_length = config.crop_config.crop_length
        self.crop_level = config.crop_config.level
        self.mol_types = config.mol_types.mol_types  # list of mol types
        self.searcher = Searcher(
            kmer_index=config.kmer_fast_align_config.kmer_index,
            fasta=config.kmer_fast_align_config.fasta,
        )
        self.kmer_align_options = {
            "kmer_threshold": config.kmer_fast_align_config.kmer_threshold,
            "gap_split": config.kmer_fast_align_config.gap_split,
            "max_mismatch": config.kmer_fast_align_config.max_mismatch,
            "max_indel": config.kmer_fast_align_config.max_indel,
            "align_num": config.kmer_fast_align_config.align_num,
            "align_thr": config.kmer_fast_align_config.align_thr,
        }

    def __len__(self):
        return len(self.preprocessing.items)

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

    def _gen_kmer_query(self, query_full_seq: str, crop_indices: torch.Tensor) -> str:
        # Step 1: Build mask
        seq_len = len(query_full_seq)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[crop_indices] = True

        # Step 2: Build gapped sequence
        gapped_seq = ''.join(res if keep else '-' for res, keep in zip(query_full_seq, mask))

        # Step 3: Trim leading and trailing gaps
        trimmed_seq = gapped_seq.strip('-')
        gap_indices = [i for i, char in enumerate(trimmed_seq) if char == '-']

        return trimmed_seq, gap_indices
 
    def __getitem__(self, idx: int) -> Batch:
        seq_cluster, hash_dict = self.preprocessing.items[idx]

        # uniformly sample a hash level graph from the list
        hash_list = list(hash_dict.keys())
        if len(hash_list) == 0:
            raise ValueError(f"No valid hash found for index {idx}.")
        query_hash = random.choice(hash_list)
        IDs = hash_dict[query_hash]  # list of IDs in the hash
        if len(IDs) == 0:
            breakpoint()
            raise ValueError(f"No valid IDs found for hash {query_hash} at index {idx}.")

        # for cropping we have to sample one ID
        ID = random.choice(IDs)
        pdb_ID, assembly_ID, model_ID, alt_ID, chain_ID = ID.split("_")
        biomol = BioMol(pdb_ID=pdb_ID)  # warning, mol type is not filtered out.
        biomol.choose(assembly_ID, model_ID, alt_ID)
        chain_list = list(biomol.structure.residue_chain_break.keys())
        # choose any chain starting with chain_ID
        chain_ID = [c for c in chain_list if c.split("_")[0] == chain_ID][0]
        # filter out chain by type
        biomol.structure.filter_by_type(self.mol_types)
        filtered_chain_IDs = biomol.structure.residue_chain_break.keys()
        if len(filtered_chain_IDs) == 0:
            raise ValueError(
                "After filtering by mol type, no chains left in the biomolecule. You have to filter train/valid items correspondingly."
            )

        crop_length = self.crop_length
        while crop_length > 0:
            crop_indices, seq_hash_to_crop_indices = biomol.get_crop_indices(
                chain_bias=chain_ID,
                interface_bias=None,
                contiguous_crop_weight=self.crop_method_prob[0],
                spatial_crop_weight=self.crop_method_prob[1],
                interface_crop_weight=0.0,
                crop_length=crop_length,
                level=self.crop_level,
                monomer_only=True,  # monomer only
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

        query_full_seq = self.preprocessing.hash_to_seq[query_hash]
        try:
            query_crop_indices = seq_hash_to_crop_indices[query_hash][0]
        except:
            breakpoint()
        query, gap_indices = self._gen_kmer_query(query_full_seq, query_crop_indices)
        aligned_seqs, num_aligned = self.searcher.search(query=query, **self.kmer_align_options)
        # assert num_aligned > 0, f"No aligned sequences found for {idx} query {query_hash} : {query}."
        # In some cases, there might be no aligned sequences due to the query being too sparse after cropping.
        # Ex) "GR-----P------------------T--VT-L--------------------------R-YLRFH---Y" (idx 16330 query 372637)
        if num_aligned == 0:
            # add query to the aligned sequences
            aligned_seqs.append({
                "id": query_hash,
                "qs_map": [(i, i) for i in range(len(query)) if query[i] != '-'],
                "s_aln": query
            })

        full_target_hashes = [query_hash]
        seq_stacked = [biomol.structure.residue_tensor[:, 0].tolist()]
        crop_indices_map = {query_hash : query_crop_indices}
        for match_result in aligned_seqs:
            target_hash = match_result["id"]
            if target_hash == query_hash:
                continue
            target_seq = self.preprocessing.hash_to_seq[target_hash]
            qs_map = match_result["qs_map"]
            aligned_indices = [indices[1] for indices in qs_map if indices[0] not in gap_indices]
            aligned_seq = [target_seq[i] if i != -1 else '-' for i in aligned_indices]
            # convert to numerical representation. 
            aligned_seq = [AA2num.get(aa, AA2num['X']) for aa in aligned_seq]
            seq_stacked.append(aligned_seq)
            crop_indices_map[target_hash] = torch.tensor(aligned_indices, dtype=torch.long)
            full_target_hashes.append(target_hash)
        seq_stacked = torch.tensor(seq_stacked, dtype=torch.long)  # (N_seq, L_seq)
        N, L = seq_stacked.shape
        same_full = (seq_stacked == seq_stacked[0])
        scores = same_full.float().mean(dim=1)

        drop_order = torch.argsort(scores)
        drop_order = drop_order[drop_order != 0]
        M = drop_order.numel()
        pos = torch.full((N,), M, dtype=torch.long)
        pos[drop_order] = torch.arange(M)

        K = M + 1
        ks = torch.arange(K)[:, None]
        keep_masks = (pos[None, :] >= ks)
        same_exp = same_full.unsqueeze(0)
        keep_exp = keep_masks.unsqueeze(-1)
        same_masked = torch.where(keep_exp, same_exp, torch.ones(1, N, L, dtype=torch.bool))
        consensus_mask_allK = same_masked.all(dim=1)
        consensus_ratio = consensus_mask_allK.float().mean(dim=1)
        ge = (consensus_ratio >= self.config.multistate_config.consensus_ratio)
        cmax = torch.cumsum(ge.to(torch.int64), dim=0) > 0
        first_idx = torch.argmax(cmax.int())
        best_k = torch.where(ge.any(), first_idx, torch.tensor(M))
        final_keep = keep_masks[best_k]
        consensus_mask = consensus_mask_allK[best_k]

        kept_indices = torch.nonzero(final_keep, as_tuple=False).squeeze(1).tolist()
        target_hashes = [full_target_hashes[i - 1] for i in kept_indices]
        target_hashes.append(query_hash)
        kept_crop_indices_map = {query_hash: crop_indices_map[query_hash]}
        for h in target_hashes:
            kept_crop_indices_map[h] = crop_indices_map[h]
        crop_indices_map = kept_crop_indices_map

        # same = (seq_stacked == seq_stacked[0])
        # consensus_mask = same.all(dim=0)

        crop_indices_map = {k: v[consensus_mask] for k, v in crop_indices_map.items()}
        biomol.crop(crop_indices=torch.where(consensus_mask)[0], crop_MSA=False)  # crop biomol with the first sequence's crop indices
        # TODO 1. deepcopy biomol or 2. second crop MSA with the consensus mask

        # load structure
        # centering atom_pos
        cif_IDs = []
        atom_pos = []
        atom_pos_mask = []

        # read query data
        query_data = read_seq_lmdb(query_hash, level="atom")
        query_cif_IDs = query_data['cif_IDs']
        query_cif_IDs = [cif_ID.split(".pt")[0] for cif_ID in query_cif_IDs]
        query_idx = np.where(np.array(query_cif_IDs) == ID)[0]
        query_tensor = query_data['tensors']['idx_related']
        query_atom_to_residue_map = query_tensor[query_idx,:,2]
        query_atom_to_residue_map = fix_padding(query_atom_to_residue_map)
        query_atom_to_residue_map = query_atom_to_residue_map-query_atom_to_residue_map[0,0]
        query_crop_indices = crop_indices_map[query_hash]
        query_crop_mask = torch.isin(query_atom_to_residue_map, query_crop_indices)
        query_chem_comp = query_tensor[query_idx,query_crop_mask[0],1].unsqueeze(0)

        for target_hash in target_hashes:
            crop_indices = crop_indices_map[target_hash]
            residue_data = read_seq_lmdb(target_hash, level="residue")
            atom_data = read_seq_lmdb(target_hash, level="atom")

            target_cif_IDs = atom_data['cif_IDs']  # list of IDs
            residue_mask = residue_data['tensors'][:, crop_indices, 4].squeeze(-1)  # (N_str, L_res)

            atom_mask = atom_data['tensors']['mask'][...,0]  # (N_str, L_atom)
            atom_to_residue_map = atom_data['tensors']['idx_related'][:,:,2]
            atom_to_residue_map = fix_padding(atom_to_residue_map)
            atom_to_residue_map = atom_to_residue_map - atom_to_residue_map[:,0:1]
            atom_crop_mask = torch.isin(atom_to_residue_map, crop_indices)
            xyz = atom_data['tensors']['xyz']  # (N_str, L_atom, 3)
            mask = atom_data['tensors']['mask'] # (N_str, L_atom)
            chem_comp = atom_data['tensors']['idx_related'][:,:,1]  # (N_str, L_atom)

            xyz_cropped = []
            mask_cropped = []
            valid_structures = []

            for ii in range(xyz.shape[0]):
                chem_comp_temp = chem_comp[ii, atom_crop_mask[ii]]
                if chem_comp_temp.shape[0] != query_chem_comp.shape[1]:
                    if ii == query_idx and query_hash==target_hash:
                        breakpoint()
                    continue
                chem_comp_diff = chem_comp_temp - query_chem_comp
                if chem_comp_diff.abs().sum(dim=1) > 0:
                    if ii == query_idx and query_hash==target_hash:
                        breakpoint()
                    continue
                mask_temp = mask[ii, atom_crop_mask[ii]]
                if mask_temp.sum() == 0:
                    if ii == query_idx and query_hash==target_hash:
                        breakpoint()
                    continue
                xyz_temp = xyz[ii, atom_crop_mask[ii]]
                xyz_cropped.append(xyz_temp)
                mask_cropped.append(mask_temp)
                valid_structures.append(ii)

            if len(valid_structures) == 0:
                if ii == query_idx and query_hash==target_hash:
                    breakpoint()
                continue

            xyz = torch.stack(xyz_cropped, dim=0)
            mask = torch.stack(mask_cropped, dim=0)

            target_cif_IDs = [target_cif_IDs[i] for i in valid_structures]
            cif_IDs.extend(target_cif_IDs)
            atom_pos.append(xyz)
            atom_pos_mask.append(mask)

        try:
            atom_pos = torch.cat(atom_pos, dim=0)  # (N_str, L_atom, 3)
        except:
            print(f"idx {idx} query_hash {query_hash}")
            breakpoint()
            assert 1==0
        atom_pos_mask = torch.cat(atom_pos_mask, dim=0)  # (N_str, L_atom)


        if atom_pos.shape[0] > self.config.multistate_config.n_prefilter:
            assembly_ID_list = ["_".join(cif_ID.split("_")[0:2]) for cif_ID in cif_IDs]
            assembly_ID_set = set(assembly_ID_list)
            ID_map = {}
            for assembly_ID in assembly_ID_set:
                ID_map[assembly_ID] = [cif_ID for cif_ID in cif_IDs if cif_ID.startswith(assembly_ID)]
            sampled_cif_IDs = []
            for assembly_ID in assembly_ID_set:
                sampled_cif_ID = random.sample(ID_map[assembly_ID], 1)
                sampled_cif_IDs.extend(sampled_cif_ID)
            sampled_cif_IDs = set(sampled_cif_IDs)
            sampled_indices = [i for i, cif_ID in enumerate(cif_IDs) if cif_ID in sampled_cif_IDs]
            atom_pos = atom_pos[sampled_indices]  # (N', L_atom, 3)
            atom_pos_mask = atom_pos_mask[sampled_indices]  # (N',
        if atom_pos.shape[0] > self.config.multistate_config.n_prefilter:
            # if the number of structures is still larger than n_prefilter, randomly sample n_prefilter structures
            sampled_indices = random.sample(range(atom_pos.shape[0]), self.config.multistate_config.n_prefilter)
            atom_pos = atom_pos[sampled_indices]
            atom_pos_mask = atom_pos_mask[sampled_indices]

        atom_pos = atom_pos.float()  # convert to float
        kabsch_rmsd_value = kabsch_rmsd(atom_pos, atom_pos_mask)

        # stochastically select the structure with the minimum RMSD
        if atom_pos.shape[0] <= self.config.multistate_config.n_samples:
            pass
        else:
            selected_indices = [0]
            while len(selected_indices) < self.config.multistate_config.n_samples:
                rmsd = kabsch_rmsd_value[selected_indices] # (N', N_str)
                repulsion = rmsd.sum(dim=0)  # (N_str, )
                repulsion[selected_indices] = float("inf")  # exclude already selected indices
                prob = torch.softmax(-repulsion/self.config.multistate_config.temperatures, dim=0) # (N_str, )
                sampled_idx = torch.multinomial(prob, num_samples=1).item()  # sample one index
                selected_indices.append(sampled_idx)
            atom_pos = atom_pos[selected_indices]
            atom_pos_mask = atom_pos_mask[selected_indices]

        mean_vector = atom_pos.mean(dim=1, keepdim=True)
        atom_pos = atom_pos - mean_vector  # center the atom positions

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
        msa_deletion_value_sampled = torch.stack(msa_deletion_value_sampled, dim=0).float()

        # consensus mask
        msa_profile = msa_profile[consensus_mask]  # (L_cropped, 20)
        msa_deletion_mean = msa_deletion_mean[consensus_mask]  # (L_cropped, 1)
        msa_sequence_sampled = msa_sequence_sampled[:, :, consensus_mask]  # (N_sample, N_seq, L_cropped)
        msa_has_deletion_sampled = msa_has_deletion_sampled[:, :, consensus_mask]  # (N_sample, N_seq, L_cropped)
        msa_deletion_value_sampled = msa_deletion_value_sampled[:, :, consensus_mask]  # (N_sample, N_seq, L_cropped)

        # Now convert biomol to batch
        atom_residue_type = biomol.structure.atom_tensor[:, 0]  # (L_atom)
        atom_mask = biomol.structure.atom_tensor[:, 4].bool()  # (L_atom)
        atom_bond = biomol.structure.atom_bond  # (n_atom_bond, 6)

        # Tensor of residue xyz, mask, bond
        cropped_len = biomol.structure.residue_tensor.shape[0]
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

        if ref_pos.shape[0] != atom_pos[0].shape[0]:
            print(ref_pos.shape, atom_pos[0].shape)
            breakpoint()

        assert ref_pos.shape[0] == atom_pos[0].shape[0], f"idx {idx}, query_hash {query_hash}, ref_pos.shape[0] {ref_pos.shape[0]}, atom_pos[0].shape[0] {atom_pos[0].shape[0]}"

        sequence = SequenceFeatures.from_sample(
            atom_residue_type = atom_residue_type.long(),
            residue_type = residue_type.long(),
            residue_ccd_idx = residue_ccd_idx.long(),
        )
        structure = StructureFeatures.from_sample(
            atom_pos=atom_pos,
            atom_pos_mask=atom_pos_mask.squeeze(-1).bool(),
            atom_mask=atom_mask,
            atom_bond=atom_bond,
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
            residue_chain_break=residue_chain_break,
        )

        msa = MSAFeatures.from_sample(
            aligned_sequences=msa_sequence_sampled,
            has_deletion=msa_has_deletion_sampled.long(),
            deletion_value=msa_deletion_value_sampled,
            profile=msa_profile,
            deletion_mean=msa_deletion_mean,
        )

        return Batch(
            name=[f"{pdb_ID}_{assembly_ID}_{model_ID}_{alt_ID}"],
            sequence=sequence,
            structure=structure,
            reference=reference,
            scheme=scheme,
            msa=msa,
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
    data_preprocessing_config = BioMolMonomerPreProcessing.Config(
        meta=BioMolMonomerPreProcessing.MetaConfig(
            seq_to_cluster_path = f"{DB_PATH}/cluster/seq_clust/seq_to_cluster.pkl",
            # seq_to_hash_path = f"{DB_PATH}/cluster/seq_clust/kmer_fast_align/sequence_hashes_wo_signalp.fasta",
            seq_to_hash_path = f"{DB_PATH}/cluster/seq_clust/kmer_fast_align/sequence_hashes.fasta",
            ideal_ligand_path=f"{DB_PATH}/metadata/ideal_ligand_list.pkl",
            metadata_path=f"{DB_PATH}/metadata/ID_to_deposition.pkl",
        ),
        pipeline=BioMolMonomerPreProcessing.PipelineConfig(
            thread_num=64,
            filter_date="2024-10-21",
            filter_min_length=32,
            filter_resolution=9.0,
            filter_chain_num=40,
            filtered_item_path="./data_tmp/monomer/valid/filtered_item.pkl",
            cluster_path=f"{DB_PATH}/cluster/seq_clust/valid_cluster.pkl",
            data_tmp_dir="./data_tmp/monomer",
        ),
    )
    config = BioMolMonomerData.BioMolConfig(
        crop_config=CropConfig(
            contiguous_prob=0.2,
            spatial_prob=0.4,
            interface_prob=0.4,
            crop_length=384,
            level="residue",
        ),
        data_preprocessing_config=data_preprocessing_config,
        kmer_fast_align_config=KmerFastAlignConfig(
            kmer_index=f"{DB_PATH}/cluster/seq_clust/kmer_fast_align/kmer_index.tsv",
            fasta=f"{DB_PATH}/cluster/seq_clust/kmer_fast_align/sequence_hashes_wo_signalp.fasta",
            kmer_threshold=0.2,
            gap_split=15,
            max_mismatch=15,
            max_indel=5,
            align_num=-1,  # -1 means no limit
            align_thr=0.9,
            seed=1123,  # for reproducibility
        ),
    )
    dataset = BioMolMonomerData(config)

    items = dataset.preprocessing.items

    # find idx that contains query_hash = "112294"
    query_hash = "617864"
    idx_list = [idx for idx, item in enumerate(items) if query_hash in item[1]]

    print(f"Found idx: {idx_list}")
    for _ in range(100):
        test_data = dataset[12249]
    breakpoint()

    from torch.utils.data import DataLoader

    def collate_fn(batch):
        for idx in batch:
            try:
                for ii in range(100):
                    _ = dataset[idx]
            except:
                print(f"Error idx: {idx}")
            print(f"Processed idx: {idx}")
        return None

    # loader = DataLoader(range(len(dataset)), batch_size=1, num_workers=8, collate_fn=collate_fn)
    # for _ in loader:
    #     pass

    # log_path = "logs/test_dataloader.log"
    # with open(log_path, "r") as f:
    #     lines = f.readlines()

    # error_idx_list = [int(line.strip().split("Error idx: ")[-1]) for line in lines if line.startswith("Error idx: ")]

    # for error_idx in error_idx_list:
    #     for _ in range(100):
    #         test_data = dataset[error_idx]

    test_path = "logs/run_af3_ms_v0.1.0.out"

    with open(test_path, "r") as f:
        lines = f.readlines()
    lines = lines[8:]
    test = "".join(lines)
    test = test.replace("\n","")
    test = test.split("|")
    starting_idx = [int(line.split("Starting load")[-1]) for line in test if line.startswith("Starting load ")]
    ending_idx = [int(line.split("Ending load")[-1]) for line in test if line.startswith("Ending load ")]

    not_loaded_idx = [idx for idx in starting_idx if idx not in ending_idx]
    not_loaded_idx = [23967,9267]
    breakpoint()
    for error_idx in not_loaded_idx:
        for _ in range(100):
            test_data = dataset[error_idx]
    breakpoint()

