from __future__ import annotations
import json
import functools
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import biomol
import numpy as np
import torch
from pydantic import BaseModel

from miniworld.configs.data_explicit import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TemplateConfig,
    TokenizerConfig,
)
from miniworld.data.features import (
    Batch,
    make_batch,
)
from miniworld.data.io import (
    load_cifmol,
    load_msa,
    load_templates,
)
from miniworld.data.pipeline import (
    ProteinTemplate,
    Tokenizer,
    get_chain_crop_indices,
    sample_msa,
)
from miniworld.data.pipeline.utils import (
    NoInterfaceError,
    find_interface_residues,
    remove_terminal_oxygen,
)
from miniworld.utils.crop import crop_spatial_segment_token

from .sampler import PDBWeightedSampler

if TYPE_CHECKING:
    from miniworld.data.mols import CIFMolAttached


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _bucketed_collate(
    batch_list: list[Batch],
    bucket_msa_multiple: int | None = None,
    bucket_template_multiple: int | None = 1,
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
    
    n_temp = batch.template_number
    msa_depth = batch.msa_depth
    n_tokens = batch.token_length
    n_atoms = batch.atom_length

    bucketed_msa = (
        _ceil_to_multiple(msa_depth, bucket_msa_multiple)
        if bucket_msa_multiple
        else msa_depth
    )

    bucketed_template = (
        _ceil_to_multiple(n_temp, bucket_template_multiple)
        if bucket_template_multiple
        else n_temp
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
        and bucketed_msa == msa_depth
        and bucketed_template == n_temp
        and bucketed_tokens == n_tokens
        and bucketed_atoms == n_atoms
    ):
        return batch

    dummy = Batch.empty(
        n_temp=n_temp,
        msa_depth=bucketed_msa,
        n_tokens=bucketed_tokens,
        n_atoms=bucketed_atoms,
    )
    padded = Batch.collate_fn([batch, dummy])
    return padded[0 : batch.batch_size]


class WrongCroppingError(ValueError):
    """Raised when the cropping strategy fails to produce a valid crop."""


class BioMolData(torch.utils.data.Dataset):
    """Dataset for biomolecular complexes based on BioMolDB."""

    class BioMolConfig(BaseModel):
        """Configuration for BioMolData."""

        crop_config: CropConfig = CropConfig()
        msa_config: MSAConfig = MSAConfig()
        template_config: TemplateConfig = TemplateConfig()
        DB_config: BioMolDBConfig = BioMolDBConfig()
        tokenizer_config: TokenizerConfig = TokenizerConfig()
        sampler_config: SamplerConfig | None = SamplerConfig()

    def __init__(
        self,
        config: BioMolConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.epoch: int = 0
        self.seed: int = 0
        self.tokenizer = Tokenizer(config=config.tokenizer_config)

        self._load_edge_to_cif_ids()

        self.chemcomp_to_fp_idx: dict[str, int] | None = None
        self.fp_unk_idx: int | None = None

        vocab_path = self.config.DB_config.fingerprint_vocab_path
        if vocab_path is not None:
            with vocab_path.open("r") as f:
                self.chemcomp_to_fp_idx = json.load(f)
                self.fp_unk_idx = int(self.chemcomp_to_fp_idx.get("UNK", 0))

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this dataset, which can be used to change sampling behavior."""
        self.epoch = epoch

    def _make_seed(self, idx: int) -> int:
        return (self.seed * 1000003 + self.epoch * 100003 + idx) & 0xFFFF_FFFF
            
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

    def get_crop_indices(
        self,
        cifmol: CIFMolAttached,
        crop_indices: np.ndarray | None,
        bias: list[str],
        max_tokens: int,
        max_atoms: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Get crop indices for a given cifmol, either by cropping or using provided indices."""
        
        if crop_indices is not None:
            crop_indices = np.asarray(crop_indices, dtype=np.int64)
            chain_id_to_crop_indices = get_chain_crop_indices(
                cifmol=cifmol,
                crop_indices=crop_indices,
            )
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
                    chain_id = rng.choice([chain_id1, chain_id2])
                    selected_atoms = cifmol.chains.select(chain_id=chain_id).atoms
            case _:
                msg = f"Unexpected chain_ids: {bias}"
                raise ValueError(msg)

        valid = np.all(np.isfinite(selected_atoms.xyz), axis=-1)
        selected_atoms = selected_atoms[valid]
        if len(selected_atoms) == 0:
            msg = f"No valid atoms found for bias {bias} in cifmol {cifmol.id}"
            raise WrongCroppingError(msg)
        center_xyz = selected_atoms.xyz[rng.integers(0, len(selected_atoms))]

        segment_size = int(
            rng.integers(
                self.config.crop_config.min_segment_size,
                self.config.crop_config.max_segment_size + 1,
            ),
        )        

        _, token_to_residue_idx_map = self.tokenizer.tokenize(cifmol)

        crop_indices = crop_spatial_segment_token(
            cifmol,
            np.asarray(center_xyz),
            tokens_to_res=token_to_residue_idx_map,
            segment_size=segment_size,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
        )

        chain_id_to_crop_indices = get_chain_crop_indices(
            cifmol=cifmol,
            crop_indices=crop_indices,
        )

        # crop_indices = cast("np.ndarray", crop_indices)
        crop_indices = np.asarray(
            cast("np.ndarray", crop_indices), dtype=np.int64
        )
        return crop_indices, chain_id_to_crop_indices  # pyright: ignore[reportPossiblyUnboundVariable]

    def __getitem__(self, idx: int) -> Batch:
        """Get a data sample by index."""
        seed = self._make_seed(idx)
        rng = np.random.default_rng(seed)
        edge_id = self.edge_id_list[idx]
        biases = self.edge_id_to_bias[edge_id]
        bias = rng.choice(biases)
        pdb_id, assembly_id, model_id, alt_id = bias.split("_")[:4]
        bias = re.findall(r"\(([^)]+)\)", bias)
        
        while True:
            try:
                item = self.get_item_by_id(
                    pdb_id=pdb_id.lower(),
                    assembly_id=assembly_id,
                    model_id=model_id,
                    alt_id=alt_id,
                    bias=bias,
                    rng=rng,
                )
                break
            except WrongCroppingError:
                idx = int(rng.integers(0, len(self)))
                edge_id = self.edge_id_list[idx]
                biases = self.edge_id_to_bias[edge_id]
                bias = rng.choice(biases)
                pdb_id, assembly_id, model_id, alt_id = bias.split("_")[:4]
                bias = re.findall(r"\(([^)]+)\)", bias)

        return item

    def get_item_by_id(
        self,
        pdb_id: str,
        assembly_id: str | None = None,
        model_id: str | None = None,
        alt_id: str | None = None,
        bias: list[str] | None = None,
        crop_indices: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> Batch:
        """Get a data sample by cif_id."""
        if rng is None:
            rng = np.random.default_rng()
        cifmol = load_cifmol(
            db_path=self.config.DB_config.cif_db_path,
            pdb_id=pdb_id,
            assembly_id=assembly_id,
            model_id=model_id,
            alt_id=alt_id,
        )

        cifmol = remove_terminal_oxygen(cifmol)
        if bias is None: # randoml sample chain_id
            chain_id = np.random.choice(cifmol.chains.chain_id.value)
            bias = [chain_id]
        if crop_indices is None:
            crop_indices, chain_id_to_crop_indices = self.get_crop_indices(
                cifmol=cifmol,
                crop_indices=crop_indices,
                bias=bias,
                max_tokens=self.config.crop_config.max_tokens,
                max_atoms=self.config.crop_config.max_atoms,
                rng=rng,
            )
            if crop_indices.shape[0] == 0:
                msg = f"Failed to crop {pdb_id}_{assembly_id}_{model_id}_{alt_id} with bias {bias}."
                raise WrongCroppingError(msg)
        else:
            chain_id_to_crop_indices = get_chain_crop_indices(
                cifmol=cifmol,
                crop_indices=crop_indices,
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
            max_msa_depth=self.config.msa_config.max_msa_depth,
            rng=rng,
        )
        
        templates: ProteinTemplate = load_templates(
            cifmol=cifmol,
            chain_id_to_crop_indices=chain_id_to_crop_indices,
            env_path=self.config.DB_config.template_db_path,
            n_templates=self.config.template_config.n_templates,
            rng=rng,
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
            
        return make_batch(
            cifmol=cifmol,
            msa=msa,
            templates=templates,
            atom_to_token_idx_map=atom_to_token_idx_map,
            token_to_residue_idx_map=token_to_residue_idx_map,
            token_type_override=token_type_override,
            rng=rng,
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
        self.seed = int(seed)
        
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

        worker_seed_rng = torch.Generator()
        worker_seed_rng.manual_seed(int(seed) + int(rank))

        params = {
            "shuffle": False,  # leave False when using a sampler
            "drop_last": False,  # override to True for train
            "num_workers": num_workers,
            "pin_memory": False,
            "generator": worker_seed_rng,
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