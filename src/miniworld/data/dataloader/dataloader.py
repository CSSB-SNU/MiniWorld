"""BioMolData — multi-source Dataset with manifest and legacy-PDB compat modes.

Consumes either:
  * Manifest mode: items_path + resources_path (TSV/CSV) list all items and
    their per-chain LMDB resources.
  * Compat mode: BioMolDBV2Config.pdb (BioMolDBConfig, legacy edge_id_to_bias
    TSV) and/or BioMolDBV2Config.distillation_sources (per-source LMDBs).

This file only owns the Dataset (item catalog + DDP loader wiring). Per-item
preprocessing lives in preprocess.py; TSV parsing / weight math in sources.py;
LMDB loaders in loading.py.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel, field_validator
from torch.utils.data import DataLoader

from miniworld.configs.data import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TemplateConfig,
    TokenizerConfig,
)
from miniworld.data.io import extract_lmdb_keys
from miniworld.data.pipeline import Tokenizer

from .collate import bucketed_collate
from .loading import FragmentedCCDMolCache
from .preprocess import Preprocessor, WrongCroppingError
from .sampler import WeightedSampler
from .sources import (
    configured_source_weights,
    load_manifest_items,
    read_pdb_edge_items,
    source_balanced_weights,
)
from .types import (
    BioMolDBV2Config,
    DataRecord,
    DistillationSourceConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from miniworld.data.features import Batch
    from miniworld.data.mols import FragmentedCCDMol

__all__ = ["BioMolData", "WrongCroppingError"]


class BioMolData(torch.utils.data.Dataset):
    """Dataset for biomolecular complexes (multi-source manifest + legacy PDB)."""

    class BioMolConfig(BaseModel):
        """Configuration for BioMolData.

        DB_config is a BioMolDBV2Config, but the validator also accepts a
        legacy BioMolDBConfig (or its dict form — detected by the flat
        cif_db_path/edge_id_to_bias_path fields) and auto-wraps as
        ``BioMolDBV2Config(pdb=...)``.
        """

        crop_config: CropConfig = CropConfig()
        msa_config: MSAConfig = MSAConfig()
        template_config: TemplateConfig = TemplateConfig()
        DB_config: BioMolDBV2Config
        tokenizer_config: TokenizerConfig = TokenizerConfig()
        sampler_config: SamplerConfig = SamplerConfig()

        @field_validator("DB_config", mode="before")
        @classmethod
        def _coerce_legacy_db_config(cls, value: object) -> object:
            if isinstance(value, BioMolDBConfig):
                return BioMolDBV2Config(pdb=value)
            if isinstance(value, dict):
                v2_keys = {
                    "pdb",
                    "distillation_sources",
                    "items_path",
                    "resources_path",
                    "source_weights",
                }
                v1_keys = {
                    "cif_db_path",
                    "a3m_db_path",
                    "edge_id_to_bias_path",
                    "template_db_path",
                }
                if v1_keys & value.keys() and not v2_keys & value.keys():
                    return {"pdb": value}
            return value

    def __init__(self, config: BioMolConfig) -> None:
        super().__init__()
        self.config = config
        self.epoch: int = 0
        self.seed: int = 0
        self.tokenizer = Tokenizer(config=config.tokenizer_config)

        self.weights: list[float] = []
        self.items: list[DataRecord] = []

        self._load_items()
        self._load_ccd_preprocessed()

        self.preprocessor = Preprocessor(
            tokenizer=self.tokenizer,
            fragmented_ccd_mols=self.fragmented_ccd_mols,
            pdb_config=config.DB_config.pdb,
            crop_config=config.crop_config,
            msa_config=config.msa_config,
            template_config=config.template_config,
            tokenizer_config=config.tokenizer_config,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this dataset (also propagates to the tokenizer)."""
        self.epoch = epoch
        self.tokenizer.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.items)

    # -- item catalog ------------------------------------------------------

    def _load_items(self) -> None:
        self.items = []
        self.weights = []
        if self.config.DB_config.items_path and self.config.DB_config.resources_path:
            records, weights = load_manifest_items(
                self.config.DB_config.items_path,
                self.config.DB_config.resources_path,
                validation=self.config.DB_config.manifest_validation,
            )
            self.items.extend(records)
            self.weights.extend(weights)
        else:
            if self.config.DB_config.pdb is not None:
                self._append_pdb_items(self.config.DB_config.pdb)
            self._append_distillation_items()

        self.weights = source_balanced_weights(
            records=self.items,
            raw_weights=self.weights,
            source_weights=configured_source_weights(self.config.DB_config),
            default_source_weight=self.config.DB_config.default_source_weight,
        )

    def _append_pdb_items(self, pdb_config: BioMolDBConfig) -> None:
        edge_items, edge_weights = read_pdb_edge_items(
            pdb_config,
            self.config.sampler_config,
        )
        for bias, weight in zip(edge_items, edge_weights, strict=True):
            chain_ids = [bias.chain_id1] + ([bias.chain_id2] if bias.chain_id2 else [])
            self.items.append(
                DataRecord(
                    item_id=f"pdb:{bias.pdb_id}:{bias.assembly_id}:"
                    f"{bias.model_id}:{bias.alt_id}:{':'.join(chain_ids)}",
                    source="pdb",
                    record_id=bias.pdb_id,
                    cif_db_path=pdb_config.cif_db_path,
                    assembly_id=bias.assembly_id,
                    model_id=bias.model_id,
                    alt_id=bias.alt_id,
                    chain_ids=tuple(chain_ids),
                    feature_keys=(),
                    seq_ids=(),
                    msa_db_paths=(),
                    template_db_paths=(),
                    weight=weight,
                    weight_group="pdb",
                ),
            )
            self.weights.append(weight)

    def _append_distillation_items(self) -> None:
        for source in self.config.DB_config.distillation_sources:
            self._append_one_distillation_source(source)

    def _append_one_distillation_source(self, source: DistillationSourceConfig) -> None:
        keys = extract_lmdb_keys(source.cif_db_path, max_keys=source.max_items)
        if len(keys) == 0:
            return

        n_chains = len(source.chain_ids)
        per_item_weight = 1.0
        for key in keys:
            self.items.append(
                DataRecord(
                    item_id=f"{source.name}:{key}",
                    source=source.name,
                    record_id=key,
                    cif_db_path=source.cif_db_path,
                    assembly_id=source.assembly_id,
                    model_id=source.model_id,
                    alt_id=source.alt_id,
                    chain_ids=tuple(source.chain_ids),
                    feature_keys=(key,) * n_chains
                    if source.lookup_mode == "record_id"
                    else (),
                    seq_ids=(),
                    msa_db_paths=(tuple(source.a3m_db_paths),) * n_chains,
                    template_db_paths=(source.template_db_path,) * n_chains,
                    weight=per_item_weight,
                    weight_group=source.name,
                ),
            )
            self.weights.append(per_item_weight)

    def _load_ccd_preprocessed(self) -> None:
        ccd_path = self.config.DB_config.ccd_preprocessed_path
        if ccd_path is None and self.config.DB_config.pdb is not None:
            ccd_path = self.config.DB_config.pdb.ccd_preprocessed_path
        if ccd_path is None:
            msg = "CCD preprocessed path is not provided in the config."
            raise ValueError(msg)

        keys = extract_lmdb_keys(ccd_path)
        self.fragmented_ccd_mols: Mapping[str, dict[int, FragmentedCCDMol]] = (
            FragmentedCCDMolCache(ccd_path, keys)
        )

    # -- fetch / DDP loader -----------------------------------------------

    def __getitem__(self, idx: int) -> Batch:
        """Get a data sample by index (retries on WrongCroppingError)."""
        rng = np.random.default_rng()
        record = self.items[idx]

        while True:
            try:
                return self.preprocessor.process(record, rng=rng)
            except WrongCroppingError:  # noqa: PERF203
                idx = int(rng.integers(0, len(self)))
                record = self.items[idx]

    def get_item_by_id(
        self,
        pdb_id: str,
        assembly_id: str | None = None,
        model_id: str | None = None,
        alt_id: str | None = None,
        chain_ids: list[str] | None = None,
        crop_indices: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> Batch:
        """Compat: preprocess one item by PDB id (routes through the pdb source)."""
        if self.config.DB_config.pdb is None:
            msg = "get_item_by_id requires the pdb source to be configured."
            raise ValueError(msg)
        if not chain_ids:
            msg = "get_item_by_id requires chain_ids."
            raise ValueError(msg)
        resolved_assembly = assembly_id or "1"
        resolved_model = model_id or "1"
        resolved_alt = alt_id or "."
        record = DataRecord(
            item_id=f"pdb:{pdb_id}:{resolved_assembly}:{resolved_model}:{resolved_alt}",
            source="pdb",
            record_id=pdb_id,
            cif_db_path=self.config.DB_config.pdb.cif_db_path,
            assembly_id=resolved_assembly,
            model_id=resolved_model,
            alt_id=resolved_alt,
            chain_ids=tuple(chain_ids),
            feature_keys=(),  # resolved from cifmol.seq_id inside the pdb branch
            seq_ids=(),
            msa_db_paths=(),
            template_db_paths=(),
        )
        return self.preprocessor.process(
            record=record,
            crop_indices=crop_indices,
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

        sampler = WeightedSampler(
            dataset=self,
            weights=self.weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            replacement=self.config.DB_config.sample_with_replacement,
        )

        kwargs.pop("shuffle", None)
        kwargs.pop("world_size", None)
        kwargs.update({"sampler": sampler})
        if num_workers == 0:
            kwargs.pop("prefetch_factor", None)

        worker_seed_rng = torch.Generator()
        worker_seed_rng.manual_seed(int(seed) + int(rank))

        params = {
            "shuffle": False,
            "drop_last": False,
            "num_workers": num_workers,
            "pin_memory": False,
            "generator": worker_seed_rng,
            "multiprocessing_context": ("spawn" if num_workers > 0 else None),
            "collate_fn": functools.partial(
                bucketed_collate,
                bucket_msa_multiple=bucket_msa_multiple,
                bucket_token_multiple=bucket_token_multiple,
                bucket_atom_multiple=bucket_atom_multiple,
            ),
        }
        params.update(kwargs)
        return DataLoader(self, **params)
