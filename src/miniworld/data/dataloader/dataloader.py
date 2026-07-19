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
import logging
from pathlib import Path
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
    SourceDBs,
    configured_source_weights,
    load_manifest_items,
    read_pdb_edge_items,
    read_train_items,
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

logger = logging.getLogger(__name__)

_CATALOG_CHUNK = 1_000_000  # rows per Arrow record batch when building the cache


def _write_catalog_arrow(
    path: Path, items: list[DataRecord], weights: list[float],
) -> None:
    """Serialize the item catalog to an Arrow IPC file (chunked to bound memory).

    Repetitive string columns (paths, source, chain_ids) dictionary-encode to almost
    nothing; the varying columns are record_id / feature_keys / weight. Written to a
    temp file then atomically renamed so a crash never leaves a half-written cache.
    """
    import pyarrow as pa

    schema = pa.schema([
        ("item_id", pa.string()),
        ("source", pa.string()),
        ("record_id", pa.string()),
        ("cif_db_path", pa.string()),
        ("assembly_id", pa.string()),
        ("model_id", pa.string()),
        ("alt_id", pa.string()),
        ("chain_ids", pa.list_(pa.string())),
        ("feature_keys", pa.list_(pa.string())),
        ("seq_ids", pa.list_(pa.string())),
        ("msa_db_paths", pa.list_(pa.list_(pa.string()))),
        ("template_db_paths", pa.list_(pa.string())),
        ("item_kind", pa.string()),
        ("weight_group", pa.string()),
        ("weight", pa.float64()),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
        for start in range(0, len(items), _CATALOG_CHUNK):
            chunk = items[start : start + _CATALOG_CHUNK]
            wchunk = weights[start : start + _CATALOG_CHUNK]
            # pa.table (not record_batch) so a column whose string values overflow
            # the 2 GiB offset limit within a chunk — feature_keys/msa_db_paths on
            # multi-million-row catalogs — is accepted as a ChunkedArray instead of
            # raising "Cannot convert ChunkedArray to Array".
            writer.write_table(
                pa.table(
                    [
                        pa.array([r.item_id for r in chunk]),
                        pa.array([r.source for r in chunk]),
                        pa.array([r.record_id for r in chunk]),
                        pa.array([str(r.cif_db_path) for r in chunk]),
                        pa.array([r.assembly_id for r in chunk]),
                        pa.array([r.model_id for r in chunk]),
                        pa.array([r.alt_id for r in chunk]),
                        pa.array([list(r.chain_ids) for r in chunk]),
                        pa.array([list(r.feature_keys) for r in chunk]),
                        pa.array([list(r.seq_ids) for r in chunk]),
                        pa.array(
                            [
                                [[str(p) for p in chain] for chain in r.msa_db_paths]
                                for r in chunk
                            ],
                        ),
                        pa.array(
                            [
                                [str(p) if p is not None else None
                                 for p in r.template_db_paths]
                                for r in chunk
                            ],
                        ),
                        pa.array([r.item_kind for r in chunk]),
                        pa.array([r.weight_group for r in chunk]),
                        pa.array([float(w) for w in wchunk]),
                    ],
                    schema=schema,
                ),
            )
    tmp.replace(path)


class _LazyArrowCatalog:
    """List-like view over an mmap'd Arrow catalog; builds one DataRecord per index.

    Holds the memory-mapped table (no Python-object materialization at load); indexing
    decodes a single row on demand. ``self.items[idx]`` in ``__getitem__`` is the only
    access pattern, so per-index decode cost is negligible next to the model step.
    """

    def __init__(self, table: object) -> None:
        self._t = table
        self._c = {name: table.column(name) for name in table.schema.names}

    def __len__(self) -> int:
        return self._t.num_rows

    def __getitem__(self, i: int) -> DataRecord:
        c = self._c
        msa = tuple(
            tuple(Path(p) for p in chain) for chain in c["msa_db_paths"][i].as_py()
        )
        tmpl = tuple(
            Path(p) if p is not None else None
            for p in c["template_db_paths"][i].as_py()
        )
        return DataRecord(
            item_id=c["item_id"][i].as_py(),
            source=c["source"][i].as_py(),
            record_id=c["record_id"][i].as_py(),
            cif_db_path=Path(c["cif_db_path"][i].as_py()),
            assembly_id=c["assembly_id"][i].as_py(),
            model_id=c["model_id"][i].as_py(),
            alt_id=c["alt_id"][i].as_py(),
            chain_ids=tuple(c["chain_ids"][i].as_py()),
            feature_keys=tuple(c["feature_keys"][i].as_py()),
            seq_ids=tuple(c["seq_ids"][i].as_py()),
            msa_db_paths=msa,
            template_db_paths=tmpl,
            weight=float(c["weight"][i].as_py()),
            item_kind=c["item_kind"][i].as_py(),
            weight_group=c["weight_group"][i].as_py(),
        )


def _load_catalog_arrow(path: Path) -> tuple[_LazyArrowCatalog, np.ndarray]:
    """mmap an Arrow catalog. Returns (lazy items view, balanced weights array)."""
    import pyarrow as pa

    source = pa.memory_map(str(path), "r")
    table = pa.ipc.open_file(source).read_all()
    weights = table.column("weight").to_numpy(zero_copy_only=False)
    return _LazyArrowCatalog(table), weights


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
        cache = self.config.DB_config.catalog_cache_path
        if cache is not None and Path(cache).exists():
            # mmap the prebuilt Arrow catalog: instant, zero Python-object build,
            # shared across DDP ranks via the OS page cache.
            self.items, self.weights = _load_catalog_arrow(Path(cache))
            logger.info(
                "loaded catalog from Arrow cache %s (%d items)", cache, len(self.items),
            )
            return

        self.items = []
        self.weights = []
        if self.config.DB_config.train_item_path:
            records, weights = read_train_items(
                self.config.DB_config.train_item_path,
                self._build_source_dbs(),
                self.config.sampler_config,
            )
            self.items.extend(records)
            self.weights.extend(weights)
        elif self.config.DB_config.items_path and self.config.DB_config.resources_path:
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

        if cache is not None:
            _write_catalog_arrow(Path(cache), self.items, self.weights)
            logger.info(
                "wrote catalog Arrow cache %s (%d items) — future inits mmap it",
                cache, len(self.items),
            )

    def _build_source_dbs(self) -> dict[str, SourceDBs]:
        """Map each source name to its cif/msa/template LMDBs for train_item rows.

        pdb comes from ``DB_config.pdb`` (a3m keyed by full seq_id, template by
        ``{RECORD}_{chain}``); each distillation source from its own entry.
        """
        source_dbs: dict[str, SourceDBs] = {}
        pdb = self.config.DB_config.pdb
        if pdb is not None:
            pdb_msa = (pdb.a3m_db_path,)
            if pdb.a3m_rna_db_path is not None:
                pdb_msa = (pdb.a3m_db_path, pdb.a3m_rna_db_path)
            source_dbs["pdb"] = SourceDBs(
                cif_db_path=pdb.cif_db_path,
                msa_db_paths=pdb_msa,
                template_db_path=pdb.template_db_path,
            )
        for source in self.config.DB_config.distillation_sources:
            source_dbs[source.name] = SourceDBs(
                cif_db_path=source.cif_db_path,
                msa_db_paths=tuple(source.a3m_db_paths),
                template_db_path=source.template_db_path,
            )
        return source_dbs

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
        # Enumerate the record list (cif keys) + resolve each key's MSA shard by
        # scanning the a3m shards once. This whole enumeration is only ever paid on a
        # cold build: the assembled catalog is cached to an Arrow file and mmap-loaded
        # on subsequent __init__ (see _load_items / catalog_cache_path).
        keys = extract_lmdb_keys(source.cif_db_path, max_keys=source.max_items)
        if len(keys) == 0:
            return
        shards = tuple(source.a3m_db_paths)
        if len(shards) <= 1:
            # Single-shard fast path: skip enumerating the MSA shard's keys (which
            # for the 2.7T msa_long_d2k lmdb scans millions of keys over the shared
            # FS — the dominant cold-build cost). Assign the one shard to every
            # record; a record whose key is absent from the shard falls back to the
            # query sequence at load time (_load_a3m_from_paths -> None ->
            # MSA.from_query), which is exactly what leaving msa_paths_for_key=()
            # would have done — so the catalog is unchanged, just built faster.
            single: tuple[Path, ...] = (shards[0],) if shards else ()
            for key in keys:
                self._append_distillation_record(
                    source, key, source.cif_db_path, single, source.template_db_path,
                )
            return
        # Multi-shard: resolve each key's MSA shard by enumerating the shards once.
        msa_shard_by_key: dict[str, Path] = {}
        for shard in shards:
            for shard_key in extract_lmdb_keys(shard):
                msa_shard_by_key.setdefault(shard_key, shard)
        for key in keys:
            resolved_shard = msa_shard_by_key.get(key)
            msa_paths_for_key: tuple[Path, ...] = (
                (resolved_shard,) if resolved_shard is not None else ()
            )
            self._append_distillation_record(
                source, key, source.cif_db_path, msa_paths_for_key, source.template_db_path,
            )

    def _append_distillation_record(
        self,
        source: DistillationSourceConfig,
        key: str,
        cif_db_path: Path,
        msa_paths_for_key: tuple[Path, ...],
        template_db_path: Path | None,
    ) -> None:
        n_chains = len(source.chain_ids)
        per_item_weight = 1.0
        self.items.append(
            DataRecord(
                item_id=f"{source.name}:{key}",
                source=source.name,
                record_id=key,
                cif_db_path=cif_db_path,
                assembly_id=source.assembly_id,
                model_id=source.model_id,
                alt_id=source.alt_id,
                chain_ids=tuple(source.chain_ids),
                feature_keys=(key,) * n_chains
                if source.lookup_mode == "record_id"
                else (),
                seq_ids=(),
                msa_db_paths=(msa_paths_for_key,) * n_chains,
                template_db_paths=(template_db_path,) * n_chains,
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
        num_samples_per_rank: int | None = None,
        bucket_msa_multiple: int | None = None,
        bucket_token_multiple: int | None = None,
        bucket_atom_multiple: int | None = None,
        **kwargs: object,
    ) -> DataLoader:
        """Create a distributed DataLoader with WeightedSampler.

        ``num_samples_per_rank`` pins the epoch length to what the training
        loop actually consumes (e.g. ``train_item // world_size``); leaving it
        None keeps the legacy behavior of iterating the full catalog per rank
        (heavy for multi-source manifests).
        """
        self.seed = int(seed)

        sampler = WeightedSampler(
            dataset=self,
            weights=self.weights,
            num_samples=num_samples_per_rank,
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
