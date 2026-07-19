"""Data types for the BioMol dataset: pydantic configs + resolved records."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lmdb
from pydantic import BaseModel

from miniworld.configs.data import BioMolDBConfig


# ---------------------------------------------------------------------------
# Pydantic configs (DB sources)
# ---------------------------------------------------------------------------


class DistillationSourceConfig(BaseModel):
    """Configuration for one legacy-style distillation LMDB source."""

    name: str
    cif_db_path: Path
    a3m_db_paths: list[Path] = []
    template_db_path: Path | None = None
    weight: float = 1.0
    max_items: int | None = None
    assembly_id: str = "1"
    model_id: str = "1"
    alt_id: str = "."
    chain_ids: list[str] = ["1"]
    lookup_mode: Literal["record_id", "seq_id"] = "record_id"


class BioMolDBV2Config(BaseModel):
    """Database configuration for manifest-driven mixed training."""

    items_path: Path | None = None
    resources_path: Path | None = None
    ccd_preprocessed_path: Path | None = None

    # Unified location authority (lmdb ResourceLocator) for train_item mode: the
    # single source of cif/msa/template LMDB paths. cif/template resolve per source
    # (baked onto each record); the exact msa shard resolves by seq_id at runtime,
    # so the loader never scans the shard list. Built by build_resources_index.py.
    resources_index_path: Path | None = None
    # Override the index's stored default root (portability across mounts).
    resources_base: str | None = None

    # Unified edge_node-style item list (source-tagged). When set, the catalog is
    # built from this file: each row routes to its source's LMDBs (pdb + the
    # distillation_sources below) and gets an AF3-style raw weight (pdb 3-tier;
    # distillation cluster-uniform). msa/template keys resolve from the CIF.
    train_item_path: Path | None = None

    # Memory-mapped Arrow catalog cache (HuggingFace-datasets / MDS-style). The full
    # item catalog (all sources) is enumerated ONCE and written here as an Arrow IPC
    # file; every later __init__ mmaps it (zero-copy, instant, shared across DDP ranks
    # via the OS page cache) and builds a DataRecord lazily per __getitem__ — so init
    # never re-scans shard LMDBs nor materializes ~31M Python objects. Delete the file
    # to force a rebuild when the sources change.
    catalog_cache_path: Path | None = None

    # Source/db sampling probabilities. PDB still uses SamplerConfig inside this mass.
    source_weights: dict[str, float] = {}
    default_source_weight: float = 1.0
    sample_with_replacement: bool = True

    # Manifest validation at dataset __init__:
    #   "off"    skip (fastest, existing behavior)
    #   "paths"  check every referenced LMDB file exists (cheap, ~ms)
    #   "keys"   also scan every referenced shard and drop items whose record_id
    #            is missing from its cif LMDB. MSA/template misses are reported
    #            but not dropped (runtime fallback covers them). Expensive
    #            (scans full key lists — potentially minutes on large datasets).
    manifest_validation: Literal["off", "paths", "keys"] = "off"

    # Compatibility path while manifest generation is being introduced.
    pdb: BioMolDBConfig | None = None
    distillation_sources: list[DistillationSourceConfig] = []


# ---------------------------------------------------------------------------
# Resolved sampling records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataRecord:
    """Resolved sampling item used by BioMolData."""

    item_id: str
    source: str
    record_id: str
    cif_db_path: Path
    assembly_id: str
    model_id: str
    alt_id: str
    chain_ids: tuple[str, ...]
    feature_keys: tuple[str, ...]
    seq_ids: tuple[str, ...]
    msa_db_paths: tuple[tuple[Path, ...], ...]
    template_db_paths: tuple[Path | None, ...]
    weight: float = 1.0
    item_kind: str = "unknown"
    weight_group: str = "default"


@dataclass(frozen=True)
class ResourceIndex:
    """Lookup index from feature/record keys to resource LMDB paths."""

    cif: dict[str, Path]
    msa: dict[str, tuple[Path, ...]]
    template: dict[str, Path]


class ResourceLocator:
    """LMDB-backed location authority for the unified train_item dataloader.

    Single mechanism behind cif/msa/template path resolution (built by
    ``scripts/build_resources_index.py``). Storage is minimal:

    * cif / template / single-shard msa are ONE lmdb per source -> resolved from
      the small ``__meta__`` json (source -> base-relative paths). No per-key rows.
    * multi-shard msa (distillation_long's 80 seqid shards) is the only thing that
      needs a per-key index; ``{source}{sep}{seq_id} -> shard_idx`` rows resolve
      the exact shard so the runtime never scans all shards.

    Paths are stored relative to ``base`` (a default root the config can override),
    so the index is portable across mounts. The lmdb env is opened lazily per
    process (survives fork AND spawn) and shared between workers via the OS page
    cache, so there is no per-worker RAM blow-up and no TSV parse at spawn.
    """

    def __init__(self, index_path: Path, base_override: str | Path | None = None):
        self.index_path = Path(index_path)
        env = lmdb.open(
            str(self.index_path), readonly=True, lock=False,
            readahead=False, max_readers=4096,
        )
        with env.begin() as txn:
            raw = txn.get(b"__meta__")
        env.close()
        if raw is None:
            msg = f"resources index has no __meta__: {self.index_path}"
            raise ValueError(msg)
        meta = json.loads(raw.decode())
        self.base = Path(base_override) if base_override else Path(meta["base"])
        self.sep = meta.get("key_sep", "|")
        self.sources: dict[str, dict] = meta["sources"]
        self._env: lmdb.Environment | None = None

    # lmdb.Environment is not picklable; drop it so spawn workers reopen lazily.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def _abs(self, rel: str | None) -> Path | None:
        if rel is None:
            return None
        return Path(rel) if rel.startswith("/") else self.base / rel

    def _env_(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                str(self.index_path), readonly=True, lock=False,
                readahead=False, max_readers=4096,
            )
        return self._env

    def has_source(self, source: str) -> bool:
        return source in self.sources

    def cif_path(self, source: str) -> Path | None:
        return self._abs(self.sources[source].get("cif"))

    def template_path(self, source: str) -> Path | None:
        return self._abs(self.sources[source].get("template"))

    def msa_paths_all(self, source: str) -> tuple[Path, ...]:
        """Every msa shard for a source (fallback / non-sharded)."""
        info = self.sources.get(source, {})
        return tuple(p for p in (self._abs(x) for x in info.get("msa", [])) if p)

    def msa_paths_for(self, source: str, seq_id: str) -> tuple[Path, ...]:
        """Exact shard(s) holding ``seq_id``.

        Non-sharded sources return their single msa path. Sharded sources look up
        the seq_id -> shard_idx index (O(1)); on a miss they fall back to all
        shards so a missing index row degrades to the old scan rather than failing.
        """
        info = self.sources.get(source)
        if info is None:
            return ()
        if not info.get("sharded_msa"):
            return self.msa_paths_all(source)
        with self._env_().begin() as txn:
            v = txn.get(f"{source}{self.sep}{seq_id}".encode())
        if v is None:
            return self.msa_paths_all(source)
        idx = struct.unpack("<H", v)[0]
        msa = info.get("msa", [])
        if idx >= len(msa):
            return self.msa_paths_all(source)
        p = self._abs(msa[idx])
        return (p,) if p else ()


# ---------------------------------------------------------------------------
# Chain-indexed accessors on DataRecord
# ---------------------------------------------------------------------------


def feature_index(record: DataRecord, chain_id: str) -> int:
    """Return the feature-key slot corresponding to a cropped chain."""
    try:
        return record.chain_ids.index(chain_id)
    except ValueError:
        if len(record.feature_keys) == 1:
            return 0
        msg = f"Chain '{chain_id}' is not present in item {record.item_id}."
        raise KeyError(msg) from None


def feature_key(record: DataRecord, chain_id: str) -> str:
    return record.feature_keys[feature_index(record, chain_id)]


def msa_paths(record: DataRecord, chain_id: str) -> tuple[Path, ...]:
    return record.msa_db_paths[feature_index(record, chain_id)]


def template_path(record: DataRecord, chain_id: str) -> Path | None:
    return record.template_db_paths[feature_index(record, chain_id)]
