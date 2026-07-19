"""Data types for the BioMol dataset: pydantic configs + resolved records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
