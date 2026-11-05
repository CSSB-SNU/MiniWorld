"""A ``bucket_unit`` catalog: sampler-independent weights, the sampler weight of the bucket is multiplied in on load."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from miniworld.configs.data import SamplerConfig
from miniworld.data.dataloader.dataloader import (
    CATALOG_BUCKET_UNIT,
    CATALOG_NO_BUCKET,
    _catalog_fingerprint,
    _catalog_fingerprint_unit,
    _catalog_weight_kind,
    _load_catalog_arrow,
)
from miniworld.data.dataloader.sources import _PDB_INTERFACE_TYPES
from miniworld.data.dataloader.types import BioMolDBV2Config

NAMES = [*_PDB_INTERFACE_TYPES, CATALOG_NO_BUCKET]


def _write(path: Path, kind: str, weights: list[float], buckets: list[int] | None, fingerprint: str = "fp") -> None:
    cols = {
        "item_id": [f"i{n}" for n in range(len(weights))],
        "source": ["pdb"] * len(weights),
        "weight": weights,
    }
    meta = {b"weight_kind": kind.encode(), b"build_fingerprint": fingerprint.encode()}
    if buckets is not None:
        cols["bucket"] = pa.array(buckets, pa.int8())
        meta[b"bucket_names"] = json.dumps(NAMES).encode()
    table = pa.table(cols).replace_schema_metadata(meta)
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)


def test_bucket_unit_weights_follow_the_sampler(tmp_path):
    ab, ppi, none = NAMES.index("antibody_protein"), NAMES.index("protein_protein"), NAMES.index(CATALOG_NO_BUCKET)
    path = tmp_path / "catalog.arrow"
    _write(path, CATALOG_BUCKET_UNIT, [0.5, 0.25, 0.125, 0.75], [ab, ab, ppi, none])
    assert _catalog_weight_kind(path) == CATALOG_BUCKET_UNIT

    _, w10, _, is_raw, _ = _load_catalog_arrow(path, SamplerConfig(antibody_protein=10.0, protein_protein=3.0))
    assert is_raw
    np.testing.assert_allclose(w10, [5.0, 2.5, 0.375, 0.75])  # unit x sampler weight of the bucket, 1.0 for a row without a bucket
    _, w25, _, _, _ = _load_catalog_arrow(path, SamplerConfig(antibody_protein=2.5, protein_protein=3.0))
    np.testing.assert_allclose(w25, [1.25, 0.625, 0.375, 0.75])  # only the antibody_protein rows change, by 2.5 / 10


def test_bucket_unit_needs_the_sampler(tmp_path):
    path = tmp_path / "catalog.arrow"
    _write(path, CATALOG_BUCKET_UNIT, [1.0], [0])
    with pytest.raises(ValueError, match="sampler config"):
        _load_catalog_arrow(path)


def test_raw_catalog_is_read_as_before(tmp_path):
    path = tmp_path / "catalog.arrow"
    _write(path, "raw", [0.5, 2.0], None)
    assert _catalog_weight_kind(path) == "raw"
    _, w, _, is_raw, fp = _load_catalog_arrow(path, SamplerConfig(antibody_protein=99.0))  # the sampler is ignored for a raw file
    assert is_raw
    assert fp == "fp"
    np.testing.assert_allclose(w, [0.5, 2.0])
    legacy = tmp_path / "legacy.arrow"
    _write(legacy, "", [1.0], None)
    assert not _load_catalog_arrow(legacy)[3]
    assert _catalog_weight_kind(legacy) == ""


def test_unit_fingerprint_ignores_the_sampler_and_keeps_the_paths():
    db = BioMolDBV2Config()
    assert _catalog_fingerprint_unit(db) == _catalog_fingerprint_unit(BioMolDBV2Config(antibody_seq_id_path=Path("/x/ab.tsv")))
    assert _catalog_fingerprint_unit(db) != _catalog_fingerprint(db, SamplerConfig())
    assert _catalog_fingerprint(db, SamplerConfig(antibody_protein=2.5)) != _catalog_fingerprint(db, SamplerConfig(antibody_protein=10.0))
    other = BioMolDBV2Config(catalog_cache_path=Path("/x/other.arrow"))
    assert _catalog_fingerprint_unit(db) == _catalog_fingerprint_unit(other)  # the cache location is not part of it
