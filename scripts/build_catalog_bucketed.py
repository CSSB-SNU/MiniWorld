"""Turn a ``raw`` catalog (the sampler weights baked into every row) into a sampler-independent ``bucket_unit`` catalog.

A ``raw`` catalog stores, per row, ``sampler.<bucket> / clusters_of_the_bucket / rows_of_the_cluster`` (pdb) or ``1 / rows_of_the_cluster``
(the other sources), so every change of ``sampler.antibody_protein`` (or any bucket) needs another 9 GB file. The ``bucket_unit`` catalog
stores the bucket of every row (``bucket``, index into the schema's ``bucket_names``) and the same weight WITHOUT the sampler factor
(``weight``); the loader (``_load_catalog_arrow(path, sampler_config)``) multiplies the sampler weight of the bucket back in. The item columns
are untouched, so the DB paths stay baked in: build one file per data location (shared disk, node-local NVMe).

    python scripts/build_catalog_bucketed.py SRC.arrow DST.arrow --train-item train_item_20210930.tsv --src-config phase1a_distogram_medium_v240

The source config must reproduce SRC's own fingerprint. Every row's bucket and unit weight are recomputed from ``train_item`` and the result
is checked against SRC: ``unit x sampler weight`` must equal SRC's weight (relative error < 1e-9) in every row, which also proves the row order
and the bucket assignment. The fingerprint stamped on DST is the DB-only one of ``--to-config`` (default: ``--src-config``).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_miniworld_distogram_train import Config

from miniworld.data.dataloader.dataloader import (
    CATALOG_BUCKET_UNIT,
    CATALOG_NO_BUCKET,
    _catalog_fingerprint,
    _catalog_fingerprint_unit,
)
from miniworld.data.dataloader.sources import _PDB_INTERFACE_TYPES, _pdb_edge_type

COLUMNS = ("source", "cluster1", "cluster2", "pdb_id", "assembly_id", "model_id", "alt_id", "chain_id1", "chain_id2")


def _data(config_name: str):
    root = Path(__file__).resolve().parents[1] / "configs" / "miniworld"
    with initialize_config_dir(str(root), version_base=None):
        return Config.model_validate(compose(config_name=config_name)).data


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


@click.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("dst", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--train-item", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--src-config", required=True, help="config name SRC was built with (its fingerprint must match SRC's)")
@click.option("--to-config", default=None, help="config name the new catalog is for (DB paths decide the fingerprint); default --src-config")
def main(src: Path, dst: Path, train_item: Path, src_config: str, to_config: str | None) -> None:
    t0 = time.time()
    src_data, to_data = _data(src_config), _data(to_config or src_config)
    table = pa.ipc.open_file(pa.memory_map(str(src), "r")).read_all()
    meta = dict(table.schema.metadata or {})
    if meta.get(b"weight_kind") != b"raw":
        msg = f"{src} is not a raw catalog (weight_kind={meta.get(b'weight_kind')!r})"
        raise SystemExit(msg)
    want = _catalog_fingerprint(src_data.train_db, src_data.sampler)
    have = (meta.get(b"build_fingerprint") or b"").decode()
    if have != want:
        msg = f"--src-config {src_config} does not reproduce the fingerprint of {src} ({want[:12]} vs {have[:12]})"
        raise SystemExit(msg)
    sources = pc.unique(table.column("source")).to_pylist()
    _log(f"{src}: {table.num_rows} rows, sources {sources}", t0)

    tsv = pacsv.read_csv(
        train_item,
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in COLUMNS}),
    )
    tsv = tsv.filter(pc.is_in(tsv.column("source"), value_set=pa.array(sources)))
    if tsv.num_rows != table.num_rows:
        msg = f"train_item has {tsv.num_rows} rows of the catalog's sources, the catalog {table.num_rows}"
        raise SystemExit(msg)
    _log(f"train_item read and filtered: {tsv.num_rows} rows", t0)

    # row alignment: every 97th row (and both ends) must give the catalog's item_id
    step = np.unique(np.concatenate([np.arange(0, tsv.num_rows, 97), np.arange(max(tsv.num_rows - 2000, 0), tsv.num_rows)]))
    sub = tsv.take(pa.array(step)).to_pydict()
    ids = table.column("item_id").take(pa.array(step)).to_pylist()
    for i, item_id in enumerate(ids):
        s, rid = sub["source"][i], sub["pdb_id"][i]
        rid = rid.lower() if s == "pdb" else rid
        chains = sub["chain_id1"][i] if sub["cluster2"][i] == "None" else f"{sub['chain_id1'][i]}:{sub['chain_id2'][i]}"
        got = f"{s}:{rid}:{sub['assembly_id'][i]}:{sub['model_id'][i]}:{sub['alt_id'][i]}:{chains}"
        if got != item_id:
            msg = f"row {step[i]}: train_item gives {got!r}, the catalog {item_id!r}: the order differs"
            raise SystemExit(msg)
    _log(f"row alignment checked on {len(step)} rows", t0)

    # the group of a row = (source, edge id); the weight is computed inside the group, exactly as read_train_items does
    c1, c2 = tsv.column("cluster1"), tsv.column("cluster2")
    edge = pc.if_else(pc.equal(c2, "None"), c1, pc.binary_join_element_wise(c1, c2, "_"))
    key = pc.binary_join_element_wise(tsv.column("source"), edge, "|").cast(pa.large_string()).combine_chunks()
    del tsv
    enc = pc.dictionary_encode(key)
    del key
    codes = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    uniq = enc.dictionary.to_pylist()
    del enc
    _log(f"{len(uniq)} groups (source, cluster edge)", t0)

    names = [*_PDB_INTERFACE_TYPES, CATALOG_NO_BUCKET]
    code_of = {n: i for i, n in enumerate(names)}
    is_pdb = np.zeros(len(uniq), dtype=bool)
    ubucket = np.full(len(uniq), code_of[CATALOG_NO_BUCKET], dtype=np.int8)
    for i, k in enumerate(uniq):
        if k.startswith("pdb|"):
            is_pdb[i] = True
            ubucket[i] = code_of[_pdb_edge_type(k[4:])]
    del uniq
    rows_of_group = np.bincount(codes, minlength=len(is_pdb))
    clusters_of_bucket = np.bincount(ubucket[is_pdb], minlength=len(names))
    divisor = rows_of_group * np.where(is_pdb, clusters_of_bucket[ubucket], 1)
    unit = (1.0 / divisor)[codes]
    bucket = ubucket[codes]
    del codes
    _log("clusters per bucket: " + ", ".join(f"{n} {clusters_of_bucket[code_of[n]]}" for n in _PDB_INTERFACE_TYPES), t0)

    lut = np.array([1.0 if n == CATALOG_NO_BUCKET else float(getattr(src_data.sampler, n)) for n in names])
    w_src = table.column("weight").to_numpy(zero_copy_only=False)
    expected = unit * lut[bucket]
    rel = np.abs(w_src - expected) / np.maximum(np.abs(w_src), 1e-300)
    _log(f"check against SRC: max relative error {rel.max():.3e}, rows differing by more than 1e-9: {int((rel > 1e-9).sum())}", t0)
    if rel.max() > 1e-9:
        bad = int(np.argmax(rel))
        msg = f"unit x sampler weight != the catalog weight (row {bad}: {expected[bad]!r} vs {w_src[bad]!r}, bucket {names[bucket[bad]]})"
        raise SystemExit(msg)

    out = table.set_column(
        table.schema.get_field_index("weight"), pa.field("weight", pa.float64()), pa.chunked_array([pa.array(unit)])
    ).append_column(pa.field("bucket", pa.int8()), pa.chunked_array([pa.array(bucket)]))
    meta.update(
        {
            b"weight_kind": CATALOG_BUCKET_UNIT.encode(),
            b"build_fingerprint": _catalog_fingerprint_unit(to_data.train_db).encode(),
            b"bucket_names": json.dumps(names).encode(),
            b"derived_from": str(src).encode(),
        }
    )
    out = out.replace_schema_metadata(meta)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, out.schema) as writer:
        writer.write_table(out, max_chunksize=1_000_000)
    tmp.replace(dst)
    _log(f"wrote {dst} ({dst.stat().st_size / 2**30:.2f} GiB)", t0)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
