"""Move a catalog Arrow cache to another server's data paths instead of rebuilding it.

The catalog cache (``train_db.catalog_cache_path``) bakes every item's DB paths in and is keyed
by a fingerprint of the data config, so a config with other paths -- the same data on another
server -- ignores it and rebuilds the whole catalog. This rewrites the path prefixes in the
cached columns and stamps the fingerprint of the target config, which is what a build there
would have produced:

    python scripts/remap_catalog_cache.py SRC.arrow DST.arrow \\
        --from-data H100_default --to-data B200_default \\
        --map /home/psk6950/data/BioMol/=/NHNHOME/.../BioMol/ --map /home/psk6950/data/=/NHNHOME/.../data/

The source config must reproduce the cache's own fingerprint (else the cache is stale and is
refused); each path in the cache must match one ``--map`` prefix (first match wins).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.compute as pc
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_miniworld_distogram_train import DataConfig

from miniworld.data.dataloader.dataloader import _catalog_fingerprint

PATH_COLUMNS = ("cif_db_path", "msa_db_paths", "template_db_paths")


def _fingerprint(data_name: str) -> str:
    root = Path(__file__).resolve().parents[1] / "configs" / "miniworld" / "data"
    with initialize_config_dir(str(root), version_base=None):
        raw = compose(config_name=data_name)
    data = DataConfig.model_validate(raw)
    return _catalog_fingerprint(data.train_db, data.sampler)


def _remap_strings(arr: pa.Array, mapping: list[tuple[str, str]]) -> pa.Array:
    """Rewrite each non-null string's prefix; every one must match some prefix."""
    out = arr
    done = pc.is_null(arr)  # nulls stay null
    for old, new in mapping:
        hit = pc.and_(pc.invert(done), pc.fill_null(pc.starts_with(arr, old), fill_value=False))
        out = pc.if_else(hit, pc.binary_join_element_wise(new, pc.utf8_slice_codeunits(arr, len(old)), ""), out)
        done = pc.or_(done, hit)
    missed = pc.invert(done)
    if pc.sum(missed.cast(pa.int64())).as_py():
        bad = pc.filter(arr, missed)
        msg = f"{len(bad)} paths match no --map prefix, e.g. {bad[0].as_py()!r}"
        raise click.ClickException(msg)
    return out


def _remap(arr: pa.Array, mapping: list[tuple[str, str]]) -> pa.Array:
    """Strings, list<string> and list<list<string>> alike: remap the leaf values, keep the nesting."""
    if isinstance(arr, pa.ChunkedArray):
        return pa.chunked_array([_remap(c, mapping) for c in arr.chunks], type=arr.type)
    if pa.types.is_list(arr.type):
        return pa.ListArray.from_arrays(arr.offsets, _remap(arr.values, mapping), mask=arr.is_null())
    return _remap_strings(arr, mapping)


@click.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("dst", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--from-data", required=True, help="data config the cache was built with (e.g. H100_default)")
@click.option("--to-data", required=True, help="data config to stamp the cache for (e.g. B200_default)")
@click.option("--map", "maps", multiple=True, required=True, help="OLD_PREFIX=NEW_PREFIX (repeatable)")
def main(src: Path, dst: Path, from_data: str, to_data: str, maps: tuple[str, ...]) -> None:
    mapping = [tuple(m.split("=", 1)) for m in maps]
    with pa.memory_map(str(src)) as source:
        reader = pa.ipc.open_file(source)
        meta = dict(reader.schema.metadata or {})
        if meta.get(b"weight_kind") != b"raw":
            msg = "legacy (source-balanced) cache: rebuild it instead"
            raise click.ClickException(msg)
        have, want = meta[b"build_fingerprint"].decode(), _fingerprint(from_data)
        if have != want:
            msg = f"{src} is not the {from_data} cache (fingerprint {have[:12]} != {want[:12]})"
            raise click.ClickException(msg)
        meta[b"build_fingerprint"] = _fingerprint(to_data).encode()
        schema = reader.schema.with_metadata(meta)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        rows = 0
        with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                cols = [_remap(batch.column(n), mapping) if n in PATH_COLUMNS else batch.column(n)
                        for n in schema.names]
                writer.write_table(pa.table(cols, schema=schema))
                rows += batch.num_rows
        tmp.replace(dst)
    click.echo(f"{dst}: {rows} items, fingerprint {meta[b'build_fingerprint'].decode()[:12]} ({to_data})")


if __name__ == "__main__":
    main()
