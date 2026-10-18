import json
from pathlib import Path

import lmdb

from miniworld.data.io.load import load_raw_data


def _write(path: Path, items: dict[bytes, bytes]) -> None:
    env = lmdb.open(str(path), map_size=1 << 20)
    with env.begin(write=True) as txn:
        for k, v in items.items():
            txn.put(k, v)
    env.close()


def test_load_raw_data_reads_a_shard_collection(tmp_path: Path) -> None:
    root = tmp_path / "db.lmdb"
    (root / "shards").mkdir(parents=True)
    _write(root / "shards" / "000.lmdb", {b"a": b"1"})
    _write(root / "shards" / "001.lmdb", {b"b": b"2"})
    manifest = {"shards": [{"path": "shards/000.lmdb"}, {"path": "shards/001.lmdb"}]}
    (root / "shards.json").write_text(json.dumps(manifest))
    assert load_raw_data("a", root) == b"1"
    assert load_raw_data("b", root) == b"2"
    assert load_raw_data("c", root) is None


def test_load_raw_data_reads_a_plain_lmdb(tmp_path: Path) -> None:
    root = tmp_path / "plain.lmdb"
    root.mkdir()
    _write(root, {b"k": b"v"})
    assert load_raw_data("k", root) == b"v"
