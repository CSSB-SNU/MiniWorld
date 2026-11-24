"""Rewrite a BioMolDB cif LMDB into the per-assembly "MWF2" layout (decoded arrays identical to the old values).

Old value: one zstd frame holding a header (JSON: template + array byte lengths) and every assembly / model / alt of the entry
(biomol ``to_bytes``). Reading one assembly decompresses all of it and parses the whole header (1.5 MB of JSON for a typical entry).
New value: b"MWF2" + u32 index length + pickled index {assembly key: (offset, length)} + one zstd frame per assembly holding the pickled
decoded sub-tree (nested dict of numpy arrays) that ``load_bytes_entry(old_value, key)`` returns. A reader decompresses only the wanted
assembly and un-pickles it (no JSON, no per-array .npy parsing).

usage: rewrite_cif_lmdb.py rewrite OLD_LMDB NEW_LMDB [--limit N] [--procs P] [--level L]
       rewrite_cif_lmdb.py verify  OLD_LMDB NEW_LMDB [--n 2000] [--seed S]
The new LMDB is written only where NEW_LMDB points; the old one is opened read-only and never modified.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import random
import struct
import sys
import time

import lmdb
import numpy as np
from zstandard import ZstdCompressor, ZstdDecompressor

from miniworld.data.io.load import load_bytes_entry, split_entry_keys

MAGIC = b"MWF2"


def encode(old_value: bytes, level: int) -> bytes:
    keys = split_entry_keys(old_value)
    cctx = ZstdCompressor(level=level)
    frames, index, off = [], {}, 0
    for k in keys:
        sub = load_bytes_entry(old_value, k)
        fr = cctx.compress(pickle.dumps(sub, protocol=5))
        frames.append(fr)
        index[k] = (off, len(fr))
        off += len(fr)
    idx = pickle.dumps(index, protocol=4)
    return MAGIC + struct.pack("<I", len(idx)) + idx + b"".join(frames)


def _work(args):
    key, val, level = args
    return key, encode(val, level)


def rewrite(a):
    old = lmdb.open(a.old, readonly=True, lock=False, readahead=False, max_readers=64)
    os.makedirs(a.new, exist_ok=True)
    new = lmdb.open(a.new, map_size=int(a.map_gb * 2**30), writemap=False, metasync=False, sync=False, lock=True)
    n_old = old.stat()["entries"]
    n = len({l.split()[0] for l in open(a.keys_file) if l.strip()}) if a.keys_file else a.random if a.random else (min(n_old, a.limit) if a.limit else n_old)
    print(f"old entries {n_old}; rewriting {n}", flush=True)

    def gen():
        with old.begin(buffers=True) as txn:
            if a.keys_file:
                ks = sorted({l.split()[0] for l in open(a.keys_file) if l.strip()})
                for k in ks:
                    yield k.encode(), bytes(txn.get(k.encode())), a.level
                return
            if a.random:  # a random subset of keys (the first keys of the LMDB are the small, early PDB entries)
                allk = sorted(bytes(k) for k in txn.cursor().iternext(keys=True, values=False))
                for k in random.Random(0).sample(allk, a.random):
                    yield k, bytes(txn.get(k)), a.level
                return
            for i, (k, v) in enumerate(txn.cursor()):
                if i >= n:
                    return
                yield bytes(k), bytes(v), a.level

    t0 = time.time(); tot_old = tot_new = 0
    txn = new.begin(write=True)
    with mp.Pool(a.procs) as pool:
        for i, (k, nv) in enumerate(pool.imap(_work, gen(), chunksize=4)):
            txn.put(k, nv)
            tot_new += len(nv)
            if (i + 1) % 2000 == 0:
                txn.commit(); txn = new.begin(write=True)
                print(f"  {i+1}/{n}  {time.time()-t0:.0f}s  new {tot_new/2**30:.1f} GiB", flush=True)
    txn.commit(); new.sync()
    print(f"done: {n} entries in {time.time()-t0:.0f}s, new data {tot_new/2**30:.2f} GiB; new entries {new.stat()['entries']}", flush=True)


def tree_equal(a, b, path=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            tree_equal(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape, (path, a.dtype, getattr(b, "dtype", None))
        assert a.tobytes() == b.tobytes(), path  # byte-level
        assert a.flags.c_contiguous == b.flags.c_contiguous and a.flags.writeable == b.flags.writeable, (path, "flags")
    else:
        assert type(a) is type(b) and a == b, (path, a, b)


def verify(a):
    from miniworld.data.io.load import load_bytes_entry_v2

    old = lmdb.open(a.old, readonly=True, lock=False, readahead=False)
    new = lmdb.open(a.new, readonly=True, lock=False, readahead=False)
    n_old, n_new = old.stat()["entries"], new.stat()["entries"]
    print(f"entries old {n_old} new {n_new}")
    with old.begin(buffers=True) as to, new.begin(buffers=True) as tn:
        allk = [bytes(k) for k in to.cursor().iternext(keys=True, values=False)]
        newk = {bytes(k) for k in tn.cursor().iternext(keys=True, values=False)}
        assert set(allk) >= newk and (a.n_expect is None or len(newk) == a.n_expect), "key sets"
        if len(newk) == n_old:
            assert set(allk) == newk, "key sets differ"
        rng = random.Random(a.seed)
        sample = rng.sample(sorted(newk), min(a.n, len(newk)))
        t0 = time.time(); n_asm = 0
        for k in sample:
            ov, nv = bytes(to.get(k)), bytes(tn.get(k))
            for ak in split_entry_keys(ov):
                tree_equal(load_bytes_entry(ov, ak), load_bytes_entry_v2(nv, ak), f"{k!r}/{ak}")
                n_asm += 1
        print(f"verified {len(sample)} random keys / {n_asm} assemblies: decoded arrays identical (dtype, shape, bytes, flags); {time.time()-t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("rewrite"); r.add_argument("old"); r.add_argument("new"); r.add_argument("--limit", type=int, default=0); r.add_argument("--random", type=int, default=0); r.add_argument("--keys-file", default="")
    r.add_argument("--procs", type=int, default=30); r.add_argument("--level", type=int, default=6); r.add_argument("--map-gb", type=float, default=120)
    v = sp.add_parser("verify"); v.add_argument("old"); v.add_argument("new"); v.add_argument("--n", type=int, default=2000); v.add_argument("--seed", type=int, default=1)
    v.add_argument("--n-expect", type=int, default=None, dest="n_expect")
    a = p.parse_args()
    rewrite(a) if a.cmd == "rewrite" else verify(a)
