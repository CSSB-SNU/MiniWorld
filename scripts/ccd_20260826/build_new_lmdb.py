"""Build preprocessed_CCD_20260826.lmdb.

Existing records are copied byte-for-byte from preprocessed_CCD.lmdb; every
chemical component referenced by cif.lmdb but absent from it is generated from
the wwPDB CCD with the pipeline in gen_record.py.
"""

import argparse
import json
import pickle
import sys
from multiprocessing import Pool

import lmdb

from biomol_codec import to_bytes
from gen_record import build_record

SRC = "/public_data/bsoohyuncd/preprocessed_CCD.lmdb"
DST = "/public_data/bsoohyuncd/preprocessed_CCD_20260826.lmdb"
MAP_SIZE = 2_000_000_000  # 2 GB; current payload is ~76 MB

_store = None


def _init(store_path):
    global _store
    _store = pickle.load(open(store_path, "rb"))


def make(cid):
    try:
        rec = _store.get(cid)
        if rec is None:
            return cid, None, "not in components.cif"
        if not rec["atoms"]:
            return cid, None, "no _chem_comp_atom table in CCD"
        return cid, to_bytes(build_record(cid, rec)), None
    except Exception as exc:  # noqa: BLE001
        return cid, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--store", default="ccd_store.pkl")
    ap.add_argument("--out", default=DST)
    args = ap.parse_args()

    missing = json.load(open("missing_ids.json"))
    print(f"components to generate: {len(missing)}", flush=True)

    built, failed = {}, {}
    with Pool(14, initializer=_init, initargs=(args.store,)) as pool:
        for i, (cid, blob, err) in enumerate(pool.imap_unordered(make, missing, chunksize=32)):
            if err:
                failed[cid] = err
            else:
                built[cid] = blob
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(missing)} generated={len(built)} failed={len(failed)}", flush=True)

    print(f"\ngenerated {len(built)}, failed {len(failed)}")
    for cid, err in sorted(failed.items()):
        print(f"  FAIL {cid}: {err}")
    json.dump(failed, open("generate_failures.json", "w"))

    if args.dry_run:
        total = sum(len(v) for v in built.values())
        print(f"\n[dry run] new payload {total/1e6:.1f} MB; nothing written")
        return

    src_env = lmdb.open(SRC, readonly=True, lock=False, readahead=False)
    dst_env = lmdb.open(args.out, map_size=MAP_SIZE)

    copied = 0
    with src_env.begin() as stx, dst_env.begin(write=True) as dtx:
        for key, value in stx.cursor():
            dtx.put(bytes(key), bytes(value))
            copied += 1
    print(f"copied {copied} existing records verbatim")

    with dst_env.begin(write=True) as dtx:
        for cid, blob in sorted(built.items()):
            dtx.put(cid.encode(), blob)
    print(f"added {len(built)} new records")

    dst_env.sync()
    with dst_env.begin() as dtx:
        n = dtx.stat()["entries"]
    dst_env.close()
    src_env.close()
    print(f"\n{args.out}: {n} keys")


if __name__ == "__main__":
    sys.exit(main())
