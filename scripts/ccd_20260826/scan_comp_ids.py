"""Collect every _chem_comp.id referenced by the entry CIFs behind cif.lmdb."""

import gzip
import json
import os
import sys
from multiprocessing import Pool

from mmcif_min import Malformed, Truncated, parse_category

PREFIX_CHARS = 512 * 1024


def scan(path):
    try:
        with gzip.open(path, "rt", errors="replace") as fh:
            text = fh.read(PREFIX_CHARS)
        lines = text.split("\n")
        try:
            tags, rows = parse_category(lines, "_chem_comp")
        except (Truncated, Malformed):
            # block ran past the prefix (or wrapped oddly): re-read in full
            with gzip.open(path, "rt", errors="replace") as fh:
                lines = fh.read().split("\n")
            tags, rows = parse_category(lines, "_chem_comp")
        if tags is None:
            return path, [], "no_chem_comp"
        if "id" not in tags:
            return path, [], "no_id_tag"
        col = tags.index("id")
        return path, sorted({r[col] for r in rows}), None
    except Exception as exc:  # noqa: BLE001
        return path, [], f"{type(exc).__name__}: {exc}"


def main():
    file_list = sys.argv[1]
    out_path = sys.argv[2]
    paths = [ln.strip() for ln in open(file_list) if ln.strip()]
    print(f"scanning {len(paths)} files", flush=True)

    per_entry = {}
    errors = {}
    done = 0
    with Pool(processes=14) as pool:
        for path, ids, err in pool.imap_unordered(scan, paths, chunksize=64):
            stem = os.path.basename(path).split(".")[0]
            if err:
                errors[stem] = err
            else:
                per_entry[stem] = ids
            done += 1
            if done % 20000 == 0:
                print(f"  {done}/{len(paths)}", flush=True)

    union = sorted({c for ids in per_entry.values() for c in ids})
    print(f"entries scanned ok: {len(per_entry)}, errors: {len(errors)}")
    print(f"distinct chem_comp ids: {len(union)}")
    json.dump(
        {"per_entry": per_entry, "union": union, "errors": errors},
        open(out_path, "w"),
    )


if __name__ == "__main__":
    main()
