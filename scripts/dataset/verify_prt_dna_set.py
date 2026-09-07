#!/usr/bin/env python3
"""Check a ``*_onlyPrtDna`` subset against its source and its edge_node TSV.

Runs over every entry and every row, not a sample.

Aid removal -- the point of this check is that ONLY aid molecules left:

  1. no residue in the output carries a CRYSTALLIZATION_AID_CHEMCOMPS code;
  2. for every sub-entry kept, each of the source's chains is accounted for:
     a chain whose residues were ALL aids must be gone, a chain with SOME aid
     residues must have shrunk by exactly those residues, and a chain with NO
     aid residues must survive with an identical residue and atom count.  So
     nothing but the aid molecules was dropped.

Structure and bookkeeping:

  3. every stored blob decodes and every sub-entry carries a chain table;
  4. every kept sub-entry has >=1 protein chain AND >=1 DNA chain, and no RNA
     (``cR``) or DNA/RNA hybrid (``cN``) chain;
  5. every TSV row is a DNA-vs-protein pair, its ``{assembly}_{model}_{alt}``
     sub-entry exists, and both chain ids exist in it with the cluster ids the
     row claims (compared as an unordered pair -- see the note below);
  6. every stored key is referenced by at least one row and vice versa;
  7. a blob copied verbatim is byte-identical to the source; a re-encoded blob
     holds a subset of the source's sub-entries.

Usage
-----
    /home/bsoohyuncd/software/StructCooker/.pixi/envs/default/bin/python \
        scripts/verify_prt_dna_set.py [--split train|valid2]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
from biomol.core.utils import load_bytes
from structcooker.mols import CIFMolAttached

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_prt_dna_set import (  # noqa: E402
    AID_CODES,
    DNA_TYPE,
    FORBIDDEN_TYPES,
    PROTEIN_TYPES,
    SPLITS,
    TSV_HEADER,
    is_protein_dna_pair,
)

MAX_REPORTED = 12


def chain_profile(item: dict) -> dict[str, dict]:
    """chain_id -> {cluster, type, n_res, n_atom, residues} for one sub-entry."""
    mol = CIFMolAttached.from_dict(item["cifmol_attached_dict"])
    chain_id = [str(x) for x in np.asarray(mol.chains.chain_id.value)]
    cluster = [str(x) for x in np.asarray(mol.chains.cluster_id.value)]
    ccd = np.asarray(mol.residues.chem_comp_id.value)
    res_to_chain = np.asarray(mol.index_table.res_to_chain)
    atom_to_chain = res_to_chain[np.asarray(mol.index_table.atom_to_res)]
    out = {}
    for i, cid in enumerate(chain_id):
        sel = res_to_chain == i
        out[cid] = {
            "cluster": cluster[i],
            "type": cluster[i][1],
            "n_res": int(sel.sum()),
            "n_atom": int((atom_to_chain == i).sum()),
            "residues": [str(c) for c in ccd[sel]],
        }
    return out


def verify(name: str, paths: dict[str, Path]) -> int:  # noqa: C901, PLR0912, PLR0915
    """Verify one split; return the number of problems found."""
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    for label in ("out_lmdb", "out_tsv", "in_lmdb"):
        print(f"{label:10s}: {paths[label]}")

    problems: list[str] = []
    counts: Counter[str] = Counter()
    aid_hits: Counter[str] = Counter()
    aids = set(AID_CODES)

    def fail(msg: str) -> None:
        counts["problems"] += 1
        if len(problems) < MAX_REPORTED:
            problems.append(msg)

    rows: list[tuple[str, ...]] = []
    with paths["out_tsv"].open() as fh:
        if next(fh).rstrip("\n") != TSV_HEADER.rstrip("\n"):
            fail("tsv header mismatch")
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped:
                rows.append(tuple(stripped.split("\t")))
    print(f"\nrows in tsv     : {len(rows)}")

    out_env = lmdb.open(str(paths["out_lmdb"]), readonly=True, lock=False, max_readers=2048)
    src_env = lmdb.open(str(paths["in_lmdb"]), readonly=True, lock=False, max_readers=2048)
    stored = out_env.stat()["entries"]
    print(f"keys in lmdb    : {stored}")

    rows_by_pdb: dict[str, list[tuple[str, ...]]] = {}
    for row in rows:
        if len(row) != 8:
            fail(f"row has {len(row)} columns: {row}")
            continue
        if not is_protein_dna_pair(row[0], row[1]):
            fail(f"row is not a DNA-protein pair: {row}")
        rows_by_pdb.setdefault(row[2].lower(), []).append(row)
    print(f"distinct pdb ids: {len(rows_by_pdb)}")

    started = time.time()
    seen: set[str] = set()

    with out_env.begin(buffers=True) as out_txn, src_env.begin(buffers=True) as src_txn:
        for n_done, (raw_key, raw_val) in enumerate(out_txn.cursor(), start=1):
            pdb = bytes(raw_key).decode()
            blob = bytes(raw_val)
            seen.add(pdb)
            if pdb not in rows_by_pdb:
                fail(f"{pdb}: stored but no row references it")

            try:
                entry = load_bytes(blob)
            except Exception as exc:  # noqa: BLE001
                fail(f"{pdb}: blob does not decode: {exc}")
                continue

            src_raw = src_txn.get(pdb.encode())
            if src_raw is None:
                fail(f"{pdb}: absent from source lmdb")
                continue
            src_blob = bytes(src_raw)
            if src_blob == blob:
                counts["blob_verbatim"] += 1
            else:
                counts["blob_reencoded"] += 1
            src_entry = load_bytes(src_blob)

            extra = set(entry) - set(src_entry)
            if extra:
                fail(f"{pdb}: sub-entries not in source: {sorted(extra)}")

            profiles: dict[str, dict[str, dict]] = {}
            for asm_key, item in entry.items():
                counts["subentries"] += 1
                try:
                    prof = chain_profile(item)
                except Exception as exc:  # noqa: BLE001
                    fail(f"{pdb}/{asm_key}: chain table unreadable: {exc}")
                    continue
                profiles[asm_key] = prof

                # ---- check 1: no aid residue survived ----------------------
                for cid, info in prof.items():
                    for code in info["residues"]:
                        if code in aids:
                            aid_hits[code] += 1
                            fail(f"{pdb}/{asm_key}/{cid}: aid residue {code} still present")

                # ---- check 4: composition ---------------------------------
                types = {i["type"] for i in prof.values()}
                bad = types & FORBIDDEN_TYPES
                if bad:
                    fail(f"{pdb}/{asm_key}: forbidden chain types {sorted(bad)}")
                if not types & PROTEIN_TYPES:
                    fail(f"{pdb}/{asm_key}: no protein chain")
                if DNA_TYPE not in types:
                    fail(f"{pdb}/{asm_key}: no DNA chain")

                # ---- check 2: only aid molecules left ----------------------
                if asm_key not in src_entry:
                    continue
                src_prof = chain_profile(src_entry[asm_key])
                for cid, src_info in src_prof.items():
                    non_aid = [c for c in src_info["residues"] if c not in aids]
                    n_aid = len(src_info["residues"]) - len(non_aid)
                    if not non_aid:
                        if cid in prof:
                            fail(f"{pdb}/{asm_key}/{cid}: all-aid chain still present")
                        else:
                            counts["aid_chain_removed"] += 1
                        continue
                    if cid not in prof:
                        fail(
                            f"{pdb}/{asm_key}/{cid}: non-aid chain vanished "
                            f"({src_info['n_res']} res, {n_aid} of them aids)",
                        )
                        continue
                    if prof[cid]["residues"] != non_aid:
                        fail(
                            f"{pdb}/{asm_key}/{cid}: residues differ from "
                            f"source-minus-aids ({len(prof[cid]['residues'])} vs {len(non_aid)})",
                        )
                    elif n_aid:
                        counts["chain_shrunk_by_aids"] += 1
                    else:
                        if prof[cid]["n_atom"] != src_info["n_atom"]:
                            fail(
                                f"{pdb}/{asm_key}/{cid}: aid-free chain atom count "
                                f"{prof[cid]['n_atom']} != source {src_info['n_atom']}",
                            )
                        if prof[cid]["cluster"] != src_info["cluster"]:
                            fail(f"{pdb}/{asm_key}/{cid}: cluster id changed")
                        counts["chain_untouched"] += 1

            # ---- check 5: rows resolve --------------------------------------
            for row in rows_by_pdb.get(pdb, []):
                asm_key = f"{row[3]}_{row[4]}_{row[5]}"
                prof = profiles.get(asm_key)
                if prof is None:
                    fail(f"{pdb}: row names missing sub-entry {asm_key}")
                    continue
                counts["rows_checked"] += 1
                missing = [c for c in (row[6], row[7]) if c not in prof]
                if missing:
                    fail(f"{pdb}/{asm_key}: chain(s) {missing} absent")
                    continue
                # The TSV lists (cluster1, cluster2) in the opposite order to
                # (chain_id1, chain_id2), so compare as an unordered pair.
                got = sorted(prof[c]["cluster"] for c in (row[6], row[7]))
                if got != sorted((row[0], row[1])):
                    fail(f"{pdb}/{asm_key}: chains carry {got}, row claims {row[0]},{row[1]}")

            if n_done % 500 == 0:
                print(f"  checked {n_done}/{stored} ({time.time() - started:.0f}s)", flush=True)

    out_env.close()
    src_env.close()

    for pdb in rows_by_pdb:
        if pdb not in seen:
            fail(f"{pdb}: row references it but it is not stored")

    print(f"\n-- results ({time.time() - started:.0f}s)")
    print(f"sub-entries checked          : {counts['subentries']}")
    print(f"rows checked against blobs   : {counts['rows_checked']} / {len(rows)}")
    print(f"blobs verbatim / re-encoded  : {counts['blob_verbatim']} / {counts['blob_reencoded']}")
    print("\n-- aid removal")
    print(f"residues with an aid code    : {sum(aid_hits.values())} {dict(aid_hits)}")
    print(f"all-aid chains removed       : {counts['aid_chain_removed']}")
    print(f"chains shrunk by aid removal : {counts['chain_shrunk_by_aids']}")
    print(f"chains untouched             : {counts['chain_untouched']}")

    if counts["problems"]:
        print(f"\nFAILED: {counts['problems']} problems, first {len(problems)}:")
        for p in problems:
            print("  ", p)
    else:
        print("\nOK: no problems found")
    return counts["problems"]


def main() -> None:
    """Verify the requested splits."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    args = ap.parse_args()

    names = list(SPLITS) if args.split == "all" else [args.split]
    total = sum(verify(name, SPLITS[name]) for name in names)
    print(f"\n{'=' * 72}")
    if total:
        raise SystemExit(f"{total} problems across {len(names)} split(s)")
    print(f"all {len(names)} split(s) verified clean")


if __name__ == "__main__":
    main()
