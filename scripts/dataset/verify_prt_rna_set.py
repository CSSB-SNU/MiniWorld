#!/usr/bin/env python3
"""Check the valid2 protein-RNA subset against its source and its edge_node TSV.

The RNA counterpart of ``verify_prt_dna_set.py``.  Runs over every entry and
every row, not a sample.

Aid removal -- ONLY aid molecules may have left:

  1. no residue in the output carries a CRYSTALLIZATION_AID_CHEMCOMPS code;
  2. for every sub-entry kept, each of the source's chains is accounted for: a
     chain whose residues were ALL aids must be gone, a chain with SOME aid
     residues must have shrunk by exactly those residues, and a chain with NO aid
     residues must survive with an identical residue count, atom count and
     cluster id.

Composition -- every kept sub-entry is a real protein-RNA complex:

  3. it holds at least one protein chain (``cP``/``cQ``/``cA``);
  4. it holds at least one RNA chain (``cR``);
  5. it holds no DNA (``cD``) and no DNA/RNA hybrid (``cN``) chain.

Bookkeeping:

  6. every stored blob decodes and every sub-entry carries a chain table;
  7. every TSV row is an RNA-vs-protein pair, its ``{assembly}_{model}_{alt}``
     sub-entry exists, and both chain ids exist in it with the cluster ids the
     row claims (compared as an unordered pair -- the TSV lists cluster1 with
     chain_id2);
  8. every stored key is referenced by at least one row and vice versa;
  9. a verbatim blob is byte-identical to the source; a re-encoded blob holds a
     subset of the source's sub-entries.

Usage
-----
    /home/bsoohyuncd/software/StructCooker/.pixi/envs/default/bin/python \
        scripts/verify_prt_rna_set.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
from biomol.core.utils import load_bytes
from structcooker.mols import CIFMolAttached

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_prt_rna_set import (  # noqa: E402
    AID_CODES,
    FORBIDDEN_TYPES,
    IN_LMDB,
    IN_TSV,
    OUT_LMDB,
    OUT_TSV,
    PROTEIN_TYPES,
    RNA_TYPE,
    TSV_HEADER,
    is_protein_rna_pair,
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


def verify() -> int:  # noqa: C901, PLR0912, PLR0915
    """Verify the protein-RNA subset; return the number of problems found."""
    print(f"{'=' * 72}\nvalid2 protein-RNA\n{'=' * 72}")
    print(f"out_lmdb  : {OUT_LMDB}")
    print(f"out_tsv   : {OUT_TSV}")
    print(f"in_lmdb   : {IN_LMDB}")
    print(f"in_tsv    : {IN_TSV}")

    problems: list[str] = []
    counts: Counter[str] = Counter()
    aid_hits: Counter[str] = Counter()
    type_hits: Counter[str] = Counter()
    aids = set(AID_CODES)

    def fail(msg: str) -> None:
        counts["problems"] += 1
        if len(problems) < MAX_REPORTED:
            problems.append(msg)

    rows: list[tuple[str, ...]] = []
    with OUT_TSV.open() as fh:
        if next(fh).rstrip("\n") != TSV_HEADER.rstrip("\n"):
            fail("tsv header mismatch")
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped:
                rows.append(tuple(stripped.split("\t")))
    print(f"\nrows in tsv     : {len(rows)}")

    out_env = lmdb.open(str(OUT_LMDB), readonly=True, lock=False, max_readers=2048)
    src_env = lmdb.open(str(IN_LMDB), readonly=True, lock=False, max_readers=2048)
    stored = out_env.stat()["entries"]
    print(f"keys in lmdb    : {stored}")

    rows_by_pdb: dict[str, list[tuple[str, ...]]] = {}
    for row in rows:
        if len(row) != 8:
            fail(f"row has {len(row)} columns: {row}")
            continue
        if not is_protein_rna_pair(row[0], row[1]):
            fail(f"row is not an RNA-protein pair: {row}")
        rows_by_pdb.setdefault(row[2].lower(), []).append(row)
    print(f"distinct pdb ids: {len(rows_by_pdb)}")

    started = time.time()
    seen: set[str] = set()

    with out_env.begin(buffers=True) as out_txn, src_env.begin(buffers=True) as src_txn:
        for raw_key, raw_val in out_txn.cursor():
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
            counts["blob_verbatim" if src_blob == blob else "blob_reencoded"] += 1
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

                # ---- 1: no aid residue survived ---------------------------
                for cid, info in prof.items():
                    for code in info["residues"]:
                        if code in aids:
                            aid_hits[code] += 1
                            fail(f"{pdb}/{asm_key}/{cid}: aid residue {code} present")

                # ---- 3/4/5: composition -----------------------------------
                types = {i["type"] for i in prof.values()}
                type_hits.update(types)
                bad = types & FORBIDDEN_TYPES
                if bad:
                    fail(f"{pdb}/{asm_key}: forbidden chain types {sorted(bad)}")
                if not types & PROTEIN_TYPES:
                    fail(f"{pdb}/{asm_key}: no protein chain (types {sorted(types)})")
                else:
                    counts["subentry_has_protein"] += 1
                if RNA_TYPE not in types:
                    fail(f"{pdb}/{asm_key}: no RNA chain (types {sorted(types)})")
                else:
                    counts["subentry_has_rna"] += 1

                # ---- 2: only aid molecules left ---------------------------
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
                            f"({src_info['n_res']} res, {n_aid} aids)",
                        )
                        continue
                    if prof[cid]["residues"] != non_aid:
                        fail(f"{pdb}/{asm_key}/{cid}: residues differ from source-minus-aids")
                    elif n_aid:
                        counts["chain_shrunk_by_aids"] += 1
                    else:
                        if prof[cid]["n_atom"] != src_info["n_atom"]:
                            fail(f"{pdb}/{asm_key}/{cid}: aid-free chain atom count changed")
                        if prof[cid]["cluster"] != src_info["cluster"]:
                            fail(f"{pdb}/{asm_key}/{cid}: cluster id changed")
                        counts["chain_untouched"] += 1

            # ---- 7: rows resolve -------------------------------------------
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
                got = sorted(prof[c]["cluster"] for c in (row[6], row[7]))
                if got != sorted((row[0], row[1])):
                    fail(f"{pdb}/{asm_key}: chains carry {got}, row claims {row[0]},{row[1]}")

    out_env.close()
    src_env.close()

    for pdb in rows_by_pdb:
        if pdb not in seen:
            fail(f"{pdb}: row references it but it is not stored")

    subs = counts["subentries"]
    print(f"\n-- results ({time.time() - started:.0f}s)")
    print(f"sub-entries checked          : {subs}")
    print(f"rows checked against blobs   : {counts['rows_checked']} / {len(rows)}")
    print(f"blobs verbatim / re-encoded  : {counts['blob_verbatim']} / {counts['blob_reencoded']}")
    print("\n-- aid removal")
    print(f"residues with an aid code    : {sum(aid_hits.values())} {dict(aid_hits)}")
    print(f"all-aid chains removed       : {counts['aid_chain_removed']}")
    print(f"chains shrunk by aid removal : {counts['chain_shrunk_by_aids']}")
    print(f"chains untouched             : {counts['chain_untouched']}")
    print("\n-- composition")
    print(f"sub-entries with >=1 protein : {counts['subentry_has_protein']} / {subs}")
    print(f"sub-entries with >=1 RNA     : {counts['subentry_has_rna']} / {subs}")
    print(f"chain types present          : {dict(sorted(type_hits.items()))}")

    if counts["problems"]:
        print(f"\nFAILED: {counts['problems']} problems, first {len(problems)}:")
        for p in problems:
            print("  ", p)
    else:
        print("\nOK: no problems found")
    return counts["problems"]


if __name__ == "__main__":
    n = verify()
    print(f"\n{'=' * 72}")
    if n:
        raise SystemExit(f"{n} problems")
    print("protein-RNA subset verified clean")
