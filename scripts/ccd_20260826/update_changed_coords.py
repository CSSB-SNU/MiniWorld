"""Refresh model_xyz from the current CCD where it has changed, and redo the SMILES.

Comparing the LMDB against the 2026-08-22 components.cif found 20 records whose
model_Cartn differs upstream -- 11 where the LMDB stores '?' and the CCD now has
values, 9 where the numbers changed. Since double-bond stereo is derived from the
record's own coordinates, both fields are updated together so the record stays
self-consistent: a reader can re-derive the SMILES stereo from the model_xyz in
the same record.

Only atoms.nodes.model_xyz and residues.nodes.rdkit_smiles are touched, and only
for records whose heavy-atom id list still matches the CCD exactly.

Usage:
    python update_changed_coords.py --store /path/ccd_store.pkl --out /path/new.lmdb [--dry-run]
"""

import argparse
import json
import pickle
from collections import Counter

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MolToSmiles

import audit_stereo_from_coords as audit
from audit_stereo_from_coords import model_coords, stereo_from_3d
from biomol_codec import load_bytes, to_bytes
from gen_record import HYDROGENS, build_rdkit_mol
from gen_record import record_to_src

RDLogger.DisableLog("rdApp.*")
SRC = "/public_data/bsoohyuncd/preprocessed_CCD_20260826.lmdb"
MAP_SIZE = 2_000_000_000
MAX_SMILES = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="coords_changed_vs_ccd.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    store = pickle.load(open(args.store, "rb"))
    audit._store = store
    targets = set(json.load(open(args.targets)))

    src_env = lmdb.open(args.src, readonly=True, lock=False, readahead=False)
    stats = Counter()
    out = {}
    report = []

    with src_env.begin() as t:
        for k, v in t.cursor():
            cid = k.decode()
            raw = bytes(v)
            if cid not in targets:
                out[cid] = raw
                continue
            rec = load_bytes(raw)
            a = store[cid]["atoms"]
            keep = [i for i, e in enumerate(a["type_symbol"]) if e.upper() not in HYDROGENS]
            ccd_ids = [a["atom_id"][i] for i in keep]
            lm_ids = [str(x) for x in rec["atoms"]["nodes"]["id"]["value"]]
            if ccd_ids != lm_ids:
                out[cid] = raw
                stats["composition mismatch, skipped"] += 1
                continue

            old_xyz = rec["atoms"]["nodes"]["model_xyz"]["value"]
            new_rows = [
                [a["model_Cartn_x"][i], a["model_Cartn_y"][i], a["model_Cartn_z"][i]]
                for i in keep
            ]
            width = max(len(s) for row in new_rows for s in row)
            rec["atoms"]["nodes"]["model_xyz"]["value"] = np.array(
                new_rows, dtype=f"<U{max(width, old_xyz.dtype.itemsize // 4)}"
            )

            old_smiles = str(rec["residues"]["nodes"]["rdkit_smiles"]["value"][0])
            try:
                src = record_to_src(rec)
                mol = build_rdkit_mol(src, emit_double_bond_stereo=False)
                geo = stereo_from_3d(mol, model_coords(rec))
            except Exception as exc:  # noqa: BLE001
                out[cid] = raw
                stats[f"rebuild failed: {type(exc).__name__}"] += 1
                continue
            if geo:
                for key, (st, sa) in geo.items():
                    i, j = tuple(key)
                    b = mol.GetBondBetweenAtoms(i, j)
                    if b is not None:
                        b.SetStereoAtoms(sa[0], sa[1])
                        b.SetStereo(st)
                Chem.SetDoubleBondNeighborDirections(mol)
            new_smiles = MolToSmiles(mol, isomericSmiles=True, canonical=True)
            if len(new_smiles) > MAX_SMILES and len(old_smiles) <= MAX_SMILES:
                stats["new SMILES too long, coords updated only"] += 1
            else:
                rec["residues"]["nodes"]["rdkit_smiles"]["value"] = np.array(
                    [new_smiles], dtype=f"<U{max(MAX_SMILES, len(new_smiles))}"
                )
            out[cid] = to_bytes(rec)
            stats["coordinates refreshed"] += 1
            if new_smiles != old_smiles:
                stats["  SMILES also changed"] += 1
            report.append((cid, old_smiles != new_smiles, geo is not None and bool(geo)))

    for key in sorted(stats):
        print(f"  {key:44s} {stats[key]}")
    print("\nper record (cid, smiles_changed, stereo_determined):")
    for r in report:
        print(f"   {r[0]:7s} smiles_changed={str(r[1]):5s} stereo_from_coords={r[2]}")

    if args.dry_run:
        print("\n[dry run] nothing written")
        return
    dst = lmdb.open(args.out, map_size=MAP_SIZE)
    with dst.begin(write=True) as t:
        for cid, blob in sorted(out.items()):
            t.put(cid.encode(), blob)
    dst.sync()
    with dst.begin() as t:
        n = t.stat()["entries"]
    dst.close()
    print(f"\nwrote {args.out}: {n} keys")


if __name__ == "__main__":
    main()
