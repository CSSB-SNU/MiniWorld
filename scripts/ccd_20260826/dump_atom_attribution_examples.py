"""Export two example CIFs per ATOM-attribution category from verify_rule.py.

The four categories are the atom (R/S) rows of the full-database verification:

    atoms_geometry              coordinates present -> geometry decided it
    atoms_label                 coordinates absent, CCD stated R/S -> label used
    atoms_unspec_coords_none    coordinates present, geometry found no
                                stereocentre, CCD stated R/S -> left unspecified
    atoms_unspec_nocoord        coordinates absent and CCD said N -> unspecified

Each CIF carries the `_miniworld_stereo_provenance` loop, and the filename
encodes the category. Only atom rows are used to select records, so a record
picked for a category genuinely exhibits it at an atom.

Usage:
    python dump_atom_attribution_examples.py --n 2 --out examples_atom_attribution
"""

import argparse
import json
from pathlib import Path

import lmdb

from biomol_codec import load_bytes
from dump_cif_from_smiles import build as cif_from_smiles
from cif_writer import emit_loop
from stereo_provenance import analyse

DB = "/public_data/bsoohyuncd/preprocessed_CCD_20260826.lmdb"
COLUMNS = [
    "comp_id",
    "kind",
    "atom_id_1",
    "atom_id_2",
    "ccd_label",
    "coords_available",
    "geometry_call",
    "ref_dihedral_deg",
    "decided_by",
]
CATEGORIES = {
    "atoms_geometry": "coordinates present at the centre -> geometry decided R/S",
    "atoms_label": "coordinates absent, CCD stated R/S -> label used",
    "atoms_unspec_coords_none": "coordinates present, geometry found no stereocentre, CCD stated R/S -> unspecified",
    "atoms_unspec_nocoord": "coordinates absent and CCD said N -> unspecified",
}


def categorise(row):
    """row = [kind, id1, id2, ccd_label, coords_avail, geom, dihedral, decided_by]"""
    if row[0] != "atom":
        return None
    label, coords, decided = row[3], row[4], row[7]
    if decided == "geometry":
        return "atoms_geometry"
    if decided == "label":
        return "atoms_label"
    if decided == "unspecified" and coords == "Y" and label in ("R", "S"):
        return "atoms_unspec_coords_none"
    if decided == "unspecified" and coords == "N":
        return "atoms_unspec_nocoord"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default="examples_atom_attribution")
    ap.add_argument("--max-atoms", type=int, default=35)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(DB, readonly=True, lock=False, readahead=False)
    picked = {c: [] for c in CATEGORIES}

    with env.begin() as t:
        for k, v in t.cursor():
            if all(len(p) >= args.n for p in picked.values()):
                break
            cid = k.decode()
            rec = load_bytes(bytes(v))
            if not rec:
                continue
            if rec["atoms"]["nodes"]["id"]["value"].shape[0] > args.max_atoms:
                continue
            try:
                rows, _sits, _smi = analyse(rec)
            except Exception:  # noqa: BLE001
                continue
            cats = {c for c in (categorise(r) for r in rows) if c}
            for c in cats:
                if len(picked[c]) < args.n and cid not in [x for x, _ in picked[c]]:
                    picked[c].append((cid, rows))

    manifest = []
    for cat, entries in picked.items():
        for cid, rows in entries:
            with env.begin() as t:
                rec = load_bytes(bytes(t.get(cid.encode())))
            text, err = cif_from_smiles(cid, rec)
            if err:
                print(f"  {cat:26s} {cid:7s} SKIPPED -- {err}")
                continue
            out = text.rstrip("\n").split("\n")
            emit_loop(out, "_miniworld_stereo_provenance", COLUMNS,
                      [[cid, *r] for r in rows])
            path = outdir / f"{cat}__{cid}.cif"
            path.write_text("\n".join(out) + "\n")
            smi = str(rec["residues"]["nodes"]["rdkit_smiles"]["value"][0])
            manifest.append({"category": cat, "description": CATEGORIES[cat],
                             "comp_id": cid, "file": path.name, "rdkit_smiles": smi})
            print(f"  {cat:26s} {cid:7s} -> {path.name}")
            print(f"       SMILES: {smi[:84]}")
            for r in rows:
                if categorise(r) == cat:
                    print(f"       atom {r[1]:5s} ccd_label={r[3]:2s} coords={r[4]} "
                          f"geometry={r[5]:2s} -> {r[7]}")

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} files + manifest.json in {outdir}")
    short = [c for c, p in picked.items() if len(p) < args.n]
    if short:
        print(f"fewer than {args.n} found for: {short}")


if __name__ == "__main__":
    main()
