"""Determine double-bond E/Z strictly from 3D coordinates, and audit the CCD labels.

For every record, a conformer is attached to the pipeline molecule and
Chem.AssignStereochemistryFrom3D decides the geometry of each double bond. That
is independent of `_chem_comp_bond.pdbx_stereo_config`, so the two can be
compared.

Coordinate sources, in order:
  ideal  -- pdbx_model_Cartn_*_ideal from components.cif (clean, generated
            geometry; complete for 50,667 of 51,055 components)
  model  -- model_Cartn_* as stored in the record (experimental; can be
            distorted, e.g. 1KC's 93.5 deg where ideal gives a clean 180)

Ideal coordinates are only used when the CCD's heavy-atom id list still matches
the record's, so a component revised upstream cannot be mis-indexed.

Usage:
    python audit_stereo_from_coords.py --store /path/ccd_store.pkl [--source ideal|model|both]
"""

import argparse
import json
import pickle
from collections import Counter
from multiprocessing import Pool

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

from biomol_codec import load_bytes
from gen_record import HYDROGENS, build_rdkit_mol
from gen_record import record_to_src

RDLogger.DisableLog("rdApp.*")
DB = "/public_data/bsoohyuncd/preprocessed_CCD_20260826.lmdb"
MISSING = {"?", ".", ""}

_store = None
_env = None


def _init(store_path):
    global _store, _env
    _store = pickle.load(open(store_path, "rb"))
    _env = lmdb.open(DB, readonly=True, lock=False, readahead=False)


def ideal_coords(cid, atom_ids):
    """Ideal coordinates aligned to the record's heavy atoms, or None."""
    rec = _store.get(cid)
    if not rec or not rec.get("atoms"):
        return None
    a = rec["atoms"]
    cols = (
        "pdbx_model_Cartn_x_ideal",
        "pdbx_model_Cartn_y_ideal",
        "pdbx_model_Cartn_z_ideal",
    )
    if any(c not in a for c in cols):
        return None
    keep = [i for i, e in enumerate(a["type_symbol"]) if e.upper() not in HYDROGENS]
    if [a["atom_id"][i] for i in keep] != list(atom_ids):
        return None  # composition drifted; refuse to index into it
    out = []
    for i in keep:
        try:
            out.append([float(a[c][i]) for c in cols])
        except (ValueError, TypeError):
            out.append([float("nan")] * 3)
    return np.array(out)


def model_coords(rec):
    out = []
    for row in rec["atoms"]["nodes"]["model_xyz"]["value"]:
        try:
            out.append([float(x) for x in row])
        except (ValueError, TypeError):
            out.append([float("nan")] * 3)
    return np.array(out)


def stereo_from_3d(mol, xyz):
    """{frozenset(bond atoms): STEREOCIS/STEREOTRANS} perceived from coordinates."""
    if xyz is None or np.isnan(xyz).all():
        return None
    probe = Chem.Mol(mol)
    probe.RemoveAllConformers()
    conf = Chem.Conformer(probe.GetNumAtoms())
    for i in range(probe.GetNumAtoms()):
        if np.isnan(xyz[i]).any():
            return None  # incomplete geometry: refuse rather than guess
        conf.SetAtomPosition(i, Point3D(*(float(v) for v in xyz[i])))
    probe.AddConformer(conf, assignId=True)
    try:
        Chem.AssignStereochemistryFrom3D(probe)
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for b in probe.GetBonds():
        # STEREOANY means the coordinates did not determine the geometry (e.g.
        # JB9's CBM=CBN). Treat it as "no opinion" -- writing it emits nothing
        # and would silently drop whatever stereo was there before.
        if b.GetStereo() in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY):
            continue
        sa = list(b.GetStereoAtoms())
        if len(sa) != 2:
            continue
        out[frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))] = (
            b.GetStereo(),
            tuple(sa),
        )
    return out


def analyse(cid):
    with _env.begin() as t:
        raw = t.get(cid.encode())
    rec = load_bytes(bytes(raw))
    if not rec:
        return cid, Counter(), []
    src = record_to_src(rec)
    atom_ids = [str(x) for x in rec["atoms"]["nodes"]["id"]["value"]]
    try:
        mol = build_rdkit_mol(src, emit_double_bond_stereo=False)
    except Exception:  # noqa: BLE001
        return cid, Counter({"build error": 1}), []

    bs = rec["atoms"]["edges"]["bond_stereo"]
    labels = {
        frozenset((int(s), int(d))): str(v)
        for v, s, d in zip(bs["value"], bs["src_indices"], bs["dst_indices"])
    }

    c = Counter()
    rows = []
    geo = {
        "ideal": stereo_from_3d(mol, ideal_coords(cid, atom_ids)),
        "model": stereo_from_3d(mol, model_coords(rec)),
    }
    for name, g in geo.items():
        if g is None:
            c[f"{name}: unusable"] += 1
            continue
        c[f"{name}: analysed"] += 1
        for key, (st, sa) in g.items():
            lab = labels.get(key)
            same_side = st in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS)
            c[f"{name}: stereogenic double bonds"] += 1
            if lab in ("E", "Z"):
                # the label is CIP; translate the geometric call the same way the
                # pipeline does (reference-atom relative) for a like-for-like test
                label_same_side = lab == "Z"
                if label_same_side == same_side:
                    c[f"{name}: label agrees"] += 1
                else:
                    c[f"{name}: label DISAGREES"] += 1
                    rows.append((cid, name, lab))
            else:
                c[f"{name}: geometry stereogenic but label is N"] += 1
    # bonds labelled E/Z that geometry does not consider stereogenic
    for key, lab in labels.items():
        if lab not in ("E", "Z"):
            continue
        for name, g in geo.items():
            if g is not None and key not in g:
                c[f"{name}: labelled E/Z but not stereogenic by geometry"] += 1
    return cid, c, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    args = ap.parse_args()
    env = lmdb.open(DB, readonly=True, lock=False, readahead=False)
    with env.begin() as t:
        keys = [k.decode() for k in t.cursor().iternext(keys=True, values=False)]
    env.close()

    total = Counter()
    disagreements = []
    with Pool(14, initializer=_init, initargs=(args.store,)) as pool:
        for i, (cid, c, rows) in enumerate(pool.imap_unordered(analyse, keys, chunksize=64)):
            total.update(c)
            disagreements.extend(rows)
            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{len(keys)}", flush=True)

    print(f"\nrecords: {len(keys)}\n")
    for k in sorted(total):
        print(f"  {k:52s} {total[k]}")
    json.dump(
        sorted({cid for cid, _n, _l in disagreements}),
        open("stereo_coord_label_disagreements.json", "w"),
    )
    print(f"\ndistinct components with a disagreement: "
          f"{len({cid for cid, _n, _l in disagreements})}")


if __name__ == "__main__":
    main()
