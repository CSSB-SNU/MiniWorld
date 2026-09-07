"""Regenerate rdkit_smiles: coordinates decide, CCD labels fill coordinate gaps.

Per centre, not per record:

  coordinates available -> geometry decides
        atoms: Chem.AssignStereochemistryFrom3D
        bonds: the same, plus a direct dihedral for bonds it abstains on
               (STEREOANY), which is what lost JB9 (0.8 deg) and LJZ (140.2)

  coordinates absent    -> the CCD label is used, if it states one
        The label is a CIP letter (R/S, E/Z) and RDKit stores order-relative
        tags, so the letter cannot be assigned directly -- that mistranslation
        is the upstream bug that leaves ~75% of chiral records wrong. Instead
        the tag is solved: set one, read back Chem.AssignCIPLabels, flip if it
        does not match. Coordinate-derived centres are held fixed while the
        gap centres are solved, since CIP codes are interdependent.

Topology (element / charge / bond_type) is unchanged, and no stereo is seeded
from labels anywhere a coordinate exists.

Usage:
    python regen_smiles_final.py --out /path/new.lmdb [--dry-run]
"""

import argparse
import json
import math
from collections import Counter

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MolToSmiles
from rdkit.Geometry import Point3D

from biomol_codec import load_bytes, to_bytes
from gen_record import BOND_MAP, charge_map, element_map, topology
from gen_record import record_to_src

RDLogger.DisableLog("rdApp.*")
SRC = "/public_data/bsoohyuncd/preprocessed_CCD_20260826.lmdb"
MAP_SIZE = 2_000_000_000
MAX_SMILES = 500
CW, CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW
CIS, TRANS = Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS



def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n = np.linalg.norm(b1)
    if n == 0:
        return None
    b1n = b1 / n
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    if np.linalg.norm(v) < 1e-6 or np.linalg.norm(w) < 1e-6:
        return None
    return abs(math.degrees(math.atan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))))


def bond_refs(mol):
    out = {}
    try:
        info = Chem.FindPotentialStereo(Chem.Mol(mol))
    except Exception:  # noqa: BLE001
        return out
    n = mol.GetNumAtoms()
    for e in info:
        if str(e.type) != "Bond_Double":
            continue
        ctrl = getattr(e, "controllingAtoms", None)
        if not ctrl or len(ctrl) < 4:
            continue
        b = next((a for a in ctrl[0:2] if a < n), None)
        d = next((a for a in ctrl[2:4] if a < n), None)
        if b is not None and d is not None:
            out[e.centeredOn] = (b, d)
    return out


def solve_labels(mol, atom_targets, bond_targets, stats):
    """Realize CIP letters on the given centres by flipping until they read back."""
    for i in atom_targets:
        mol.GetAtomWithIdx(i).SetChiralTag(CW)
    for bidx, (ra, rb, _want) in bond_targets.items():
        b = mol.GetBondWithIdx(bidx)
        b.SetStereoAtoms(ra, rb)
        b.SetStereo(CIS)
    for _ in range(12):
        try:
            Chem.AssignCIPLabels(mol)
        except Exception:  # noqa: BLE001
            return
        flipped = 0
        for i, want in atom_targets.items():
            a = mol.GetAtomWithIdx(i)
            if a.GetPropsAsDict().get("_CIPCode") != want:
                a.SetChiralTag(CCW if a.GetChiralTag() == CW else CW)
                flipped += 1
        for bidx, (_ra, _rb, want) in bond_targets.items():
            b = mol.GetBondWithIdx(bidx)
            if b.GetPropsAsDict().get("_CIPCode") != want:
                b.SetStereo(TRANS if b.GetStereo() == CIS else CIS)
                flipped += 1
        if not flipped:
            stats["label fallback: solved"] += len(atom_targets) + len(bond_targets)
            return
    stats["label fallback: did not converge"] += len(atom_targets) + len(bond_targets)


TETRAHEDRAL = (CW, CCW)


def build(rec, stats, *, strip_exotic=True):
    src = record_to_src(rec)
    mol = topology(src)
    n = mol.GetNumAtoms()

    xyz, missing = [], set()
    for i, row in enumerate(rec["atoms"]["nodes"]["model_xyz"]["value"]):
        try:
            xyz.append([float(x) for x in row])
        except (TypeError, ValueError):
            missing.add(i)
            xyz.append([1000.0 + 10.0 * i, 0.0, 0.0])
    xyz = np.array(xyz)

    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(*xyz[i]))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(mol)

    atom_label = [str(x) for x in rec["atoms"]["nodes"]["stereo"]["value"]]
    bs = rec["atoms"]["edges"]["bond_stereo"]
    bond_label = {
        frozenset((int(s), int(d))): str(v)
        for v, s, d in zip(bs["value"], bs["src_indices"], bs["dst_indices"])
    }
    refs = bond_refs(mol)

    # --- atoms: drop anything resting on a placeholder, collect label fallbacks
    atom_targets = {}
    for a in mol.GetAtoms():
        i = a.GetIdx()
        touched = i in missing or any(nb.GetIdx() in missing for nb in a.GetNeighbors())
        if not touched:
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                stats["atom stereo from coordinates"] += 1
            continue
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if atom_label[i] in ("R", "S"):
            atom_targets[i] = atom_label[i]

    # --- bonds: coordinates first, dihedral where perception abstained
    bond_targets = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        sa = list(b.GetStereoAtoms())
        key = frozenset((i, j))
        determined = b.GetStereo() not in (
            Chem.BondStereo.STEREONONE,
            Chem.BondStereo.STEREOANY,
        )
        touched = (
            i in missing
            or j in missing
            or (len(sa) == 2 and (sa[0] in missing or sa[1] in missing))
        )
        if determined and not touched:
            stats["bond stereo from coordinates"] += 1
            continue
        if determined and touched:
            b.SetStereo(Chem.BondStereo.STEREONONE)
        if b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        pair = refs.get(b.GetIdx())
        if pair is None:
            continue
        ra, rb = pair
        coords_ok = not any(x in missing for x in (i, j, ra, rb))
        if coords_ok:
            ang = dihedral(xyz[ra], xyz[i], xyz[j], xyz[rb])
            if ang is not None:
                b.SetStereoAtoms(ra, rb)
                b.SetStereo(CIS if ang < 90 else TRANS)
                stats["bond stereo recovered by dihedral"] += 1
                continue
        if bond_label.get(key) in ("E", "Z"):
            bond_targets[b.GetIdx()] = (ra, rb, bond_label[key])

    if atom_targets or bond_targets:
        solve_labels(mol, atom_targets, bond_targets, stats)

    if strip_exotic:
        # Coordinate perception also assigns non-tetrahedral stereo on
        # hypervalent centres (@TB / @OH / @SP). Those descriptors are absent
        # from the label-derived database entirely, are usually an artefact of
        # how the experimental structure was modelled rather than a resolvable
        # configuration, and they account for most of the round-trip
        # instability. Keep only tetrahedral chirality.
        for a in mol.GetAtoms():
            tag = a.GetChiralTag()
            if tag != Chem.ChiralType.CHI_UNSPECIFIED and tag not in TETRAHEDRAL:
                a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                stats["non-tetrahedral stereo stripped"] += 1

    Chem.SetDoubleBondNeighborDirections(mol)
    mol.RemoveAllConformers()
    return MolToSmiles(mol, isomericSmiles=True, canonical=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-exotic", action="store_true",
                    help="keep @TB/@OH/@SP descriptors instead of stripping them")
    ap.add_argument("--widen", action="store_true",
                    help="widen the dtype for SMILES over 500 chars instead of skipping")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_env = lmdb.open(args.src, readonly=True, lock=False, readahead=False)
    stats = Counter()
    out, examples, too_long = {}, [], {}

    with src_env.begin() as t:
        for k, v in t.cursor():
            cid = k.decode()
            raw = bytes(v)
            rec = load_bytes(raw)
            stats["records"] += 1
            if not rec:
                out[cid] = raw
                continue
            try:
                new = build(rec, stats, strip_exotic=not args.keep_exotic)
            except Exception as exc:  # noqa: BLE001
                out[cid] = raw
                stats[f"failed: {type(exc).__name__}"] += 1
                continue
            stored = str(rec["residues"]["nodes"]["rdkit_smiles"]["value"][0])
            if new == stored:
                out[cid] = raw
                stats["unchanged"] += 1
                continue
            if len(new) > MAX_SMILES and len(stored) <= MAX_SMILES and not args.widen:
                too_long[cid] = len(new)
                out[cid] = raw
                stats["skipped: would exceed <U500"] += 1
                continue
            if len(new) > MAX_SMILES:
                too_long[cid] = len(new)
                stats["dtype widened past <U500"] += 1
            rec["residues"]["nodes"]["rdkit_smiles"]["value"] = np.array(
                [new], dtype=f"<U{max(MAX_SMILES, len(new))}"
            )
            out[cid] = to_bytes(rec)
            stats["rewritten"] += 1
            if stored.count("@") != new.count("@"):
                stats["  atom stereo count changed"] += 1
            if (stored.count("/") + stored.count("\\")) != (new.count("/") + new.count("\\")):
                stats["  bond stereo count changed"] += 1
            if len(examples) < 8:
                examples.append((cid, stored, new))

    for key in sorted(stats):
        print(f"  {key:46s} {stats[key]}")
    json.dump(too_long, open("final_too_long.json", "w"))
    if too_long:
        print(f"\nskipped for length ({len(too_long)}): {list(too_long.items())[:6]}")
    print("\nexamples:")
    for cid, old, new in examples:
        print(f"  {cid}\n    before: {old[:86]}\n    after : {new[:86]}")

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
