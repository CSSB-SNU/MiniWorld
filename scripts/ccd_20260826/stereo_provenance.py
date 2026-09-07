"""Per-centre provenance for the stereochemistry in rdkit_smiles.

Replays the production stereo decision with instrumentation, so
every stereocentre and every candidate double bond can be attributed to one of:

    geometry    coordinates present, Chem.AssignStereochemistryFrom3D decided it
    dihedral    coordinates present, 3D perception abstained (STEREOANY), the
                dihedral between the reference atoms decided it
    label       coordinates absent at that centre, the CCD label was used
    unspecified nothing was written -- either no coordinates and no label, or
                coordinates were present and geometry found no stereocentre
"""

import math

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MolToSmiles
from rdkit.Geometry import Point3D

from gen_record import record_to_src, topology

RDLogger.DisableLog("rdApp.*")
CW, CCW = Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW
CIS, TRANS = Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS

SITUATIONS = {
    "geometry_atom": "coordinates present at that centre -> geometry",
    "geometry_bond": "coordinates present at that bond -> geometry",
    "dihedral_bond": "coordinates present, RDKit abstained -> direct dihedral",
    "label_fallback": "coordinates absent, CCD states R/S or E/Z -> label",
    "nocoord_nolabel": "coordinates absent, CCD says N -> unspecified",
    "coords_say_none": "coordinates present, geometry finds nothing, CCD says R/S -> unspecified",
}



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


def analyse(rec):
    """(rows, situations, final_smiles). rows are provenance table entries."""
    src = record_to_src(rec)
    ids = [str(x) for x in rec["atoms"]["nodes"]["id"]["value"]]
    atom_label = [str(x) for x in rec["atoms"]["nodes"]["stereo"]["value"]]
    bs = rec["atoms"]["edges"]["bond_stereo"]
    bond_label = {
        frozenset((int(s), int(d))): str(v)
        for v, s, d in zip(bs["value"], bs["src_indices"], bs["dst_indices"])
    }

    xyz, missing = [], set()
    for i, row in enumerate(rec["atoms"]["nodes"]["model_xyz"]["value"]):
        try:
            xyz.append([float(x) for x in row])
        except (TypeError, ValueError):
            missing.add(i)
            xyz.append([1000.0 + 10.0 * i, 0.0, 0.0])
    xyz = np.array(xyz)

    mol = topology(src)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*xyz[i]))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(mol)
    try:
        Chem.AssignCIPLabels(mol)
    except Exception:  # noqa: BLE001
        pass

    geom_atom = {
        a.GetIdx(): a.GetPropsAsDict().get("_CIPCode") for a in mol.GetAtoms()
    }
    geom_bond = {}
    abstained = set()
    for b in mol.GetBonds():
        key = frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        if b.GetStereo() == Chem.BondStereo.STEREOANY:
            abstained.add(key)
        geom_bond[key] = b.GetPropsAsDict().get("_CIPCode")

    refs = bond_refs(mol)
    rows, sits = [], set()

    def coords_ok(*idx):
        return not any(i in missing for i in idx)

    # ---- atoms
    for a in mol.GetAtoms():
        i = a.GetIdx()
        lab = atom_label[i] if atom_label[i] in ("R", "S") else "N"
        touched = i in missing or any(nb.GetIdx() in missing for nb in a.GetNeighbors())
        g = geom_atom.get(i)
        g = g if g in ("R", "S") else None
        if not touched and g:
            decided, sit = "geometry", "geometry_atom"
        elif not touched and lab != "N":
            decided, sit = "unspecified", "coords_say_none"
        elif touched and lab != "N":
            decided, sit = "label", "label_fallback"
        elif touched:
            decided, sit = "unspecified", "nocoord_nolabel"
        else:
            continue
        sits.add(sit)
        # geometry_call is only meaningful where coordinates exist; with a
        # placeholder in play the perceived value is an artefact, so blank it.
        rows.append(
            ["atom", ids[i], "-", lab, "Y" if not touched else "N",
             (g or ".") if not touched else ".", "-", decided]
        )

    # ---- bonds
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        key = frozenset((i, j))
        lab = bond_label.get(key)
        lab = lab if lab in ("E", "Z") else "N"
        pair = refs.get(b.GetIdx())
        g = geom_bond.get(key)
        g = g if g in ("E", "Z") else None
        bond_coords = coords_ok(i, j, *(pair or ()))
        ang = None
        if pair and bond_coords:
            ang = dihedral(xyz[pair[0]], xyz[i], xyz[j], xyz[pair[1]])
        if bond_coords and g:
            decided, sit = "geometry", "geometry_bond"
        elif bond_coords and key in abstained and ang is not None:
            decided, sit = "dihedral", "dihedral_bond"
        elif bond_coords and pair and ang is not None and lab != "N" and not g:
            decided, sit = "dihedral", "dihedral_bond"
        elif bond_coords and lab != "N":
            decided, sit = "unspecified", "coords_say_none"
        elif not bond_coords and lab != "N":
            decided, sit = "label", "label_fallback"
        elif not bond_coords and pair:
            decided, sit = "unspecified", "nocoord_nolabel"
        else:
            continue
        sits.add(sit)
        rows.append(
            [
                "bond",
                ids[i],
                ids[j],
                lab,
                "Y" if bond_coords else "N",
                (g or ".") if bond_coords else ".",
                f"{ang:.1f}" if ang is not None else ".",
                decided,
            ]
        )
    return rows, sits, MolToSmiles(mol, isomericSmiles=True, canonical=True)
