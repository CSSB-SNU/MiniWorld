"""Build a preprocessed_CCD record from a raw CCD component definition.

Mirrors the pipeline in
  Ligand-Tokenizer/src/ligand_tokenizer/io/_preprocess_instr.py
  Ligand-Tokenizer/src/ligand_tokenizer/io/_preprocess_recipe.py
applied to the CCD BioMol that datacooker's ccd_recipe_book produces
(hydrogens dropped, bonds in CCD file order).
"""

import math

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AssignStereochemistry,
    BondStereo,
    FastFindRings,
    FindPotentialStereoBonds,
    MolToSmiles,
    RWMol,
    SanitizeFlags,
    SanitizeMol,
)

RDLogger.DisableLog("rdApp.*")

CHIRAL_MAP = {
    "N": Chem.ChiralType.CHI_UNSPECIFIED,
    "R": Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    "S": Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
}
BOND_MAP = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
    "OTHE": Chem.BondType.OTHER,
}
HYDROGENS = {"H", "D", "T"}


def charge_map(x):
    return int(x) if x != "?" else 0


def element_map(x: str):
    if x == "D":
        return "H"
    if x == "X":
        return 0
    if len(x) > 1:
        return x[0] + x[1:].lower()
    return x


def ccd_to_src(cid, rec, *, drop_hydrogens=True):
    """Reproduce the datacooker CCD BioMol arrays for one component."""
    atoms, bonds = rec["atoms"], rec["bonds"]
    if not atoms:
        msg = f"{cid}: no _chem_comp_atom table"
        raise ValueError(msg)
    n_raw = len(atoms["atom_id"])
    if drop_hydrogens:
        keep = [i for i in range(n_raw) if atoms["type_symbol"][i].upper() not in HYDROGENS]
    else:
        keep = list(range(n_raw))
    if not keep:
        msg = f"{cid}: no heavy atoms"
        raise ValueError(msg)
    pos = {atoms["atom_id"][i]: n for n, i in enumerate(keep)}

    # Arrays are built over ALL atoms and then row-filtered, matching the
    # original pipeline: numpy fancy indexing keeps the wider <U dtype that the
    # hydrogen rows induced.
    rows = np.asarray(keep)
    src = {
        "id": np.array(atoms["atom_id"])[rows],
        "element": np.array(atoms["type_symbol"])[rows],
        "aromatic": np.array(atoms["pdbx_aromatic_flag"])[rows],
        "stereo": np.array(atoms["pdbx_stereo_config"])[rows],
        "charge": np.array(
            ["0" if c in ("?", ".", "") else c for c in atoms["charge"]]
        )[rows],
        "model_xyz": np.array(
            [atoms["model_Cartn_x"], atoms["model_Cartn_y"], atoms["model_Cartn_z"]]
        ).T[rows],
    }

    sel = []
    if bonds:
        for j, (a1, a2) in enumerate(zip(bonds["atom_id_1"], bonds["atom_id_2"])):
            if a1 in pos and a2 in pos:
                sel.append(j)
    bsel = np.asarray(sel, dtype=np.int64)
    if bonds:
        src["bond_type"] = np.array(bonds["value_order"])[bsel]
        src["bond_aromatic"] = np.array(bonds["pdbx_aromatic_flag"])[bsel]
        src["bond_stereo"] = np.array(bonds["pdbx_stereo_config"])[bsel]
    else:
        # No _chem_comp_bond table at all: np.array([], dtype=str) -> <U1
        src["bond_type"] = np.array([], dtype=str)
        src["bond_aromatic"] = np.array([], dtype=str)
        src["bond_stereo"] = np.array([], dtype=str)
    src["bond_src"] = np.array([pos[bonds["atom_id_1"][j]] for j in sel], dtype=np.int64)
    src["bond_dst"] = np.array([pos[bonds["atom_id_2"][j]] for j in sel], dtype=np.int64)
    return src


def build_rdkit_mol(src, *, metal_bond=True, use_h=True, emit_double_bond_stereo=False):
    """`build_rdkit_mol` from _preprocess_instr.py, on plain arrays.

    `emit_double_bond_stereo` is an addition, off by default so the original
    pipeline stays reproducible. Upstream sets E/Z on the double bonds but never
    calls SetDoubleBondNeighborDirections, so the geometry never reaches
    MolToSmiles -- and the trailing AssignStereochemistry(cleanIt=True) clears
    it anyway. When enabled, the CCD E/Z labels are re-applied after that call
    and the neighbour bond directions are materialized.
    """
    rw = RWMol()
    for element, stereo, charge in zip(src["element"], src["stereo"], src["charge"]):
        rd_at = Chem.Atom(element_map(str(element)))
        rd_at.SetChiralTag(CHIRAL_MAP.get(str(stereo), CHIRAL_MAP["N"]))
        rd_at.SetFormalCharge(charge_map(str(charge)))
        rw.AddAtom(rd_at)

    for val, s, d in zip(src["bond_type"], src["bond_src"], src["bond_dst"]):
        bt = BOND_MAP.get(str(val), Chem.BondType.SINGLE)
        if not metal_bond and bt == Chem.BondType.OTHER:
            continue
        rw.AddBond(int(s), int(d), bt)

    SanitizeMol(rw, SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_PROPERTIES)
    mol = rw.GetMol()

    # stereo_applicator
    mol.UpdatePropertyCache(strict=False)
    FastFindRings(mol)
    FindPotentialStereoBonds(mol, cleanIt=True)
    s_map = {"E": BondStereo.STEREOE, "Z": BondStereo.STEREOZ}

    # Snapshot the stereo atoms while FindPotentialStereoBonds' assignment is
    # still intact. A bond with no stereo atoms cannot carry a meaningful E/Z
    # label -- setting one leaves RDKit inconsistent and segfaults
    # SetDoubleBondNeighborDirections downstream (e.g. ring double bonds such as
    # 08L's 3=2, which FindPotentialStereoBonds deliberately does not flag).
    modern = None  # lazily built: bond idx -> controllingAtoms from FindPotentialStereo
    pristine = Chem.Mol(mol) if emit_double_bond_stereo else None

    def modern_stereo_atoms(bond):
        """Reference atoms for a bond the legacy perception skipped.

        Legacy FindPotentialStereoBonds ignores ring double bonds outright, so
        macrocyclic alkenes (e.g. 08L's C9=C8 in an 18-membered ring) come back
        with no stereo atoms even though they are genuinely stereogenic. The
        modern API does flag them.
        """
        nonlocal modern
        if modern is None:
            modern = {}
            # Query a pristine copy: FindPotentialStereo raises on a molecule
            # that already has stereo set without matching stereo atoms.
            try:
                for entry in Chem.FindPotentialStereo(pristine):
                    if str(entry.type) == "Atom_Tetrahedral":
                        continue
                    ctrl = getattr(entry, "controllingAtoms", None)
                    if ctrl is not None:
                        modern[entry.centeredOn] = list(ctrl)
            except Exception:  # noqa: BLE001
                modern = {}
        ctrl = modern.get(bond.GetIdx())
        if not ctrl or len(ctrl) < 4:
            return None
        n_atoms = mol.GetNumAtoms()
        begin = next((a for a in ctrl[0:2] if a < n_atoms), None)
        end = next((a for a in ctrl[2:4] if a < n_atoms), None)
        if begin is None or end is None:
            return None
        return begin, end

    recoverable = []
    for val, s, d in zip(src["bond_stereo"], src["bond_src"], src["bond_dst"]):
        bond = mol.GetBondBetweenAtoms(int(s), int(d))
        if bond and bond.GetBondType() == Chem.BondType.DOUBLE and str(val) in s_map:
            bond.SetStereo(s_map[str(val)])
            sa = list(bond.GetStereoAtoms())
            if len(sa) != 2 and emit_double_bond_stereo:
                found = modern_stereo_atoms(bond)
                if found is not None:
                    sa = list(found)
            if len(sa) == 2:
                recoverable.append((int(s), int(d), sa[0], sa[1], s_map[str(val)]))

    AssignStereochemistry(mol, force=True, cleanIt=True)

    if emit_double_bond_stereo and recoverable:
        # AssignStereochemistry(cleanIt=True) has just cleared the E/Z labels,
        # so restore them along with their stereo atoms, then materialize the
        # neighbour bond directions that MolToSmiles writes as / and \.
        for s, d, sa0, sa1, label in recoverable:
            bond = mol.GetBondBetweenAtoms(s, d)
            if bond is None:
                continue
            bond.SetStereoAtoms(sa0, sa1)
            bond.SetStereo(label)
        Chem.SetDoubleBondNeighborDirections(mol)

    if not use_h:
        mol = Chem.RemoveHs(mol)
    return mol


def compute_sssr(rd_mol, max_size=14):
    rings = rd_mol.GetRingInfo().AtomRings()
    sssrs = [set(r) for r in rings if len(r) <= max_size]
    num_atoms = rd_mol.GetNumAtoms()
    max_rings = max(len(sssrs), 1)
    ring_idx = np.full((num_atoms, max_rings), -1, dtype=np.int16)
    for sssr_idx, sssr in enumerate(sssrs):
        for atom_idx in sssr:
            slot = np.where(ring_idx[atom_idx] == -1)[0][0]
            ring_idx[atom_idx, slot] = sssr_idx
    return ring_idx


def bond_feature(rd_mol, method, dtype):
    src, dst, values = [], [], []
    for bond in rd_mol.GetBonds():
        src.append(bond.GetBeginAtomIdx())
        dst.append(bond.GetEndAtomIdx())
        values.append(dtype(getattr(bond, method)()))
    return np.array(src), np.array(dst), np.array(values)


def atom_feature(rd_mol, method, dtype):
    return np.array([dtype(getattr(a, method)()) for a in rd_mol.GetAtoms()])


def build_record(cid, rec, *, drop_hydrogens=True, emit_double_bond_stereo=False):
    """Return the record dict ready for biomol_codec.to_bytes."""
    src = ccd_to_src(cid, rec, drop_hydrogens=drop_hydrogens)
    mol = build_rdkit_mol(src, metal_bond=True, use_h=True,
                          emit_double_bond_stereo=emit_double_bond_stereo)

    sssr_idx = compute_sssr(mol, max_size=14)
    hyb = atom_feature(mol, "GetHybridization", str)
    cj_src, cj_dst, cj_val = bond_feature(mol, "GetIsConjugated", bool)
    ar_src, ar_dst, ar_val = bond_feature(mol, "GetIsAromatic", bool)
    smiles = MolToSmiles(mol, isomericSmiles=True, canonical=True)

    n_at = len(src["id"])
    return {
        "atoms": {
            "nodes": {
                "id": {"value": src["id"]},
                "element": {"value": src["element"]},
                "aromatic": {"value": src["aromatic"]},
                "stereo": {"value": src["stereo"]},
                "charge": {"value": src["charge"]},
                "model_xyz": {"value": src["model_xyz"]},
                "sssr_idx": {"value": sssr_idx},
                "hybridization": {"value": hyb},
            },
            "edges": {
                "bond_type": {
                    "value": src["bond_type"],
                    "src_indices": src["bond_src"],
                    "dst_indices": src["bond_dst"],
                },
                "bond_aromatic": {
                    "value": src["bond_aromatic"],
                    "src_indices": src["bond_src"],
                    "dst_indices": src["bond_dst"],
                },
                "bond_stereo": {
                    "value": src["bond_stereo"],
                    "src_indices": src["bond_src"],
                    "dst_indices": src["bond_dst"],
                },
                "bond_conjugation": {
                    "value": cj_val,
                    "src_indices": cj_src,
                    "dst_indices": cj_dst,
                },
                "bond_aromaticity": {
                    "value": ar_val,
                    "src_indices": ar_src,
                    "dst_indices": ar_dst,
                },
            },
        },
        "residues": {
            "nodes": {
                "id": {"value": np.array([rec["comp"].get("name", "")])},
                "formula": {"value": np.array([rec["comp"].get("formula", "")])},
                "rdkit_smiles": {"value": np.array([smiles] * 1, dtype="<U500")},
            },
            "edges": {},
        },
        "chains": {"nodes": {"id": {"value": np.array([cid])}}, "edges": {}},
        "index_table": {
            "atom_to_res": [0] * n_at,
            "res_to_chain": [0],
            "res_atom_indptr": [0, n_at],
            "res_atom_indices": list(range(n_at)),
            "chain_res_indptr": [0, 1],
            "chain_res_indices": [0],
        },
        "metadata": {},
    }


def record_to_src(rec):
    """Recover the build_rdkit_mol input arrays from a stored LMDB record.

    Moved here from the superseded regen_smiles_ez pass, which no longer
    contributes to the database but was being kept alive purely as this
    function's home.
    """
    an, ae = rec["atoms"]["nodes"], rec["atoms"]["edges"]
    bt = ae["bond_type"]
    return {
        "element": an["element"]["value"],
        "stereo": an["stereo"]["value"],
        "charge": an["charge"]["value"],
        "bond_type": bt["value"],
        "bond_src": bt["src_indices"],
        "bond_dst": bt["dst_indices"],
        "bond_stereo": ae["bond_stereo"]["value"],
    }


def dihedral(p0, p1, p2, p3):
    """Torsion angle in degrees across p1-p2, or None if degenerate."""
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


def topology(src):
    """Molecule from element / charge / bond_type only -- no stereo seeded.

    The bare graph, used wherever stereochemistry must be decided from another
    source (coordinates, or a CCD label) rather than inherited.
    """
    rw = RWMol()
    for el, ch in zip(src["element"], src["charge"]):
        a = Chem.Atom(element_map(str(el)))
        a.SetFormalCharge(charge_map(str(ch)))
        rw.AddAtom(a)
    for v, s, d in zip(src["bond_type"], src["bond_src"], src["bond_dst"]):
        rw.AddBond(int(s), int(d), BOND_MAP.get(str(v), Chem.BondType.SINGLE))
    SanitizeMol(rw, SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_PROPERTIES)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    FastFindRings(mol)
    return mol
