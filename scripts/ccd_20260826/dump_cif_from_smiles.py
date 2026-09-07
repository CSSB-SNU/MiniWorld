"""Build a CIF from the stored rdkit_smiles alone, to check it against the record.

dump_examples.py writes CIFs from the record's own arrays. This writes them from
the SMILES: the string is parsed with RDKit and the atom/bond tables are derived
from the resulting molecule, with nothing read from the record except the atom
names (needed to make the two files comparable) and the coordinates (a SMILES
carries none).

Atom names are transferred by canonical-rank matching, which is valid because the
two molecules have the same graph -- the script verifies that first and refuses
otherwise.

A `_miniworld_smiles_vs_record` loop then compares, per atom and per bond, what
the SMILES says against what the record stores, with an agreement flag, so
disagreement is visible on the page rather than needing a separate diff.

Usage:
    python dump_cif_from_smiles.py --ids 00U,004,5YI --out examples_from_smiles
"""

import argparse
import json
from pathlib import Path

import lmdb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MolToSmiles

from biomol_codec import load_bytes
from cif_writer import emit_keyvalue, emit_loop
from gen_record import record_to_src, topology

RDLogger.DisableLog("rdApp.*")
DB = "/public_data/bsoohyuncd/preprocessed_CCD_20260826.lmdb"
ORDER_NAME = {
    Chem.BondType.SINGLE: "SING",
    Chem.BondType.DOUBLE: "DOUB",
    Chem.BondType.TRIPLE: "TRIP",
    Chem.BondType.AROMATIC: "AROM",
    Chem.BondType.DATIVE: "DATI",
    Chem.BondType.OTHER: "OTHE",
}


def flat(mol):
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in m.GetBonds():
        b.SetStereo(Chem.BondStereo.STEREONONE)
    return MolToSmiles(m)


def map_smiles_to_record(mol_s, mol_r):
    """{smiles atom idx: record atom idx} via canonical ranks, or None."""
    if flat(mol_s) != flat(mol_r):
        return None
    rs = list(Chem.CanonicalRankAtoms(mol_s, breakTies=True, includeChirality=False))
    rr = list(Chem.CanonicalRankAtoms(mol_r, breakTies=True, includeChirality=False))
    by_rank = {r: i for i, r in enumerate(rr)}
    if len(by_rank) != mol_r.GetNumAtoms():
        return None
    try:
        return {i: by_rank[r] for i, r in enumerate(rs)}
    except KeyError:
        return None


def embed_coords(mol_s):
    """Generate a 3D conformer from the SMILES alone.

    Hydrogens are added for the embedding (geometry is poor without them) and
    stripped again afterwards, so the returned coordinates line up 1:1 with the
    heavy atoms of `mol_s`. Seeded, so the output is reproducible.

    Returns (positions, stereo_preserved) or (None, None) if embedding fails.
    stereo_preserved re-reads the stereochemistry back out of the generated
    geometry and compares it with what the SMILES specified -- if the conformer
    realizes the SMILES, the CIP labels must round-trip.
    """
    want = {}
    Chem.AssignCIPLabels(mol_s)
    for a in mol_s.GetAtoms():
        c = a.GetPropsAsDict().get("_CIPCode")
        if c:
            want[a.GetIdx()] = c
    mh = Chem.AddHs(Chem.Mol(mol_s))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mh, params) != 0:
        return None, None
    try:
        AllChem.MMFFOptimizeMolecule(mh, maxIters=2000)
    except Exception:  # noqa: BLE001
        pass
    heavy = Chem.RemoveHs(mh)
    if heavy.GetNumAtoms() != mol_s.GetNumAtoms():
        return None, None
    Chem.AssignStereochemistryFrom3D(heavy)
    try:
        Chem.AssignCIPLabels(heavy)
    except Exception:  # noqa: BLE001
        return heavy.GetConformer().GetPositions(), None
    got = {}
    for a in heavy.GetAtoms():
        c = a.GetPropsAsDict().get("_CIPCode")
        if c:
            got[a.GetIdx()] = c
    return heavy.GetConformer().GetPositions(), (got == want)


def build(cid, rec):
    smi = str(rec["residues"]["nodes"]["rdkit_smiles"]["value"][0])
    mol_s = Chem.MolFromSmiles(smi)
    if mol_s is None:
        return None, "stored SMILES does not parse"
    mol_r = topology(record_to_src(rec))
    mapping = map_smiles_to_record(mol_s, mol_r)
    if mapping is None:
        return None, "graph from SMILES differs from the record's graph"

    ids = [str(x) for x in rec["atoms"]["nodes"]["id"]["value"]]
    generated, stereo_ok = embed_coords(mol_s)
    if generated is None:
        return None, "could not generate a conformer from the SMILES"
    rec_elem = [str(x) for x in rec["atoms"]["nodes"]["element"]["value"]]
    rec_charge = [str(x) for x in rec["atoms"]["nodes"]["charge"]["value"]]
    rec_stereo = [str(x) for x in rec["atoms"]["nodes"]["stereo"]["value"]]

    # CIP as the SMILES encodes it
    Chem.AssignCIPLabels(mol_s)
    s_cip = {a.GetIdx(): a.GetPropsAsDict().get("_CIPCode") for a in mol_s.GetAtoms()}
    s_bond_cip = {}
    for b in mol_s.GetBonds():
        c = b.GetPropsAsDict().get("_CIPCode")
        if c:
            s_bond_cip[frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))] = c

    # Kekulize so bond orders are comparable with the record's SING/DOUB
    kek = Chem.Mol(mol_s)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        pass

    out = [f"data_{cid}", "#"]
    emit_keyvalue(out, "_chem_comp", [
        ("id", cid),
        ("name", str(rec["residues"]["nodes"]["id"]["value"][0])),
        ("formula", str(rec["residues"]["nodes"]["formula"]["value"][0])),
    ])
    emit_keyvalue(out, "_miniworld_source", [
        ("built_from", "residues.nodes.rdkit_smiles (parsed with RDKit)"),
        ("rdkit_smiles", smi),
        ("atom_names_from", "record, matched by canonical rank"),
        ("coordinates_from",
         "generated from the SMILES by RDKit ETKDGv3 + MMFF (seeded)"),
        ("generated_geometry_reproduces_smiles_stereo",
         {True: "yes", False: "NO", None: "not checked"}[stereo_ok]),
        ("source_lmdb", DB),
    ])

    rows = []
    for i in range(mol_s.GetNumAtoms()):
        r = mapping[i]
        a = mol_s.GetAtomWithIdx(i)
        rows.append([
            cid, ids[r], a.GetSymbol().upper(), str(a.GetFormalCharge()),
            "Y" if a.GetIsAromatic() else "N",
            s_cip.get(i) or "N",
            f"{generated[i][0]:.3f}", f"{generated[i][1]:.3f}", f"{generated[i][2]:.3f}",
            i + 1,
        ])
    emit_loop(out, "_chem_comp_atom", [
        "comp_id", "atom_id", "type_symbol", "charge", "pdbx_aromatic_flag",
        "pdbx_stereo_config", "model_Cartn_x", "model_Cartn_y", "model_Cartn_z",
        "pdbx_ordinal",
    ], rows)

    brows = []
    for n, b in enumerate(kek.GetBonds(), 1):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        key = frozenset((i, j))
        brows.append([
            cid, ids[mapping[i]], ids[mapping[j]],
            ORDER_NAME.get(b.GetBondType(), "OTHE"),
            "Y" if mol_s.GetBondBetweenAtoms(i, j).GetIsAromatic() else "N",
            s_bond_cip.get(key, "N"), n,
        ])
    emit_loop(out, "_chem_comp_bond", [
        "comp_id", "atom_id_1", "atom_id_2", "value_order",
        "pdbx_aromatic_flag", "pdbx_stereo_config", "pdbx_ordinal",
    ], brows)

    # ---- side-by-side comparison against the record ----
    cmp_rows = []
    for i in range(mol_s.GetNumAtoms()):
        r = mapping[i]
        a = mol_s.GetAtomWithIdx(i)
        got_el, want_el = a.GetSymbol().upper(), rec_elem[r].upper()
        got_ch, want_ch = str(a.GetFormalCharge()), rec_charge[r]
        got_st = s_cip.get(i) or "N"
        want_st = rec_stereo[r] if rec_stereo[r] in ("R", "S") else "N"
        agree = (got_el == want_el) and (got_ch == want_ch)
        cmp_rows.append([
            cid, "atom", ids[r], "-",
            f"{got_el}/{want_el}", f"{got_ch}/{want_ch}", f"{got_st}/{want_st}",
            "Y" if agree else "N",
            "Y" if got_st == want_st else "N",
        ])
    rec_bonds = {}
    bt = rec["atoms"]["edges"]["bond_type"]
    bstereo = rec["atoms"]["edges"]["bond_stereo"]
    barom = rec["atoms"]["edges"]["bond_aromatic"]
    for v, s, d, st, ar in zip(
        bt["value"], bt["src_indices"], bt["dst_indices"], bstereo["value"], barom["value"]
    ):
        rec_bonds[frozenset((int(s), int(d)))] = (str(v), str(st), str(ar))
    for b in kek.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        rk = frozenset((mapping[i], mapping[j]))
        got_o = ORDER_NAME.get(b.GetBondType(), "OTHE")
        want_o, want_st, want_ar = rec_bonds.get(rk, ("absent", "absent", "N"))
        got_st = s_bond_cip.get(frozenset((i, j)), "N")
        want_st = want_st if want_st in ("E", "Z") else "N"
        # An aromatic ring has several equivalent Kekule forms and RDKit picks its
        # own when re-kekulizing a lowercase aromatic SMILES, so comparing
        # SING/DOUB there reports differences that are not differences. Where both
        # sides call the bond aromatic, compare aromaticity instead of order.
        got_ar = "Y" if mol_s.GetBondBetweenAtoms(i, j).GetIsAromatic() else "N"
        if got_ar == "Y" and want_ar == "Y":
            shown, agrees = "AROM/AROM", True
        else:
            shown, agrees = f"{got_o}/{want_o}", got_o == want_o
        cmp_rows.append([
            cid, "bond", ids[mapping[i]], ids[mapping[j]],
            shown, "-", f"{got_st}/{want_st}",
            "Y" if agrees else "N",
            "Y" if got_st == want_st else "N",
        ])
    emit_loop(out, "_miniworld_smiles_vs_record", [
        "comp_id", "kind", "atom_id_1", "atom_id_2",
        "element_or_order_smiles_vs_record", "charge_smiles_vs_record",
        "stereo_smiles_vs_record", "topology_agrees", "stereo_agrees",
    ], cmp_rows)

    return "\n".join(out) + "\n", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", default="examples_from_smiles")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(DB, readonly=True, lock=False, readahead=False)
    manifest = []
    for cid in args.ids.split(","):
        with env.begin() as t:
            raw = t.get(cid.encode())
        if raw is None:
            print(f"  {cid}: not in database")
            continue
        rec = load_bytes(bytes(raw))
        text, err = build(cid, rec)
        if err:
            print(f"  {cid}: SKIPPED -- {err}")
            continue
        path = outdir / f"fromsmiles_{cid}.cif"
        path.write_text(text)
        n_atom_bad = text.count(" N Y\n") + 0
        disagree = sum(
            1 for line in text.split("\n")
            if line.startswith(cid + " atom") or line.startswith(cid + " bond")
        )
        print(f"  wrote {path.name}")
        manifest.append({"comp_id": cid, "file": path.name})
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} files in {outdir}")


if __name__ == "__main__":
    main()
