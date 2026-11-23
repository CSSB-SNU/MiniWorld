"""Independent geometry check of predicted / ground-truth mmCIFs (Biopython only, atom names and residue ids from the files themselves):
peptide C-N length, CA-CA(i,i+1) distance, backbone/CB intra-residue bond lengths, and heavy-atom clashes (< 2.0 A between atoms of residues that are not sequence neighbours).
usage: cif_geometry.py LABEL=PATH_OR_GLOB [LABEL=PATH_OR_GLOB ...]   (several files under one LABEL are averaged)"""
import glob
import sys

import numpy as np
from Bio.PDB import MMCIFParser, NeighborSearch

IDEAL = {("N", "CA"): 1.458, ("CA", "C"): 1.525, ("C", "O"): 1.231, ("CA", "CB"): 1.530}
PEPTIDE_CN = 1.329


def geometry(path):
    s = MMCIFParser(QUIET=True).get_structure("x", path)
    model = next(iter(s))
    pep, caca, intra, n_res = [], [], [], 0
    clash_pairs = []
    atoms = []
    for chain in model:
        res = [r for r in chain if r.id[0] == " " and "CA" in r]
        n_res += len(res)
        for a, b in zip(res, res[1:]):
            if "C" in a and "N" in b:
                pep.append(float(np.linalg.norm(a["C"].coord - b["N"].coord)))
            caca.append(float(np.linalg.norm(a["CA"].coord - b["CA"].coord)))
        for r in res:
            for (x, y), ideal in IDEAL.items():
                if x in r and y in r:
                    intra.append(abs(float(np.linalg.norm(r[x].coord - r[y].coord)) - ideal))
        atoms += [a for r in chain for a in r if r.id[0] == " " and (a.element or "").upper() not in ("H", "D")]
    index = {}
    for ci, chain in enumerate(model):
        for k, r in enumerate(x for x in chain if x.id[0] == " "):
            index[id(r)] = (ci, k)
    for a1, a2 in NeighborSearch(atoms).search_all(2.0):
        r1, r2 = a1.get_parent(), a2.get_parent()
        if r1 is r2:
            continue
        c1, k1 = index[id(r1)]
        c2, k2 = index[id(r2)]
        if c1 == c2 and abs(k1 - k2) <= 1:
            continue
        if a1.get_id() == "SG" and a2.get_id() == "SG":
            continue
        clash_pairs.append((a1, a2))
    pep, caca, intra = np.array(pep), np.array(caca), np.array(intra)
    return {
        "residues": n_res, "atoms": len(atoms),
        "pep_CN_mean": pep.mean() if len(pep) else np.nan, "pep_CN_|d|>0.1": (np.abs(pep - PEPTIDE_CN) > 0.1).mean() if len(pep) else np.nan,
        "CACA_outside_3.6-4.0": ((caca < 3.6) | (caca > 4.0)).mean() if len(caca) else np.nan,
        "intra_mean_|d|": intra.mean() if len(intra) else np.nan, "intra_>0.1": (intra > 0.1).mean() if len(intra) else np.nan,
        "clash_pairs<2.0": len(clash_pairs),
    }


rows = []
for spec in sys.argv[1:]:
    label, pattern = spec.split("=", 1)
    files = sorted(glob.glob(pattern))
    gs = [geometry(f) for f in files]
    rows.append((label, len(files), {k: float(np.nanmean([g[k] for g in gs])) for k in gs[0]}))
keys = list(rows[0][2])
print(f"{'label':<34}{'files':>6}" + "".join(f"{k:>22}" for k in keys))
for label, n, g in rows:
    print(f"{label:<34}{n:>6}" + "".join(f"{g[k]:>22.3f}" for k in keys))
print("GEOMETRY_DONE")
