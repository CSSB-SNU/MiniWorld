"""Peptide C-N bond (and CA-CA) of the saved x0-hat trajectory, per solver step. Atom identities come from the predicted CIF of the same run (atom order = batch atom order);
the npz (x0hat_inference.py) holds x0-hat [steps, L_pad, 3] and sigma [steps]. usage: traj_peptide.py NPZ PRED_CIF"""
import sys

import numpy as np
from Bio.PDB import MMCIFParser

npz, cif = sys.argv[1], sys.argv[2]
d = np.load(npz)
x0hat, sigma = d["x0hat"], d["sigma"]
model = next(iter(MMCIFParser(QUIET=True).get_structure("x", cif)))
atoms = list(model.get_atoms())
n = len(atoms)
idx = {id(a): k for k, a in enumerate(atoms)}
cn, caca = [], []
for chain in model:
    res = [r for r in chain if r.id[0] == " " and "CA" in r]
    for a, b in zip(res, res[1:]):
        if "C" in a and "N" in b:
            cn.append((idx[id(a["C"])], idx[id(b["N"])]))
        caca.append((idx[id(a["CA"])], idx[id(b["CA"])]))
cn, caca = np.array(cn), np.array(caca)
print(f"{npz}: atoms in CIF {n}, x0hat atoms {x0hat.shape[1]}, steps {x0hat.shape[0]}, peptide bonds {len(cn)}")
assert x0hat.shape[1] >= n
X = x0hat[:, :n]
dcn = np.linalg.norm(X[:, cn[:, 0]] - X[:, cn[:, 1]], axis=-1)       # [steps, n_bonds]
dca = np.linalg.norm(X[:, caca[:, 0]] - X[:, caca[:, 1]], axis=-1)
dev = np.abs(dcn - 1.329)
print(f"{'step':>5}{'sigma':>10}{'C-N mean':>10}{'|dev|':>8}{'>0.1A':>8}{'>0.3A':>8}{'CA-CA out 3.6-4.0':>20}")
for s in sorted(set(list(range(0, X.shape[0], 10)) + [X.shape[0] - 1])):
    out = ((dca[s] < 3.6) | (dca[s] > 4.0)).mean()
    print(f"{s:5d}{sigma[s]:10.3f}{dcn[s].mean():10.3f}{dev[s].mean():8.3f}{(dev[s] > 0.1).mean():8.2f}{(dev[s] > 0.3).mean():8.2f}{out:20.2f}")
print("TRAJ_PEPTIDE_DONE")
