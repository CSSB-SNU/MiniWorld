"""Chain / residue layout of a predicted CIF and its ground truth: chains, residue counts, first residues, C-N distance of consecutive residues in CIF order (where it jumps)."""
import sys

import numpy as np
from Bio.PDB import MMCIFParser

for path in sys.argv[1:]:
    model = next(iter(MMCIFParser(QUIET=True).get_structure("x", path)))
    print(f"\n=== {path}")
    for chain in model:
        res = [r for r in chain if r.id[0] == " "]
        het = [r for r in chain if r.id[0] != " "]
        ids = [r.id[1] for r in res]
        print(f"chain {chain.id}: {len(res)} residues (+{len(het)} hetero), resseq {ids[:6]}...{ids[-3:]}, monotonic={all(b > a for a, b in zip(ids, ids[1:]))}")
        cn = []
        for a, b in zip(res, res[1:]):
            if "C" in a and "N" in b:
                cn.append(float(np.linalg.norm(a["C"].coord - b["N"].coord)))
        cn = np.array(cn)
        if len(cn):
            print(f"   C-N: n={len(cn)} median={np.median(cn):.3f} p90={np.percentile(cn, 90):.3f} max={cn.max():.2f} frac>2.0A={(cn > 2.0).mean():.2f} "
                  f"first 12: {np.round(cn[:12], 2).tolist()}")
print("INSPECT_DONE")
