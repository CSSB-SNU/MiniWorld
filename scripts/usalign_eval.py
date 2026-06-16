"""Re-score sampled structures with US-align (chain-mapping-aware RMSD/TM-score).

The inference scripts log ``cal_aligned_rmsd``, a single Kabsch fit over all
atoms in fixed order. For homomers / multimers that is wrong: identical chains
can be predicted in a permuted order, so a structurally correct prediction gets
a huge RMSD. US-align (``-mm 1 -ter 0``) finds the optimal chain-to-chain
mapping first, then superposes, giving the meaningful complex RMSD + TM-score.

Per target it scans ``<dir>/structures`` for ``<name>_gt.cif`` and the matching
``<name>_pred_<k>.cif``, runs US-align on each, and prints a table with the
best sample (lowest RMSD and highest TM-score, reported separately).

Usage:
    python scripts/usalign_eval.py outputs/.../edm0500_protein_noema \
        [more_dirs ...] [--usalign tools/USalign/USalign]
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

RMSD_RE = re.compile(r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+)")
# Two TM-score lines are printed; the reference is Structure_2 (the GT).
TM_REF_RE = re.compile(r"TM-score=\s*([\d.]+).*normalized by length of Structure_2")


def run_usalign(usalign: Path, pred: Path, gt: Path) -> dict | None:
    """Run US-align (multimer, all chains) and parse RMSD / TM / aligned len."""
    out = subprocess.run(
        [str(usalign), str(pred), str(gt), "-mm", "1", "-ter", "0"],
        capture_output=True, text=True, check=False,
    ).stdout
    m_rmsd = RMSD_RE.search(out)
    m_tm = TM_REF_RE.search(out)
    if not m_rmsd or not m_tm:
        return None
    return {
        "aligned": int(m_rmsd.group(1)),
        "rmsd": float(m_rmsd.group(2)),
        "tm": float(m_tm.group(1)),
    }


def eval_dir(run_dir: Path, usalign: Path) -> list[dict]:
    struct_dir = run_dir / "structures"
    rows: list[dict] = []
    # Filenames contain '[' / ']' (e.g. "['8G2C']_..."), which are glob
    # metacharacters, so match pred files by string prefix, not glob pattern.
    all_cifs = sorted(struct_dir.iterdir())
    for gt in [p for p in all_cifs if p.name.endswith("_gt.cif")]:
        prefix = gt.name[: -len("_gt.cif")]
        preds = [p for p in all_cifs
                 if p.name.startswith(f"{prefix}_pred_") and p.name.endswith(".cif")]
        samples = []
        for pred in preds:
            res = run_usalign(usalign, pred, gt)
            if res is not None:
                k = pred.name.split("_pred_")[-1].split(".cif")[0]
                samples.append({"k": k, **res})
        if samples:
            best_rmsd = min(samples, key=lambda s: s["rmsd"])
            best_tm = max(samples, key=lambda s: s["tm"])
            rows.append({
                "name": prefix,
                "n": len(samples),
                "best_rmsd": best_rmsd["rmsd"],
                "best_rmsd_tm": best_rmsd["tm"],
                "best_tm": best_tm["tm"],
                "aligned": best_rmsd["aligned"],
                "samples": samples,
            })
        else:
            rows.append({"name": prefix, "n": 0, "samples": []})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="Run dir(s) containing a structures/ subfolder.")
    ap.add_argument("--usalign", type=Path,
                    default=Path("tools/USalign/USalign"))
    args = ap.parse_args()

    for run_dir in args.run_dirs:
        rows = eval_dir(run_dir, args.usalign)
        print(f"\n=== {run_dir}  (US-align -mm 1 -ter 0, best-of-N) ===")
        print(f"{'target':<22} {'n':>2} {'bestRMSD':>9} {'TM@best':>8} "
              f"{'bestTM':>7} {'aln':>5}")
        for r in rows:
            if r["n"] == 0:
                print(f"{r['name']:<22} {'--':>2}  (US-align failed)")
                continue
            print(f"{r['name']:<22} {r['n']:>2} {r['best_rmsd']:>9.2f} "
                  f"{r['best_rmsd_tm']:>8.4f} {r['best_tm']:>7.4f} "
                  f"{r['aligned']:>5}")


if __name__ == "__main__":
    main()
