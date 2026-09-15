"""How much of our lDDT deficit is broken local geometry rather than a wrong fold?

OST's `--lddt` runs stereochemistry checks and DROPS residues that fail them, so a
model with impossible bond lengths scores near zero even when its fold is roughly
right. `--lddt-no-stereochecks` keeps every residue. TM-score is computed by the
built-in USalign and takes no such filter, so it is the control: if lDDT moves and TM
does not, the deficit is geometry, not topology.

The stereochecked number is the one FoldBench reports and the one to quote -- every
published method is scored the same way. This is a diagnosis, not a rescoring.

CPU only, but it fans out OST containers: run it as a batch job.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FB = Path("/home/psk6950/data/foldbench")
OST = FB / "upstream" / "bin" / "ost"
BASE = ("{ost} compare-structures -m {m} -r {r} -o {o} --fault-tolerant "
        "--min-pep-length 4 --min-nuc-length 4 --lddt --rigid-scores --tm-score {extra}")


def run(job):
    m, r, o, extra = job
    if o.exists():
        return
    o.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(BASE.format(ost=OST, m=m, r=r, o=o, extra=extra),
                   shell=True, check=False, capture_output=True, executable="/bin/bash")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path,
                    default=Path("runs/foldbench/v1.0.1-phase2b-swa-full"))
    ap.add_argument("--out", type=Path,
                    default=Path("eval_results/foldbench/stereo_audit"))
    ap.add_argument("--per-category", type=int, default=25)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    cats = ["monomer_protein", "monomer_rna", "monomer_dna", "interface_protein_protein",
            "interface_protein_dna", "interface_protein_ligand"]
    rng = random.Random(0)
    jobs, index = [], []
    for cat in cats:
        tg = sorted({r["pdb_id"] for r in
                     csv.DictReader(open(FB / "targets" / f"{cat}.csv"))})
        have = [t for t in tg if (args.pred_dir / t /
                                  f"{t}_seed0_sample0_pred.cif").exists()]
        rng.shuffle(have)
        for t in have[:args.per_category]:
            m = (args.pred_dir / t / f"{t}_seed0_sample0_pred.cif").resolve()
            r = FB / "ground_truth" / f"{t}.cif"
            for tag, extra in (("with", ""), ("without", "--lddt-no-stereochecks")):
                o = args.out / cat / f"{t}_{tag}.json"
                jobs.append((m, r, o, extra))
                index.append((cat, t, tag, o))
        print(f"  {cat}: {len(have[:args.per_category])} targets")

    print(f"{len(jobs)} OST runs")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, jobs))

    got = defaultdict(dict)
    for cat, t, tag, o in index:
        if not o.exists():
            continue
        try:
            d = json.loads(o.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("status") == "SUCCESS":
            got[(cat, t)][tag] = d

    print(f"\n{'category':<28}{'n':>4}{'lDDT with':>11}{'lDDT w/o':>10}{'delta':>8}"
          f"{'TM with':>9}{'TM w/o':>9}{'bad bonds':>11}{'residues':>10}")
    print("-" * 100)
    for cat in cats:
        rows = [v for (c, _t), v in got.items() if c == cat and {"with", "without"} <= v.keys()]
        if not rows:
            continue
        g = lambda tag, k: [x[tag][k] for x in rows if x[tag].get(k) is not None]  # noqa: E731
        nres = [len(x["with"].get("aln", [[]])[0][1]) if x["with"].get("aln") else 0
                for x in rows]
        print(f"{cat:<28}{len(rows):>4}{statistics.mean(g('with','lddt')):>11.3f}"
              f"{statistics.mean(g('without','lddt')):>10.3f}"
              f"{statistics.mean(g('without','lddt')) - statistics.mean(g('with','lddt')):>+8.3f}"
              f"{statistics.mean(g('with','tm_score')):>9.3f}"
              f"{statistics.mean(g('without','tm_score')):>9.3f}"
              f"{statistics.mean(len(x['with'].get('model_bad_bonds', [])) for x in rows):>11.1f}"
              f"{statistics.mean(nres):>10.0f}")


if __name__ == "__main__":
    main()
