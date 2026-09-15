"""Re-score the monomer DNA/RNA categories with and without OST's stereochemistry filter.

`--lddt` drops residues that fail bond/angle checks, so a model whose local geometry is
broken scores near zero even when its fold is partly right. The stereochecked column is
FoldBench's official number and the one to quote; the other column says how much of the
gap is geometry rather than topology. TM-score takes no such filter and is the control.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FB = Path("/home/psk6950/data/foldbench")
OST = FB / "upstream" / "bin" / "ost"
CMD = ("{ost} compare-structures -m {m} -r {r} -o {o} --fault-tolerant "
       "--min-pep-length 4 --min-nuc-length 4 --lddt --rigid-scores --tm-score "
       "--lddt-no-stereochecks")


def run(job):
    m, r, o = job
    if o.exists():
        return
    o.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(CMD.format(ost=OST, m=m, r=r, o=o), shell=True, check=False,
                   capture_output=True, executable="/bin/bash")


def paper_scores(cat):
    allr = list(csv.reader(open(FB / "meta/paper_scores/Supplementary_Table_3.csv")))
    st = [i for i, r in enumerate(allr) if r and r[0] and not any(r[1:])]
    for n, i in enumerate(st):
        if allr[i][0] != cat:
            continue
        hdr = allr[i + 1]
        end = st[n + 1] if n + 1 < len(st) else len(allr)
        out = defaultdict(dict)
        for r in allr[i + 2:end]:
            if not r or not r[0]:
                continue
            d = dict(zip(hdr, r))
            try:
                out[d["pdb_id"]][d["model"]] = float(d["lddt"])
            except (ValueError, KeyError):
                pass
        return out
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path,
                    default=Path("runs/foldbench/v1.0.1-phase2b-swa-full"))
    ap.add_argument("--official", type=Path,
                    default=Path("eval_results/foldbench/p2b-swa-full-partial/detail"))
    ap.add_argument("--out", type=Path,
                    default=Path("eval_results/foldbench/na_monomer_nostereo"))
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    models = ["af3", "Protenix", "Boltz-1", "Chai-1", "HF3"]
    for cat in ("monomer_dna", "monomer_rna"):
        tg = sorted({r["pdb_id"] for r in csv.DictReader(open(FB / "targets" / f"{cat}.csv"))})
        jobs, idx = [], []
        for t in tg:
            for cif in sorted((args.pred_dir / t).glob("*_pred.cif")) if (args.pred_dir / t).is_dir() else []:
                o = args.out / cat / f"{cif.stem}.json"
                jobs.append((cif.resolve(), FB / "ground_truth" / f"{t}.cif", o))
                idx.append((t, cif.stem, o))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(run, jobs))

        per = defaultdict(lambda: {"off": [], "on": [], "tm": []})
        for t, stem, o in idx:
            if o.exists():
                d = json.loads(o.read_text())
                if d.get("status") == "SUCCESS":
                    if d.get("lddt") is not None:
                        per[t]["off"].append(d["lddt"])
                    if d.get("tm_score") is not None:
                        per[t]["tm"].append(d["tm_score"])
            p = args.official / f"{stem.replace('_pred','')}_structure_ost.json"
            p = args.official / f"{t}_{stem.split('seed')[1].split('_')[0]}_{stem.split('sample')[1].split('_')[0]}_structure_ost.json"
            if p.exists():
                d = json.loads(p.read_text())
                if d.get("status") == "SUCCESS" and d.get("lddt") is not None:
                    per[t]["on"].append(d["lddt"])

        pap = paper_scores(cat)
        print(f"\n=== {cat} ===")
        print("%-18s %5s | %9s %9s | %9s %9s | %6s %6s %7s %6s %5s" % (
            "pdb_id", "n", "lDDT off", "lDDT on", "TM med", "TM max",
            "af3", "Ptnx", "Boltz-1", "Chai", "HF3"))
        print("-" * 108)
        agg = {"on": [], "off": [], "tm": []}
        for t in tg:
            v = per.get(t)
            if not v or not v["off"]:
                print("%-18s %5s | (not predicted yet)" % (t.split("-")[0], "-"))
                continue
            on = statistics.median(v["on"]) if v["on"] else float("nan")
            off = statistics.median(v["off"])
            agg["on"].append(on); agg["off"].append(off); agg["tm"].append(statistics.median(v["tm"]))
            p = pap.get(t, {})
            row = "%-18s %5d | %9.3f %9.3f | %9.3f %9.3f |" % (
                t.split("-")[0], len(v["off"]), off, on,
                statistics.median(v["tm"]), max(v["tm"]))
            for m in models:
                row += ("%6.2f" % p[m]) if p.get(m) is not None else "   n/a"
            print(row)
        if agg["on"]:
            print("-" * 108)
            print("%-18s %5s | %9.3f %9.3f | %9.3f" % (
                "MEAN", len(agg["on"]), statistics.mean(agg["off"]),
                statistics.mean(x for x in agg["on"] if x == x), statistics.mean(agg["tm"])))


if __name__ == "__main__":
    main()
