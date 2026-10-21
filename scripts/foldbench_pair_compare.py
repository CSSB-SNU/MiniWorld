"""Paired comparison of two FoldBench scorings of the same task on their common interfaces.

Usage: foldbench_pair_compare.py --eval-dir DIR --a NAME_A --b NAME_B --task interface_protein_rna

Besides the reduced numbers of foldbench_compare.py (ours@1 picks an arbitrary sample, so it is noisy), this
reports the mean over ALL samples (every seed x sample of an interface counts), which is the stable estimate
of single-sample quality, plus per-interface win/tie/loss on the best-of-25 and the mean-of-25 DockQ.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

NUC = {"interface_protein_dna", "interface_protein_rna"}


def load(eval_dir: Path, name: str, task: str):
    raw = eval_dir / name / "raw"
    src = raw / f"{task}_dockqv2.csv" if task in NUC and (raw / f"{task}_dockqv2.csv").exists() else raw / f"{task}_ost.csv"
    with src.open() as f:
        rows = list(csv.DictReader(f))
    lddt = {}
    ost = raw / f"{task}_ost.csv"
    if ost.exists():
        with ost.open() as f:
            lddt = {(r["pdb_id"], r["interface_chain_id_1"], r["interface_chain_id_2"], r["seed"], r["sample"]): r.get("lddt") for r in csv.DictReader(f)}
    out = defaultdict(list)
    for r in rows:
        k = (r["pdb_id"], r["interface_chain_id_1"], r["interface_chain_id_2"])
        f = lambda v: float(v) if v not in (None, "") else None
        out[k].append(dict(dockq=f(r.get("dockq_score")), irmsd=f(r.get("irmsd")), lrmsd=f(r.get("lrmsd")),
                           lddt=f(lddt.get(k + (r["seed"], r["sample"])) or r.get("lddt"))))
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def stats(d, keys):
    allr = [r for k in keys for r in d[k]]
    dq = [r["dockq"] for r in allr if r["dockq"] is not None]
    best = [max(r["dockq"] for r in d[k] if r["dockq"] is not None) for k in keys if any(r["dockq"] is not None for r in d[k])]
    per_mean = {k: mean([r["dockq"] for r in d[k]]) for k in keys}
    return dict(
        success_all=100 * sum(x >= 0.23 for x in dq) / len(dq),
        success_best=100 * sum(x >= 0.23 for x in best) / len(best),
        dockq_mean=mean(dq), irmsd=mean([r["irmsd"] for r in allr]), lrmsd=mean([r["lrmsd"] for r in allr]),
        lddt=mean([r["lddt"] for r in allr]), per_mean=per_mean,
        best={k: max((r["dockq"] for r in d[k] if r["dockq"] is not None), default=None) for k in keys},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--a", required=True, help="baseline eval name")
    ap.add_argument("--b", required=True, help="new eval name")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    da, db = load(args.eval_dir, args.a, args.task), load(args.eval_dir, args.b, args.task)
    keys = sorted(set(da) & set(db))
    sa, sb = stats(da, keys), stats(db, keys)
    lines = [f"{args.task}: {len(keys)} common interfaces over {len({k[0] for k in keys})} targets "
             f"(A={args.a}: {len(da)} interfaces, B={args.b}: {len(db)})",
             f"  {'metric':34}{'A':>10}{'B':>10}{'B-A':>10}"]
    for label, key in [("DockQ success % (all samples)", "success_all"), ("DockQ success % (best of 25)", "success_best"),
                       ("mean DockQ (all samples)", "dockq_mean"), ("iRMSD (all samples)", "irmsd"),
                       ("LRMSD (all samples)", "lrmsd"), ("lDDT (all samples)", "lddt")]:
        lines.append(f"  {label:34}{sa[key]:10.2f}{sb[key]:10.2f}{sb[key]-sa[key]:+10.2f}")
    for label, field in [("mean-of-25 DockQ", "per_mean"), ("best-of-25 DockQ", "best")]:
        w = t = l = 0
        for k in keys:
            a, b = sa[field][k], sb[field][k]
            if a is None or b is None:
                continue
            if abs(b - a) < 0.02:
                t += 1
            elif b > a:
                w += 1
            else:
                l += 1
        lines.append(f"  per-interface {label}: B better {w}, tie(|d|<0.02) {t}, A better {l}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
