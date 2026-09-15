"""FoldBench success rates under both denominator conventions.

FoldBench's published numbers divide by the interfaces each method has a score for
(verified: reproducing that convention matches the README leaderboard to 0.01 on all
20 protein/NA interface values). Interfaces a method has no row for are dropped, so a
method that failed to predict something is not charged for it.

The strict convention charges a method for what it failed to produce. The missing rows
split into two kinds, which the overlap tells apart:

* rows missing for EVERY method -- 11 of 558 protein-ligand interfaces, 13 of 279
  protein-protein, 5 of 172 antibody-antigen. Nobody has a score, so these were never
  evaluated; they are dropped from the denominator for everyone.
* rows missing for only SOME methods -- protein-ligand: HF3 80, Boltz-1 71, Protenix
  48, Chai-1 20, AF3 0. The other methods scored these same interfaces, so the absence
  is that method's own missing prediction and counts as a failure.

Ours is scored the same way in both. Under the strict convention our not-yet-run
targets count as failures too, so that column is a lower bound until the sweep ends.
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

FB = Path("/home/psk6950/data/foldbench")
MODELS = ["af3", "Protenix", "Boltz-1", "Chai-1", "HF3"]
CATS = [
    ("interface_protein_protein", "ost"), ("interface_protein_ligand", "lig"),
    ("interface_protein_peptide", "ost"), ("interface_antibody_antigen", "ost"),
    ("interface_protein_dna", "dockqv2"), ("interface_protein_rna", "dockqv2"),
]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def paper_sections():
    rows = list(csv.reader((FB / "meta/paper_scores/Supplementary_Table_3.csv").open()))
    starts = [i for i, r in enumerate(rows) if r and r[0] and not any(r[1:])]
    out = {}
    for n, i in enumerate(starts):
        hdr = rows[i + 1]
        end = starts[n + 1] if n + 1 < len(starts) else len(rows)
        out[rows[i][0]] = [dict(zip(hdr, r, strict=False)) for r in rows[i + 2:end] if r and r[0]]
    return out


def ours_successes(raw_dir: Path, cat: str, src: str,
                   allowed: set | None = None) -> tuple[int, int, int, int]:
    """(worst, @1, best, interfaces scored) success counts.

    The three columns bracket every possible ranking head, since all it can do is
    choose one of the 25 samples:

    * ours@worst -- the interface counts only if EVERY sample succeeds. This is what
      an adversarially bad ranker would score, and the true lower bound.
    * ours@1     -- one arbitrary sample. What we actually have today, with no
      confidence head to pick with, and the honest comparison against the published
      methods' ranked top-1.
    * ours@best  -- the interface counts if ANY sample succeeds: a perfect oracle
      ranker, and the upper bound.

    ``allowed`` restricts the count to the same interface set the denominator uses.
    Without it the strict convention counted our successes on interfaces it had just
    removed from the denominator (the ones no published method scored), which
    inflated our rate.
    """
    p = raw_dir / f"{cat}_{'dockqv2' if src == 'dockqv2' else 'ost'}.csv"
    if not p.exists():
        return 0, 0, 0, 0
    by = collections.defaultdict(list)
    with p.open() as f:
        for r in csv.DictReader(f):
            k = (r["pdb_id"],
                 r.get("interface_chain_id_1") or r.get("native_chain_id_1"),
                 r.get("interface_chain_id_2") or r.get("native_chain_id_2"))
            if src == "lig":
                a, b = fnum(r.get("rmsd")), fnum(r.get("lddt-pli"))
                if a is not None and b is not None:
                    by[k].append(a < 2.0 and b > 0.8)
            else:
                v = fnum(r.get("dockq_score"))
                if v is not None:
                    by[k].append(v >= 0.23)
    if allowed is not None:
        by = {k: v for k, v in by.items() if k in allowed}
    return (sum(1 for v in by.values() if all(v)),
            sum(1 for v in by.values() if v[0]),
            sum(1 for v in by.values() if any(v)),
            len(by))


def bench_keys(cat: str) -> set:
    with (FB / "targets" / f"{cat}.csv").open() as f:
        return {(r["pdb_id"],
                 r.get("interface_chain_id_1") or r.get("native_chain_id_1"),
                 r.get("interface_chain_id_2") or r.get("native_chain_id_2"))
                for r in csv.DictReader(f)}


def covered_keys(sec, cat: str, model: str) -> set:
    return {(r["pdb_id"], r.get("interface_Chai-1n_id_1"), r.get("interface_Chai-1n_id_2"))
            for r in sec[cat] if r["model"] == model}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path("eval_results/foldbench/p2b-swa-full-partial/raw"))
    args = ap.parse_args()
    sec = paper_sections()

    for title, strict in (("A. FoldBench 공식 규약 (커버한 것만 분모)", False),
                          ("B. 엄격 규약 (예측 못 낸 것 = 실패, 분모는 전체)", True)):
        print(f"\n{title}")
        head = "%-27s %6s %9s %9s %9s |" % (
            "category", "total", "ours@worst", "ours@1", "ours@best")
        head += "".join(m.rjust(8) for m in MODELS)
        print(head)
        print("-" * len(head))
        for cat, src in CATS:
            bench = bench_keys(cat)
            # interfaces no method scored were never evaluated -- out of everyone's
            # denominator, including ours.
            never = set.intersection(*(bench - covered_keys(sec, cat, m) for m in MODELS))
            allowed = (bench - never) if strict else bench
            total = len(allowed)
            sw, s1, sb, scored = ours_successes(args.raw, cat, src, allowed)
            den = total if strict else max(scored, 1)
            row = "%-27s %6d %8.1f%% %8.1f%% %8.1f%% |" % (
                cat, total, 100 * sw / den, 100 * s1 / den, 100 * sb / den)
            for m in MODELS:
                s = n = 0
                for r in sec[cat]:
                    if r["model"] != m:
                        continue
                    n += 1
                    if src == "lig":
                        a, b = fnum(r.get("rmsd")), fnum(r.get("lddt-pli"))
                        ok = a is not None and b is not None and a < 2.0 and b > 0.8
                    else:
                        v = fnum(r.get("dockq_score"))
                        ok = v is not None and v >= 0.23
                    s += ok
                row += "%7.1f " % (100 * s / (total if strict else max(n, 1)))
            if strict and never:
                row += "  (전 방법 미평가 %d 제외)" % len(never)
            print(row)
        if strict:
            print("  (ours 열은 아직 안 돌린 타겟도 실패로 세므로 하한)")


if __name__ == "__main__":
    main()
