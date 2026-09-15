"""Put our FoldBench scores next to the paper's, on exactly the targets we predicted.

The FoldBench paper ships per-target, per-method scores (Source Data MOESM4) for
AlphaFold 3, Protenix, Boltz-1, Chai-1 and HelixFold 3; ``meta/paper_scores/
Supplementary_Table_3.csv`` holds them for all nine categories, one row per
(pdb_id, interface, method) already reduced to that method's submitted
prediction. Our own scores come out of ``scripts/foldbench_evaluate.py``.

The two are only comparable on the same interfaces, so this restricts the paper's
rows to the (pdb_id, chain_1, chain_2) keys we actually scored and recomputes each
method's category metric over that subset. A number here is therefore "on these N
interfaces", not the published category number.

Our column appears twice, because this checkpoint has no confidence head and so
cannot rank its own samples:

* ``ours@1`` — one arbitrary sample per interface. This is the honest comparison:
  the paper's methods submit the sample their own ranker chose, and an unranked
  model can only offer a random one.
* ``ours@best`` — the best of our N samples per interface, an oracle upper bound.
  It is what a working ranking head could reach, not what we have.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

FB = Path("/home/psk6950/data/foldbench")
PAPER = FB / "meta" / "paper_scores" / "Supplementary_Table_3.csv"

# metric -> (higher is better?, success predicate or None)
DOCKQ_CATS = {
    "interface_antibody_antigen", "interface_protein_protein",
    "interface_protein_peptide", "interface_protein_dna", "interface_protein_rna",
}
MONOMER_CATS = {"monomer_protein", "monomer_dna", "monomer_rna"}


def load_paper() -> dict[str, tuple[list[str], list[dict]]]:
    rows = list(csv.reader(PAPER.open()))
    starts = [i for i, r in enumerate(rows) if r and r[0] and not any(r[1:])]
    out = {}
    for n, idx in enumerate(starts):
        name = rows[idx][0]
        hdr = rows[idx + 1]
        end = starts[n + 1] if n + 1 < len(starts) else len(rows)
        recs = [dict(zip(hdr, r, strict=False)) for r in rows[idx + 2 : end] if r and r[0]]
        out[name] = (hdr, recs)
    return out


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def key_of(row: dict, cat: str) -> tuple:
    pid = row["pdb_id"]
    if cat in MONOMER_CATS:
        return (pid,)
    c1 = row.get("interface_chain_id_1") or row.get("native_chain_id_1") \
        or row.get("interface_Chai-1n_id_1")
    c2 = row.get("interface_chain_id_2") or row.get("native_chain_id_2") \
        or row.get("interface_Chai-1n_id_2")
    return (pid, c1, c2)


def reduce_ours(rows: list[dict], cat: str) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    """One row per interface: an arbitrary sample, and the best sample."""
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[key_of(r, cat)].append(r)
    first, best = {}, {}
    # "best" is selected by the same metric task_score_summary selects on:
    # DockQ (max) for interfaces, RMSD (min) for ligands, lDDT (max) for monomers.
    sel, hi = ("dockq_score", True) if cat in DOCKQ_CATS else \
              ("rmsd", False) if cat == "interface_protein_ligand" else ("lddt", True)
    for k, rs in by_key.items():
        first[k] = rs[0]
        scored = [r for r in rs if fnum(r.get(sel)) is not None]
        if scored:
            best[k] = (max if hi else min)(scored, key=lambda r: fnum(r[sel]))
    return first, best


def category_metrics(rows: list[dict], cat: str) -> dict[str, float | None]:
    def avg(field):
        # Our OST csv spells these gdt_ts / tm_score; the paper sheet uses
        # gdt-ts / tm-score. Accept either so one code path reads both.
        alt = field.replace("-", "_")
        v = [fnum(r.get(field, r.get(alt))) for r in rows]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    if cat in DOCKQ_CATS:
        d = [fnum(r.get("dockq_score")) for r in rows]
        d = [x for x in d if x is not None]
        return {
            "DockQ success %": 100 * sum(x >= 0.23 for x in d) / len(d) if d else None,
            "iRMSD": avg("irmsd"), "LRMSD": avg("lrmsd"), "lDDT": avg("lddt"),
        }
    if cat == "interface_protein_ligand":
        pairs = [(fnum(r.get("rmsd")), fnum(r.get("lddt-pli"))) for r in rows]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        return {
            "RMSD<2 & lDDT-PLI>0.8 %":
                100 * sum(a < 2.0 and b > 0.8 for a, b in pairs) / len(pairs) if pairs else None,
            "lDDT-LP": avg("lddt-lp"), "lDDT-PLI": avg("lddt-pli"),
        }
    return {"GDT-TS": avg("gdt-ts"), "TM-score": avg("tm-score"),
            "RMSD": avg("rmsd"), "lDDT": avg("lddt")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="<eval-dir>/<name>/raw/*.csv from foldbench_evaluate.py")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    raw_dir = args.eval_dir / args.name / "raw"
    paper = load_paper()
    lines: list[str] = []

    for cat, (_hdr, precs) in paper.items():
        ours_csv = raw_dir / f"{cat}_ost.csv"
        dockq_csv = raw_dir / f"{cat}_dockqv2.csv"
        # FoldBench reports DockQ for nucleic interfaces from DockQv2, not OST —
        # but lDDT is only in the OST csv, so merge the two rather than picking one.
        src = dockq_csv if (cat in {"interface_protein_dna", "interface_protein_rna"}
                            and dockq_csv.exists()) else ours_csv
        if not src.exists():
            continue
        with src.open() as f:
            ours_rows = list(csv.DictReader(f))
        if not ours_rows:
            continue
        if src is dockq_csv and ours_csv.exists():
            with ours_csv.open() as f:
                lddt_by = {
                    (r["pdb_id"], r.get("seed"), r.get("sample")): r.get("lddt")
                    for r in csv.DictReader(f)
                }
            for r in ours_rows:
                r.setdefault("lddt", None)
                r["lddt"] = lddt_by.get((r["pdb_id"], r.get("seed"), r.get("sample")))
        first, best = reduce_ours(ours_rows, cat)
        keys = set(first)
        if not keys:
            continue

        cols: dict[str, dict[str, float | None]] = {
            f"ours@1": category_metrics(list(first.values()), cat),
            f"ours@best": category_metrics(list(best.values()), cat),
        }
        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in precs:
            if key_of(r, cat) in keys:
                by_model[r["model"]].append(r)
        for model in sorted(by_model):
            cols[model] = category_metrics(by_model[model], cat)

        n_t = len({k[0] for k in keys})
        lines.append(f"\n{cat}  —  {len(keys)} interfaces over {n_t} targets")
        if cat in {"interface_protein_dna", "interface_protein_rna"} and src is ours_csv:
            lines.append("    note: our DockQ is OST's, not DockQv2's — FoldBench reports "
                         "DockQv2 here, so this column is not strictly comparable")
        metrics = list(cols[f"ours@1"])
        head = "  " + "metric".ljust(26) + "".join(c.rjust(12) for c in cols)
        lines.append(head)
        lines.append("  " + "-" * (len(head) - 2))
        for m in metrics:
            row = "  " + m.ljust(26)
            for c in cols:
                v = cols[c].get(m)
                row += ("n/a" if v is None else f"{v:.2f}").rjust(12)
            lines.append(row)
        # paper coverage warning: a method missing rows for some interfaces
        for model, rs in by_model.items():
            if len({key_of(r, cat) for r in rs}) != len(keys):
                lines.append(f"    note: {model} covers "
                             f"{len({key_of(r, cat) for r in rs})}/{len(keys)} interfaces")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
