"""Score a FoldBench prediction directory with the upstream FoldBench evaluator.

Wraps ``upstream/evaluate.py`` + ``upstream/task_score_summary.py`` (BEAM-Labs
FoldBench, Nat. Commun. 2025) so the numbers we report are produced by the
benchmark's own code, not a re-implementation.

Three things this adds on top of upstream:

* **prediction_reference.csv** — upstream expects one row per (pdb_id, seed,
  sample) with a path. We build it from the runner's output layout,
  ``<pred_dir>/<target>/<target>_seed{s}_sample{k}_pred.cif`` (the older 1x25
  layout ``<target>_s{k}_pred.cif`` is also accepted and mapped to seed 0).
* **an ``ost`` on PATH** — ``evaluation/eval_by_ost.py`` shells out to a bare
  ``ost``; this cluster has none, so ``upstream/bin/ost`` runs the upstream
  OpenStructure container through singularity. Pull it once with::

      singularity pull ost.sif docker://registry.scicore.unibas.ch/schwede/openstructure:latest

* **an interpreter with the upstream deps** — DockQv2 needs biopython and parallelbar,
  which the pixi env does not carry. ``upstream/.venv`` is a --system-site-packages venv
  on the pixi python with those two plus networkx added; override with ``--python``.

* **a filtered target list** — the task CSVs cover all 1,522 FoldBench assemblies. We
  predict a subset, and upstream left-joins predictions onto the full list, so every
  unpredicted target becomes a NaN row that crashes ``os.path.exists`` inside the OST and
  DockQ workers. The driver writes task CSVs restricted to the predicted ids and points
  upstream at those. Numbers from a subset are a subset benchmark, not FoldBench totals.
* **both aggregations** — upstream's default ``--metric_type rank`` takes the
  sample with the highest ``ranking_score``. We have no confidence head, so
  ranking_score is a constant and "rank" degenerates to "whichever row pandas
  saw first", i.e. a random sample. We therefore emit BOTH ``rank`` (read it as
  random-sample) and ``best`` (oracle best-of-N, an upper bound), and a
  per-sample mean in ``sample_stats.csv``. Never quote the ``best`` column
  against a published top-1 number.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

FB = Path("/home/psk6950/data/foldbench")
UPSTREAM = FB / "upstream"

ALL_TARGETS = [
    "interface_protein_ligand", "interface_protein_protein",
    "interface_antibody_antigen", "interface_protein_peptide",
    "interface_protein_dna", "interface_protein_rna",
    "monomer_protein", "monomer_dna", "monomer_rna",
]

# <target>_seed{s}_sample{k}_pred.cif  (5x5 protocol) | <target>_s{k}_pred.cif (legacy 1xN)
RE_SEEDED = re.compile(r"^(?P<tid>.+)_seed(?P<seed>\d+)_sample(?P<sample>\d+)_pred$")
RE_FLAT = re.compile(r"^(?P<tid>.+)_s(?P<sample>\d+)_pred$")


def build_reference(pred_dir: Path, out_csv: Path, require: int = 0) -> tuple[set[str], int]:
    rows = []
    targets = set()
    if require:
        # The prediction job may still be writing into this directory. Scoring a
        # half-finished target would silently average over fewer samples than every
        # other target, so drop anything short of the full set.
        keep = {d.name for d in pred_dir.iterdir()
                if d.is_dir() and len(list(d.glob("*_pred.cif"))) >= require}
        skipped = sum(1 for d in pred_dir.iterdir()
                      if d.is_dir() and d.name not in keep and any(d.glob("*_pred.cif")))
        if skipped:
            print(f"  skipping {skipped} target(s) with fewer than {require} predictions")
    else:
        keep = None
    for cif in sorted(pred_dir.glob("*/*_pred.cif")):
        if keep is not None and cif.parent.name not in keep:
            continue
        stem = cif.stem
        m = RE_SEEDED.match(stem)
        if m is not None:
            tid, seed, sample = m["tid"], int(m["seed"]), int(m["sample"])
        else:
            m = RE_FLAT.match(stem)
            if m is None:
                continue
            tid, seed, sample = m["tid"], 0, int(m["sample"])
        targets.add(tid)
        # ranking_score is a constant: this checkpoint has no confidence head.
        rows.append((tid, seed, sample, 1, str(cif.resolve())))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("pdb_id,seed,sample,ranking_score,prediction_path\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return targets, len(rows)


def write_filtered_targets(pdb_ids: set[str], targets: list[str], out_dir: Path) -> list[str]:
    """Copy each task CSV keeping only rows whose pdb_id we predicted."""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for t in targets:
        src = FB / "targets" / f"{t}.csv"
        if not src.exists():
            continue
        with src.open() as f:
            rows = list(csv.DictReader(f))
        sub = [r for r in rows if r["pdb_id"] in pdb_ids]
        if not sub:
            continue
        with (out_dir / f"{t}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(sub)
        kept.append(t)
        print(f"  {t}: {len(sub)}/{len(rows)} rows over "
              f"{len({r['pdb_id'] for r in sub})} predicted targets")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="Evaluation root; results land in <eval-dir>/<name>/.")
    ap.add_argument("--python", type=Path, default=UPSTREAM / ".venv" / "bin" / "python",
                    help="Interpreter for the upstream scripts (needs biopython/parallelbar; "
                         "see module docstring).")
    ap.add_argument("--name", required=True, help="Algorithm name (a column in the summary).")
    ap.add_argument("--targets", nargs="+", default=ALL_TARGETS)
    ap.add_argument("--ground-truth", type=Path, default=FB / "ground_truth")
    ap.add_argument("--require-samples", type=int, default=0,
                    help="Only score targets that already have at least this many "
                         "predictions. Use it when the prediction job is still running.")
    ap.add_argument("--skip-ost", action="store_true",
                    help="Reuse the per-prediction OST json/csv already in <eval-dir> and only "
                         "re-aggregate.")
    args = ap.parse_args()
    # The upstream scripts run with cwd=upstream/, so every path handed to them must be
    # absolute or it silently resolves against the wrong directory.
    args.pred_dir = args.pred_dir.resolve()
    args.eval_dir = args.eval_dir.resolve()
    args.ground_truth = args.ground_truth.resolve()

    ost_shim = UPSTREAM / "bin"
    if not (UPSTREAM / "images" / "ost.sif").exists():
        sys.exit(f"no OpenStructure image at {UPSTREAM / 'images' / 'ost.sif'} — see module docstring")

    eval_sub = args.eval_dir / args.name
    ref_csv = eval_sub / "prediction_reference.csv"
    pdb_ids, n_rows = build_reference(args.pred_dir, ref_csv, args.require_samples)
    print(f"prediction_reference.csv: {len(pdb_ids)} targets, {n_rows} predictions -> {ref_csv}")
    if n_rows == 0:
        sys.exit("no predictions found")

    targets_dir = eval_sub / "targets"
    print("filtered task CSVs:")
    kept = write_filtered_targets(pdb_ids, list(args.targets), targets_dir)
    if not kept:
        sys.exit("none of the predicted targets appear in the requested task CSVs")

    env = dict(os.environ)
    env["PATH"] = f"{ost_shim}:{env['PATH']}"

    if not args.skip_ost:
        cmd = [
            str(args.python), "evaluate.py",
            "--targets_dir", str(targets_dir),
            "--evaluation_dir", str(args.eval_dir),
            "--algorithm_name", args.name,
            "--ground_truth_dir", str(args.ground_truth),
            "--targets", *args.targets,
        ]
        print("+", " ".join(cmd))
        subprocess.run(cmd, cwd=UPSTREAM, env=env, check=True)

    # A pass that produced no rows still leaves a headerless csv behind (DockQv2 does
    # this when every target failed to parse), and task_score_summary reads it
    # unconditionally -> EmptyDataError. Drop those so the category falls back to its
    # OST-only metrics instead of killing the whole summary.
    for raw in sorted((eval_sub / "raw").glob("*.csv")):
        if "," not in raw.read_text(errors="ignore").partition("\n")[0]:
            print(f"  dropping empty {raw.name}")
            raw.unlink()

    # Only summarise the target types that actually produced a raw csv.
    have = [t for t in kept if (eval_sub / "raw" / f"{t}_ost.csv").exists()]
    if not have:
        sys.exit("no raw OST results were produced")
    for metric_type in ("rank", "best"):
        out = args.eval_dir / f"summary_{args.name}_{metric_type}.csv"
        cmd = [
            str(args.python), "task_score_summary.py",
            "--evaluation_dir", str(args.eval_dir),
            "--target_dir", str(targets_dir),
            "--output_path", str(out),
            "--algorithm_names", args.name,
            "--targets", *have,
            "--metric_type", metric_type,
        ]
        print("+", " ".join(cmd))
        subprocess.run(cmd, cwd=UPSTREAM, env=env, check=True)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
