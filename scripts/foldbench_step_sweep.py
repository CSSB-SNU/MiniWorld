"""Score a --timesteps sweep and report quality against denoising-step count.

Reads ``<sweep-dir>/steps<N>/<target>/*_pred.cif`` (what
``run_miniworld_diffusion_inference.py foldbench --timesteps 20,50,100,200`` writes),
scores every prediction against the FoldBench ground truth with OpenStructure, and
prints one row per step count so the smallest N that has not lost quality is readable
off the table.

Why it matters: phase 3 trains the confidence head on structures the frozen diffusion
module rolls out inline at every training step (``predict_timesteps`` in
phase3a_confidence.yaml), so the step count multiplies straight into phase-3 cost.

RUN AS A BATCH JOB. Like foldbench_evaluate.py this fans out OpenStructure containers;
on the login node it locks the machine out for everyone.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FB = Path("/home/psk6950/data/foldbench")
OST = FB / "upstream" / "bin" / "ost"
CMD = ("{ost} compare-structures -m {model} -r {ref} -o {out} --fault-tolerant "
       "--min-pep-length 4 --min-nuc-length 4 --lddt --rigid-scores --tm-score --dockq")


def score_one(args: tuple[Path, Path, Path]) -> None:
    model, ref, out = args
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        CMD.format(ost=OST, model=model, ref=ref, out=out),
        shell=True, check=False, capture_output=True, executable="/bin/bash",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None, help="Where the OST json goes.")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    cache = args.cache or (args.sweep_dir / "_ost")
    step_dirs = sorted(args.sweep_dir.glob("steps*"), key=lambda p: int(p.name[5:]))
    if not step_dirs:
        sys.exit(f"no steps*/ under {args.sweep_dir}")

    jobs, index = [], []
    for sd in step_dirs:
        n = int(sd.name[5:])
        for cif in sorted(sd.glob("*/*_pred.cif")):
            tid = cif.parent.name
            out = cache / sd.name / f"{cif.stem}.json"
            jobs.append((cif, FB / "ground_truth" / f"{tid}.cif", out))
            index.append((n, tid, out))
    print(f"{len(jobs)} predictions over {len(step_dirs)} step counts -> scoring")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(score_one, jobs))

    # metric -> {step: [values]}, and per (step, target) for the paired view
    agg: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_target: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for n, tid, out in index:
        if not out.exists():
            continue
        try:
            d = json.loads(out.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("status") != "SUCCESS":
            continue
        vals = {"lDDT": d.get("lddt"), "TM": d.get("tm_score"),
                "GDT-TS": d.get("oligo_gdtts"), "RMSD": d.get("rmsd"),
                "clashes": float(len(d.get("model_clashes", [])))}
        dq = d.get("dockq") or []
        if dq:
            vals["DockQ(max iface)"] = max(dq)
        for k, v in vals.items():
            if v is not None:
                agg[n][k].append(float(v))
                per_target[(n, tid)][k].append(float(v))

    metrics = ["lDDT", "TM", "GDT-TS", "RMSD", "DockQ(max iface)", "clashes"]
    print(f"\n{'steps':>6}" + "".join(m.rjust(18) for m in metrics) + f"{'n':>7}")
    print("-" * (6 + 18 * len(metrics) + 7))
    for n in sorted(agg):
        row = f"{n:>6}"
        for m in metrics:
            v = agg[n].get(m)
            row += (f"{statistics.mean(v):.3f}" if v else "n/a").rjust(18)
        row += f"{len(agg[n].get('lDDT', [])):>7}"
        print(row)

    # paired per-target lDDT, so a step count is not credited for an easy target mix
    print("\nper-target mean lDDT (paired across step counts)")
    targets = sorted({t for _, t in per_target})
    hdr = "  " + "target".ljust(22) + "".join(str(n).rjust(9) for n in sorted(agg))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for t in targets:
        row = "  " + t.ljust(22)
        for n in sorted(agg):
            v = per_target.get((n, t), {}).get("lDDT")
            row += (f"{statistics.mean(v):.3f}" if v else "n/a").rjust(9)
        print(row)


if __name__ == "__main__":
    main()
