"""Record every FoldBench target's real token/atom length and the bucket it lands in.

Inference pads to a shape ladder, so what actually drives memory is the BUCKET, not the
target. This writes one row per target and a bucket histogram, which is what you need to
know before picking a sampling batch size: the largest bucket decides whether aug=N fits.

CPU only, and parallel -- run it as a batch job, not on the login node.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FB = Path("/home/psk6950/data/foldbench")


def one(tid: str) -> tuple[str, int, int, int, int] | None:
    from miniworld.data.inference.build import build_inference_batch
    from miniworld.data.inference.spec import InferenceSpec
    p = FB / "inputs" / tid / "data.yaml"
    if not p.exists():
        return None
    try:
        spec = InferenceSpec.from_yaml(p)
        spec = spec.model_copy(update={"no_pairing_msa": True})
        b = build_inference_batch(spec, max_msa_depth=2048, missing_policy="query",
                                  seed=0, msa_subsample=True)
        return (tid, int(b.token_length), int(b.atom_length),
                int(b.msa_depth), int(b.template_number))
    except Exception:  # noqa: BLE001 - one bad target must not kill the scan
        return (tid, -1, -1, -1, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("submits/phase2b/foldbench_full_index.txt"))
    ap.add_argument("--out", type=Path, default=Path("submits/phase2b/foldbench_shapes.csv"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "32")))
    args = ap.parse_args()

    from run_miniworld_diffusion_inference import (  # noqa: PLC0415
        _ATOM_BUCKETS, _TOKEN_BUCKETS, _bucket,
    )

    ids = [ln.strip() for ln in args.index.read_text().splitlines() if ln.strip()]
    print(f"scanning {len(ids)} targets with {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(one, ids, chunksize=4) if r is not None]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "tokens", "atoms", "msa_depth", "n_template",
                    "tok_bucket", "atom_bucket"])
        for tid, t, a, m, nt in rows:
            tb = _bucket(t, _TOKEN_BUCKETS) if t > 0 else -1
            ab = _bucket(a, _ATOM_BUCKETS) if a > 0 else -1
            w.writerow([tid, t, a, m, nt, tb, ab])
    bad = [r for r in rows if r[1] < 0]
    print(f"wrote {args.out}  ({len(rows)} rows, {len(bad)} failed to build)")
    for r in bad[:10]:
        print("   FAILED", r[0])

    hist = Counter((_bucket(t, _TOKEN_BUCKETS), _bucket(a, _ATOM_BUCKETS))
                   for _, t, a, _m, _nt in rows if t > 0)
    print(f"\n{len(hist)} distinct (token, atom) buckets:")
    print("  %8s %8s %8s   %s" % ("tok_bkt", "atom_bkt", "targets", "largest real target"))
    big = {}
    for _tid, t, a, _m, _nt in rows:
        if t < 0:
            continue
        k = (_bucket(t, _TOKEN_BUCKETS), _bucket(a, _ATOM_BUCKETS))
        if k not in big or t > big[k][1]:
            big[k] = (_tid, t, a)
    for k in sorted(hist, key=lambda k: (k[0], k[1])):
        tid, t, a = big[k]
        print("  %8d %8d %8d   %s (tok=%d atom=%d)" % (k[0], k[1], hist[k], tid, t, a))


if __name__ == "__main__":
    main()
