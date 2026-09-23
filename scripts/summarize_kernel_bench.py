"""Table for runs/bench_trimul_triattn: ms per module call, arms side by side, speedups."""
from __future__ import annotations

import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/bench_trimul_triattn")
arms = ["cueq", "engine_main", "engine_main_compiled"]
rows: dict[tuple, dict] = {}
for f in sorted(d.glob("*.json")):
    arm = next((a for a in sorted(arms, key=len, reverse=True) if f.stem.endswith("_" + a)), None)
    if arm is None:
        continue
    for r in json.loads(f.read_text()):
        rows.setdefault((r["family"], r["length"], r["training"]), {})[arm] = r["ms"] if r.get("ms") is not None else float("nan")
fam_name = {"outgoing": "TriMul outgoing", "bidir": "TriMul bidirectional",
            "triattn": "TriAttn bias-only", "triattn_qk": "TriAttn q/k"}
for training in (False, True):
    print(f"\n## {'TRAINING (fwd+bwd, dropout 0.25)' if training else 'INFERENCE (fwd)'}  -- ms per call, BF16, width 128")
    print(f"{'module':22s}{'L':>5s} | {'cuEquiv':>9s} | {'engine':>9s} {'engine+cmp':>11s} | {'cuEq/engine':>12s}")
    for (fam, L, tr), v in sorted(rows.items(), key=lambda kv: (list(fam_name).index(kv[0][0]), kv[0][1], kv[0][2])):
        if tr != training:
            continue
        g = lambda k: v.get(k, float("nan"))  # noqa: E731
        r1 = g("cueq") / g("engine_main") if g("engine_main") else float("nan")
        print(f"{fam_name[fam]:22s}{L:>5d} | {g('cueq'):9.3f} | {g('engine_main'):9.3f} {g('engine_main_compiled'):11.3f} | {r1:11.2f}x")
