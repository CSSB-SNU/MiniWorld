"""Summarise an A/B of sweep_pro.py runs: <tag> <variant> [<variant> ...] over logs/<tag>-<variant>-L<len>-r*.json.

Prints, per length, the median over rounds of op / core / prologue / epilogue and the core kernel's name, so a run that
routed somewhere else (or fell back) is visible rather than averaged in.
"""
import glob, json, os, statistics as st, sys

R = os.path.dirname(os.path.abspath(__file__))
tag, variants = sys.argv[1], sys.argv[2:]
CORE = ("_fwd", "triattn_m1_kernel", "triattn_m")


def core_kernel(k):
    hot = sorted(k.items(), key=lambda kv: -kv[1])
    for n, us in hot:
        if "prologue" in n or "epilogue" in n or "stage_bias" in n or "mask_words" in n or "uniform_rows" in n:
            continue
        return n, us
    return "-", 0.0


rows = {}
for v in variants:
    for f in sorted(glob.glob("%s/logs/%s-%s-L*-r*.json" % (R, tag, v))):
        d = json.load(open(f))
        L = d["length"]
        n, us = core_kernel(d["kernels"])
        rows.setdefault((L, v), []).append((d["op_us"], us, d["prologue_us"], d["epilogue_us"], d["rel_rms"], n))

med = lambda xs: st.median(xs) if xs else float("nan")
for L in sorted({L for L, _ in rows}):
    print("== L%d" % L)
    base = None
    for v in variants:
        rs = rows.get((L, v)) or []
        if not rs:
            print("   %-8s no runs" % v); continue
        op, core = med([r[0] for r in rs]), med([r[1] for r in rs])
        pro, epi = med([r[2] for r in rs]), med([r[3] for r in rs])
        rel = med([r[4] for r in rs])
        names = {r[5] for r in rs}
        base = base or (op, core)
        print("   %-8s op %8.1f (%+6.1f%%)  core %7.1f (%+6.1f%%)  pro %7.1f  epi %7.1f  rel_rms %.4e  n=%d  core=%s"
              % (v, op, 100 * (op / base[0] - 1), core, 100 * (core / base[1] - 1) if base[1] else float("nan"),
                 pro, epi, rel, len(rs), "/".join(sorted(n[:40] for n in names))))
