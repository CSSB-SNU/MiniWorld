import glob, json, sys
import numpy as np
d = sys.argv[1]; classes = sys.argv[2].split(",")
def metrics(p, y):
    n_pos = int(y.sum()); n = len(y); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0: return None
    order = np.argsort(-p, kind="stable"); ys = y[order]
    tp = np.cumsum(ys); prec = tp / np.arange(1, n + 1); rec = tp / n_pos
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    ranks = np.empty(n); ranks[np.argsort(p, kind="stable")] = np.arange(1, n + 1)
    return dict(ap=float((prec * ys).sum() / n_pos), f1=float(f1.max()), auroc=float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)))
acc = {}
skipped = 0; ntg = {c: 0 for c in classes}; niface = {c: 0 for c in classes}; nlayout = {c: 0 for c in classes}
for f in sorted(glob.glob(f"{d}/*.npz")):
    try:
        z = np.load(f, allow_pickle=True); z.files
    except Exception:
        continue
    if "skipped" in z.files: skipped += 1; continue
    info = json.loads(str(z["info"]))
    for c in classes:
        rows = [r for r in info if r[0] == c]
        if rows: ntg[c] += 1
        niface[c] += sum(1 for r in rows if isinstance(r[3], int)); nlayout[c] += sum(1 for r in rows if not isinstance(r[3], int))
        if f"{c}|y" not in z.files: continue
        a = acc.setdefault(c, {"y": [], "d": [], "m": {}})
        a["y"].append(z[f"{c}|y"]); a["d"].append(z[f"{c}|d"])
        for k in z.files:
            if k.startswith(f"{c}|") and k.endswith("|p"):
                a["m"].setdefault(k.split("|")[1], {"p": [], "e": []}); a["m"][k.split("|")[1]]["p"].append(z[k]); a["m"][k.split("|")[1]]["e"].append(z[k[:-1] + "e"])
print("skipped targets (oom/cap/build):", skipped)
res = {}
for c in classes:
    if c not in acc: continue
    y = np.concatenate(acc[c]["y"]); dt = np.concatenate(acc[c]["d"])
    print(f"\n== {c}: targets {ntg[c]}, interfaces scored {niface[c]} (not scored: {nlayout[c]}), pairs {len(y):,}, contacts {int(y.sum()):,}")
    print("  epoch |   AP  | bestF1 | AUROC |  BCE   | MAE(Å,d<20)")
    for name in sorted(acc[c]["m"], key=lambda s: int(s[2:])):
        p = np.concatenate(acc[c]["m"][name]["p"]); e = np.concatenate(acc[c]["m"][name]["e"]); m = metrics(p, y)
        pe = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6); bce = float(-(y * np.log(pe) + (~y) * np.log(1 - pe)).mean())
        near = dt < 20; mae = float(np.abs(e[near] - dt[near]).mean())
        res.setdefault(c, {})[name] = dict(m, bce=bce, mae=mae)
        print(f"  {name:>5} | {m['ap']:.3f} | {m['f1']:.3f}  | {m['auroc']:.4f} | {bce:.5f} | {mae:.2f}")
json.dump(res, open(f"{d}/../merged_{'_'.join(classes)}.json", "w"), indent=1)
