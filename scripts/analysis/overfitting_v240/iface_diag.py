"""What does the predicted distogram do on the true interface? Per class and epoch, from the sweep npz files (valid-pair P(contact) 'p', true contact 'y', true distance 'd', per-interface masks).
For every interface: top-k predicted pairs (k = number of true contacts), where do they fall? Reports precision@k, the true-distance buckets of those pairs, the residue-index miss to the nearest true contact,
how much probability mass the model puts on the interface vs the true count, p on true contacts vs the rest."""
import glob, json, sys
import numpy as np
dirs = sys.argv[1:]
res = {}
for d in dirs:
    for f in sorted(glob.glob(f"{d}/*.npz")):
        z = np.load(f, allow_pickle=True)
        if "skipped" in z.files: continue
        meta = json.loads(str(z["ifmeta"]))
        models = sorted({k.split("|")[1] for k in z.files if k.endswith("|p")}, key=lambda s: int(s[2:]))
        for cls in {m["cls"] for m in meta}:
            ifs = [m for m in meta if m["cls"] == cls]
            off = 0
            for m in ifs:
                v = z[f"if{m['k']}|valid"]; nv = int(v.sum())
                y = z[f"{cls}|y"][off:off + nv]; dd = z[f"{cls}|d"][off:off + nv]
                for mn in models:
                    p = z[f"{cls}|{mn}|p"][off:off + nv]
                    res.setdefault((cls, mn), []).append((v, y, dd, p, f.split("/")[-1][:-4]))
                off += nv
def stats(items):
    out = dict(n=0, k=0, hit=0, b8=0, b12=0, b20=0, bfar=0, miss=[], top1_hit=0, top1_12=0, mass=0.0, pos=[], neg=[], npairs=0, base=0.0, tot_k_cap=0)
    for v, y, dd, p, t in items:
        k = int(y.sum())
        if k == 0: continue
        out["n"] += 1; out["k"] += k; out["npairs"] += len(y)
        out["mass"] += float(p.sum()); out["base"] += k / len(y) * k
        o = np.argsort(-p, kind="stable")[:k]
        out["hit"] += int(y[o].sum())
        dt = dd[o]
        out["b8"] += int((dt < 8).sum()); out["b12"] += int(((dt >= 8) & (dt < 12)).sum()); out["b20"] += int(((dt >= 12) & (dt < 20)).sum()); out["bfar"] += int((dt >= 20).sum())
        t1 = int(np.argmax(p)); out["top1_hit"] += int(y[t1]); out["top1_12"] += int(dd[t1] < 12)
        out["pos"].append(float(p[y].mean())); out["neg"].append(float(p[~y].mean()))
        # residue-index miss of each top-k pair to the nearest true contact
        ii, jj = np.nonzero(v)
        ti, tj = ii[y], jj[y]
        for a in o[: min(k, 60)]:
            out["miss"].append(int(np.min(np.maximum(np.abs(ti - ii[a]), np.abs(tj - jj[a])))))
    return out
print("rows: class | epoch | interfaces | contacts | precision@k | top-k pairs by TRUE distance: <8 / 8-12 / 12-20 / >=20 A | top-1 is a contact / within 12A | median index miss (res) | sum of p / true contacts | mean p on contacts vs non-contacts | random precision@k")
for (cls, mn), items in sorted(res.items(), key=lambda x: (x[0][0], int(x[0][1][2:]))):
    s = stats(items)
    if s["n"] == 0: continue
    tk = s["b8"] + s["b12"] + s["b20"] + s["bfar"]
    print(f"{cls:17s} {mn:6s} n={s['n']:3d} k={s['k']:5d} | P@k {s['hit']/s['k']:.3f} | <8 {s['b8']/tk:.2f} 8-12 {s['b12']/tk:.2f} 12-20 {s['b20']/tk:.2f} >=20 {s['bfar']/tk:.2f} | "
          f"top1 hit {s['top1_hit']/s['n']:.2f} <12A {s['top1_12']/s['n']:.2f} | med miss {np.median(s['miss']):.0f} | sum_p/k {s['mass']/s['k']:.1f} | p+ {np.mean(s['pos']):.3f} p- {np.mean(s['neg']):.4f} | rand {s['base']/s['k']:.4f}")
