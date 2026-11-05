"""True vs predicted distogram (L x L) for a few FoldBench targets: ab-ag (best / median / worst by AP at ep100) and two protein-DNA, ep100 and ep400 EMA.
Predicted = expected distance E[d] over the 64 bins and P(d < 8 A); true = representative-atom distance of the answer CIF (grey where unresolved). Writes PNGs + meta.json."""
import csv, glob, json, sys, time, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, yaml
warnings.filterwarnings("ignore")
REPO = Path("/home/psk6950/MiniWorld"); FB = Path("/home/psk6950/data/foldbench_v240eval")
SWEEP = ["/home/psk6950/MiniWorld-scratch/fb-v240/sweep", "/home/psk6950/MiniWorld-scratch/fb-v240/sweep_ep400"]
OUT = Path("/home/psk6950/MiniWorld-scratch/fb-v240/viz_dist50"); OUT.mkdir(exist_ok=True)
RUN = sys.argv[1]
CK = {"ep100": f"{RUN}/checkpoints/epoch=0100.pt", "ep400": f"{RUN}/checkpoints/epoch=0400.pt"}
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src"), str(REPO / "libs/team-gm/src")]
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from hydra import compose, initialize_config_dir
from run_miniworld_diffusion_inference import _pad_inference_batch
from run_miniworld_distogram_train import Config
from miniworld.data.inference import InferenceSpec, build_inference_batch
from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel
from miniworld.training.precision import apply_precision, model_autocast
import matplotlib; matplotlib.use("Agg")
import matplotlib.image as mpimg, matplotlib.pyplot as plt

# ---- choose targets from the finished sweep (ep100 AP per target)
def ap(p, y):
    o = np.argsort(-p, kind="stable"); ys = y[o]; tp = np.cumsum(ys); return float((tp / np.arange(1, len(ys) + 1) * ys).sum() / max(ys.sum(), 1))
score = {"antibody_antigen": {}, "protein_dna": {}, "protein_rna": {}}
for f in glob.glob(SWEEP[0] + "/*.npz"):
    z = np.load(f, allow_pickle=True)
    if "skipped" in z.files: continue
    t = Path(f).stem; tok = int(z["tokens"])
    for cls in score:
        if f"{cls}|y" in z.files and f"{cls}|ep100|p" in z.files and z[f"{cls}|y"].sum() >= (5 if cls == "protein_rna" else 8) and tok <= 900:
            score[cls][t] = (ap(z[f"{cls}|ep100|p"], z[f"{cls}|y"]), tok)
def spread(lst, n):
    idx = sorted({int(round(i)) for i in np.linspace(0, len(lst) - 1, min(n, len(lst)))})
    return [(lst[i][0], i + 1, len(lst)) for i in idx]
ab = sorted(score["antibody_antigen"].items(), key=lambda x: x[1][0], reverse=True)
dn = sorted(score["protein_dna"].items(), key=lambda x: -x[1][0])
rn = sorted(score["protein_rna"].items(), key=lambda x: -x[1][0])
pick = [(t, "antibody_antigen", f"AP 순위 {r}/{n}") for t, r, n in spread(ab, 38)] + [(t, "protein_dna", f"AP 순위 {r}/{n}") for t, r, n in spread(dn, 7)] + [(t, "protein_rna", f"AP 순위 {r}/{n}") for t, r, n in spread(rn, 5)]
print("PICK", [(t, c, w, round(score[c][t][0], 3), score[c][t][1]) for t, c, w in pick], flush=True)

# ---- model + gt helpers (as fb_epoch_sweep.py)
CLASSES = {"protein_dna": "interface_protein_dna", "antibody_antigen": "interface_antibody_antigen", "protein_rna": "interface_protein_rna"}
ifaces = defaultdict(lambda: defaultdict(set))
for cls, f in CLASSES.items():
    for r in csv.DictReader(open(FB / "targets" / f"{f}.csv")):
        c1, c2 = (r["antibody_chain"], r["antigen_chain"]) if cls == "antibody_antigen" else (r["interface_chain_id_1"], r["interface_chain_id_2"])
        ifaces[r["pdb_id"]][cls].add((c1, c2))
chain_meta = {}
for r in csv.DictReader(open(FB / "meta/chain_seqid_map.tsv"), delimiter="\t"):
    chain_meta[(r["target"], r["chain_id"])] = (r["entity_type"], int(r["n_res_used"] or 0), int(r["trim_offset"] or 0))
with initialize_config_dir(str(REPO / "configs/miniworld"), version_base=None):
    cfg = Config.model_validate(compose(config_name="phase1a_distogram_medium_v240_local"))
nb = cfg.model.shared.n_distogram_bins; dev = torch.device("cuda")
models = {}
for name, path in CK.items():
    st = torch.load(path, map_location="cpu", weights_only=False)
    m = apply_precision(MiniSWAModel(cfg.model), "bf16-mixed"); m.load_state_dict(st["ema_state_dict"], strict=False); models[name] = m.to(dev).eval(); del st
cen = 2.25 + 0.25 * (np.arange(nb) - 0.5); cen[0], cen[-1] = 2.0, 26.0; cen_t = torch.from_numpy(cen).float().to(dev)
PURINE = {"A", "G", "DA", "DG", "I", "DI"}; PYRIM = {"C", "U", "T", "DC", "DT", "DU"}
def gt_reps(p):
    d = MMCIF2Dict(str(p)); asym, seq, name, comp = d["_atom_site.label_asym_id"], d["_atom_site.label_seq_id"], d["_atom_site.label_atom_id"], d["_atom_site.label_comp_id"]
    x = [np.asarray(d[f"_atom_site.Cartn_{c}"], dtype=np.float64) for c in "xyz"]
    model = d.get("_atom_site.pdbx_PDB_model_num", ["1"] * len(asym)); alt = d.get("_atom_site.label_alt_id", ["."] * len(asym)); out = {}
    for i in range(len(asym)):
        if model[i] not in ("1", "?", ".") or alt[i] not in (".", "?", "A"): continue
        out.setdefault((asym[i], seq[i], name[i]), (comp[i], x[0][i], x[1][i], x[2][i]))
    return out
def rep_xyz(table, chain, seq, etype):
    s = str(seq)
    if "peptide" in etype:
        g = table.get((chain, s, "CA"))
        if g is None: return None
        v = table.get((chain, s, "CA" if g[0] == "GLY" else "CB")); return None if v is None else v[1:]
    g = table.get((chain, s, "C1'")) or table.get((chain, s, "P")) or table.get((chain, s, "C4")) or table.get((chain, s, "C2"))
    if g is None: return None
    atom = "C4" if g[0] in PURINE else ("C2" if g[0] in PYRIM else None)
    if atom is None: return None
    v = table.get((chain, s, atom)); return None if v is None else v[1:]
def pool(a, f):
    if f == 1: return a
    n = a.shape[0]; m = -(-n // f) * f
    b = np.full((m, m), np.nan, dtype=np.float32); b[:n, :n] = a
    return np.nanmean(b.reshape(m // f, f, m // f, f), axis=(1, 3))
from PIL import Image
def save_img(path, arr, cmap, vmin, vmax):
    mpimg.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)
    Image.open(path).convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
cm_d = plt.get_cmap("viridis_r").copy(); cm_d.set_bad("#9aa0a6")
cm_p = plt.get_cmap("magma").copy(); cm_p.set_bad("#9aa0a6")
meta = []
for ti, (t, cls, why) in enumerate(pick):
    spec = InferenceSpec.from_yaml(FB / "inputs" / t / "data.yaml")
    if not spec.no_pairing_msa: spec = spec.model_copy(update={"no_pairing_msa": True})
    ydoc = yaml.safe_load(open(FB / "inputs" / t / "data.yaml")); name2idx = {v: int(k) for k, v in ydoc["chain_names"].items()}
    batch = build_inference_batch(spec, max_msa_depth=2048, missing_policy="query", seed=0, msa_subsample=True)
    L = int(batch.token_length); asym = batch.scheme.token_asym_id[0].cpu().numpy()
    table = gt_reps(FB / "ground_truth" / f"{t}.cif")
    gt = np.full((L, 3), np.nan); chains = []
    for c, ci in name2idx.items():
        idx = np.nonzero(asym == ci)[0]
        if len(idx) == 0: continue
        et, n_used, off = chain_meta.get((t, c), (None, 0, 0))
        chains.append(dict(name=c, start=int(idx[0]), end=int(idx[-1]) + 1, etype=et))
        if et is None or len(idx) != n_used or not (("peptide" in et) or ("ribo" in et)): continue
        for k in range(len(idx)):
            v = rep_xyz(table, c, k + 1 + off, et)
            if v is not None: gt[idx[k]] = v
    D = np.linalg.norm(gt[:, None, :] - gt[None, :, :], axis=-1).astype(np.float32); D[np.isnan(D)] = np.nan
    padded = _pad_inference_batch(batch, 8).to(device=dev)
    f = max(1, -(-L // 480)); tag = f"{t}"
    save_img(OUT / f"{tag}_true_d.png", pool(np.clip(D, 2, 26), f), cmap=cm_d, vmin=2, vmax=26)
    save_img(OUT / f"{tag}_true_c.png", pool(np.where(np.isnan(D), np.nan, (D < 8).astype(np.float32)), f), cmap=cm_p, vmin=0, vmax=1)
    rec = dict(id=t, cls=cls, why=why, L=L, f=f, chains=chains, pairs=[list(p) for p in sorted(ifaces[t][cls])], models={})
    for mn, m in models.items():
        m._forced_n_recycle = 4
        with torch.no_grad(), model_autocast(m):
            lg = m(padded.msa, padded.reference, padded.scheme, padded.sequence, padded.structure, padded.template)[0, :L, :L].float()
        pr = torch.softmax(lg, -1); ed = (pr * cen_t).sum(-1).cpu().numpy(); pc = pr[..., :24].sum(-1).cpu().numpy()
        save_img(OUT / f"{tag}_{mn}_d.png", pool(ed.astype(np.float32), f), cmap=cm_d, vmin=2, vmax=26)
        save_img(OUT / f"{tag}_{mn}_c.png", pool(np.sqrt(pc).astype(np.float32), f), cmap=cm_p, vmin=0, vmax=1)
        # AP over every scored interface of this class (resolved pairs only)
        ys, ps = [], []
        for (a, b) in sorted(ifaces[t][cls]):
            ia, ib = np.nonzero(asym == name2idx.get(a, -1))[0], np.nonzero(asym == name2idx.get(b, -1))[0]
            if len(ia) == 0 or len(ib) == 0: continue
            blk = D[np.ix_(ia, ib)]; ok = ~np.isnan(blk)
            ys.append((blk < 8)[ok]); ps.append(pc[np.ix_(ia, ib)][ok])
        if ys:
            y, p = np.concatenate(ys), np.concatenate(ps); k = int(y.sum())
            o = np.argsort(-p, kind="stable")[:max(k, 1)]
            rec["models"][mn] = dict(ap=ap(p, y) if k else None, pk=float(y[o].mean()) if k else None, k=k, mean_ed_err=float(np.nanmean(np.abs(ed - np.where(np.isnan(D), np.nan, np.clip(D, 2, 26))))))
        else: rec["models"][mn] = dict(ap=None, pk=None, k=0, mean_ed_err=None)
    meta.append(rec); print(f"[{ti+1}/{len(pick)}] {t} {cls} L={L} f={f} {rec['models']}", flush=True)
    del padded; torch.cuda.empty_cache()
json.dump(meta, open(OUT / "meta.json", "w"))
print("VIZDIST_DONE", len(meta), flush=True)
