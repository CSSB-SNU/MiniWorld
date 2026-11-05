"""Why is 8fbw's AP low although the map looks roughly right? ep100 / ep400, blocks A x C and B x C. Saves figures + prints numbers."""
import csv, json, sys, warnings
from pathlib import Path
import numpy as np, torch, yaml
warnings.filterwarnings("ignore")
REPO = Path("/home/psk6950/MiniWorld"); FB = Path("/home/psk6950/data/foldbench_v240eval"); OUT = Path("/home/psk6950/MiniWorld-scratch/fb-v240/why_low"); OUT.mkdir(exist_ok=True)
T = sys.argv[2]; RUN = sys.argv[1]
CK = {"ep100": f"{RUN}/checkpoints/epoch=0100.pt", "ep400": f"{RUN}/checkpoints/epoch=0400.pt"}
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src"), str(REPO / "libs/team-gm/src")]
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from hydra import compose, initialize_config_dir
from run_miniworld_diffusion_inference import _pad_inference_batch
from run_miniworld_distogram_train import Config
from miniworld.data.inference import InferenceSpec, build_inference_batch
from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel
from miniworld.training.precision import apply_precision, model_autocast
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

pairs = [(r["antibody_chain"], r["antigen_chain"]) for r in csv.DictReader(open(FB / "targets/interface_antibody_antigen.csv")) if r["pdb_id"] == T]
chain_meta = {(r["target"], r["chain_id"]): (r["entity_type"], int(r["n_res_used"] or 0), int(r["trim_offset"] or 0)) for r in csv.DictReader(open(FB / "meta/chain_seqid_map.tsv"), delimiter="\t")}
with initialize_config_dir(str(REPO / "configs/miniworld"), version_base=None):
    cfg = Config.model_validate(compose(config_name="phase1a_distogram_medium_v240_local"))
nb = cfg.model.shared.n_distogram_bins; dev = torch.device("cuda")
cen = 2.25 + 0.25 * (np.arange(nb) - 0.5); cen[0], cen[-1] = 2.0, 26.0; cen_t = torch.from_numpy(cen).float().to(dev)
def gt_reps(p):
    d = MMCIF2Dict(str(p)); asym, seq, name, comp = d["_atom_site.label_asym_id"], d["_atom_site.label_seq_id"], d["_atom_site.label_atom_id"], d["_atom_site.label_comp_id"]
    x = [np.asarray(d[f"_atom_site.Cartn_{c}"], dtype=np.float64) for c in "xyz"]
    model = d.get("_atom_site.pdbx_PDB_model_num", ["1"] * len(asym)); alt = d.get("_atom_site.label_alt_id", ["."] * len(asym)); out = {}
    for i in range(len(asym)):
        if model[i] not in ("1", "?", ".") or alt[i] not in (".", "?", "A"): continue
        out.setdefault((asym[i], seq[i], name[i]), (comp[i], x[0][i], x[1][i], x[2][i]))
    return out
def rep_xyz(table, chain, seq):
    g = table.get((chain, str(seq), "CA"))
    if g is None: return None
    v = table.get((chain, str(seq), "CA" if g[0] == "GLY" else "CB")); return None if v is None else v[1:]
def ap(p, y):
    o = np.argsort(-p, kind="stable"); ys = y[o]; tp = np.cumsum(ys); return float((tp / np.arange(1, len(ys) + 1) * ys).sum() / max(ys.sum(), 1))
spec = InferenceSpec.from_yaml(FB / "inputs" / T / "data.yaml")
if not spec.no_pairing_msa: spec = spec.model_copy(update={"no_pairing_msa": True})
ydoc = yaml.safe_load(open(FB / "inputs" / T / "data.yaml")); name2idx = {v: int(k) for k, v in ydoc["chain_names"].items()}
batch = build_inference_batch(spec, max_msa_depth=2048, missing_policy="query", seed=0, msa_subsample=True)
L = int(batch.token_length); asym = batch.scheme.token_asym_id[0].cpu().numpy(); table = gt_reps(FB / "ground_truth" / f"{T}.cif")
gt = np.full((L, 3), np.nan); rng_of = {}
for c, ci in name2idx.items():
    idx = np.nonzero(asym == ci)[0]; rng_of[c] = idx
    et, n_used, off = chain_meta.get((T, c), (None, 0, 0))
    if et is None or len(idx) != n_used or "peptide" not in et: print("chain", c, "not scored", et, len(idx), n_used); continue
    for k in range(len(idx)):
        v = rep_xyz(table, c, k + 1 + off)
        if v is not None: gt[idx[k]] = v
D = np.linalg.norm(gt[:, None, :] - gt[None, :, :], axis=-1)
print("TARGET", T, "L", L, "chains", {c: (int(v[0]), int(v[-1]) + 1) for c, v in rng_of.items()}, "pairs", pairs, flush=True)
padded = _pad_inference_batch(batch, 8).to(device=dev)
res = {}
for name, path in CK.items():
    st = torch.load(path, map_location="cpu", weights_only=False)
    m = apply_precision(MiniSWAModel(cfg.model), "bf16-mixed"); m.load_state_dict(st["ema_state_dict"], strict=False); m = m.to(dev).eval(); del st
    m._forced_n_recycle = 4
    with torch.no_grad(), model_autocast(m):
        lg = m(padded.msa, padded.reference, padded.scheme, padded.sequence, padded.structure, padded.template)[0, :L, :L].float()
    pr = torch.softmax(lg, -1); cum = torch.cumsum(pr, -1).cpu().numpy()
    ed = (pr * cen_t).sum(-1).cpu().numpy()
    P = {thr: cum[..., int(round((thr - 2.25) / 0.25))] for thr in (8, 10, 12, 16)}
    res[name] = (ed, P, pr.max(-1).values.cpu().numpy())
    del m; torch.cuda.empty_cache()
out = {}
for name, (ed, P, pm) in res.items():
    allY = {thr: [] for thr in P}; allP = {thr: [] for thr in P}
    for (a, b) in pairs:
        ia, ib = rng_of[a], rng_of[b]; blk = D[np.ix_(ia, ib)]; ok = ~np.isnan(blk)
        for thr in P: allY[thr].append((blk < thr)[ok]); allP[thr].append(P[thr][np.ix_(ia, ib)][ok])
    r = {}
    for thr in P:
        y, p = np.concatenate(allY[thr]), np.concatenate(allP[thr]); r[f"AP@{thr}A"] = ap(p, y); r[f"n_pos@{thr}A"] = int(y.sum()); r["n_pairs"] = int(len(y))
    # block detail at 8 A
    det = []
    for (a, b) in pairs:
        ia, ib = rng_of[a], rng_of[b]; blk = D[np.ix_(ia, ib)]; ok = ~np.isnan(blk); y = (blk < 8) & ok; p8 = P[8][np.ix_(ia, ib)]
        k = int(y.sum()); pp = np.where(ok, p8, -1); o = np.argsort(-pp.ravel(), kind="stable"); topk = o[:max(k, 1)]
        ti, tj = np.nonzero(y); ci, cj = np.unravel_index(topk, blk.shape)
        cheb = [int(np.min(np.maximum(np.abs(ti - i), np.abs(tj - j)))) if k else -1 for i, j in zip(ci, cj)]
        dtop = blk[ci, cj]
        rank = np.empty(pp.size); rank[o] = np.arange(pp.size); rk = (rank.reshape(blk.shape)[y] / ok.sum()) if k else np.array([])
        det.append(dict(block=f"{a}x{b}", shape=list(blk.shape), valid=int(ok.sum()), k=k, ap8=ap(p8[ok], y[ok]) if k else None,
                        p_at_k=float(y.ravel()[topk].mean()) if k else None,
                        top_within={w: float(np.mean(np.array(cheb) <= w)) for w in (0, 1, 2, 3, 5, 10)},
                        topk_true_dist={"<8": float(np.mean(dtop < 8)), "8-10": float(np.mean((dtop >= 8) & (dtop < 10))), "10-12": float(np.mean((dtop >= 10) & (dtop < 12))), "12-16": float(np.mean((dtop >= 12) & (dtop < 16))), ">=16": float(np.mean(dtop >= 16))},
                        true_contact_rank_pct={"median": float(np.median(rk) * 100) if k else None, "in_top1%": float(np.mean(rk < .01)) if k else None, "in_top5%": float(np.mean(rk < .05)) if k else None, "in_top10%": float(np.mean(rk < .10)) if k else None},
                        ed_at_true_contacts={"mean": float(ed[np.ix_(ia, ib)][y].mean()) if k else None, "frac_pred_lt8": float((ed[np.ix_(ia, ib)][y] < 8).mean()) if k else None},
                        sum_p8=float(p8[ok].sum())))
    r["blocks"] = det; out[name] = r
    print("=====", name, json.dumps(r, ensure_ascii=False)[:100000], flush=True)
json.dump(out, open(OUT / "stats.json", "w"))
# figure: each block: true D | predicted E[d] | predicted P(d<8) with true contacts circled
for (a, b) in pairs:
    ia, ib = rng_of[a], rng_of[b]
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 2 * 4.2 * len(ia) / max(len(ib), 1) ** 0.0 if False else 8.6), dpi=90)
    blk = D[np.ix_(ia, ib)]; y = blk < 8
    for r_i, name in enumerate(("ep100", "ep400")):
        ed, P, _ = res[name]
        ax[r_i, 0].imshow(np.clip(blk, 2, 26), cmap="viridis_r", vmin=2, vmax=26, aspect="auto"); ax[r_i, 0].set_title(f"정답 거리 ({a}×{b})" if r_i == 0 else "정답 거리", fontsize=9)
        ax[r_i, 1].imshow(ed[np.ix_(ia, ib)], cmap="viridis_r", vmin=2, vmax=26, aspect="auto"); ax[r_i, 1].set_title(f"{name} 예측 E[d]", fontsize=9)
        ax[r_i, 2].imshow(np.sqrt(P[8][np.ix_(ia, ib)]), cmap="magma", vmin=0, vmax=1, aspect="auto"); ax[r_i, 2].set_title(f"{name} P(d<8Å) √ · 흰 점 = 정답 접촉", fontsize=9)
        ti, tj = np.nonzero(y); ax[r_i, 2].scatter(tj, ti, s=14, facecolors="none", edgecolors="white", linewidths=0.7)
        for a_ in ax[r_i]: a_.set_xlabel(f"{b} 잔기", fontsize=8); a_.set_ylabel(f"{a} 잔기", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"{T}_{a}x{b}.png"); plt.close(fig)
print("WHYLOW_DONE", flush=True)
