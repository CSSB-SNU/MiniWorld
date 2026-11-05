"""FoldBench distogram (trunk only) by interface class for a list of checkpoints: protein-RNA / protein-DNA / protein-protein / antibody-antigen.
Inter-chain token pairs of every listed interface (targets/*.csv rows), contact = representative-atom distance < 8 A (protein CB/CA, purine C4, pyrimidine C2),
pairs scored when both residues are resolved in the answer CIF. One npz per target in --out (resumable). No repo code is modified.
  python fb_epoch_sweep.py --shard 0 --nshards 2 --out DIR ep100=ckpt ep200=ckpt ..."""
import argparse, csv, sys, time, json
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, default=Path.home() / "MiniWorld")
ap.add_argument("--fb", type=Path, default=Path.home() / "data/foldbench")
ap.add_argument("--config", default="phase1b_distogram_medium_v200_b200")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=100000)
ap.add_argument("--pp-sample", type=int, default=60, help="number of protein-protein targets (random, seed --pp-seed); 0 = all")
ap.add_argument("--pp-seed", type=int, default=0)
ap.add_argument("--weights", choices=("ema", "raw"), default="ema")
ap.add_argument("--override", action="append", default=[], help="Hydra override for the model config, e.g. model.atom_swa.swa_window_size=128")
ap.add_argument("--classes", default="protein_rna,protein_dna,protein_protein,antibody_antigen")
ap.add_argument("--light", action="store_true", help="no logits blocks, no PNG")
ap.add_argument("--recycles", type=int, default=4)
ap.add_argument("--max-msa-depth", type=int, default=2048)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("ckpts", nargs="+", help="name=path (full training checkpoint, EMA weights used)")
a = ap.parse_args()
sys.path[:0] = [str(a.repo / "scripts"), str(a.repo / "src"), str(a.repo / "libs/team-gm/src")]
import yaml
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from hydra import compose, initialize_config_dir
from run_miniworld_diffusion_inference import _pad_inference_batch
from run_miniworld_distogram_train import Config
from miniworld.data.inference import InferenceSpec, build_inference_batch
from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel
from miniworld.training.precision import apply_precision, model_autocast

CB_BIN = 23
a.out.mkdir(parents=True, exist_ok=True)
# ---- interfaces per class
CLASSES = {"protein_rna": "interface_protein_rna", "protein_dna": "interface_protein_dna", "protein_protein": "interface_protein_protein", "antibody_antigen": "interface_antibody_antigen"}
ifaces = defaultdict(lambda: defaultdict(set))  # target -> class -> {(c1, c2)}
for cls, f in CLASSES.items():
    for r in csv.DictReader(open(a.fb / "targets" / f"{f}.csv")):
        c1, c2 = (r["antibody_chain"], r["antigen_chain"]) if cls == "antibody_antigen" else (r["interface_chain_id_1"], r["interface_chain_id_2"])
        ifaces[r["pdb_id"]][cls].add((c1, c2))
chain_meta = {}  # (target, chain) -> (entity_type, n_res_used, trim_offset)
for r in csv.DictReader(open(a.fb / "meta/chain_seqid_map.tsv"), delimiter="\t"):
    chain_meta[(r["target"], r["chain_id"])] = (r["entity_type"], int(r["n_res_used"] or 0), int(r["trim_offset"] or 0))
if a.pp_sample:
    pp_targets = sorted(t for t in ifaces if "protein_protein" in ifaces[t])
    keep = set(np.random.default_rng(a.pp_seed).choice(pp_targets, min(a.pp_sample, len(pp_targets)), replace=False).tolist())
    for t in pp_targets:
        if t not in keep: del ifaces[t]["protein_protein"]
    for t in [t for t in ifaces if not ifaces[t]]: del ifaces[t]
    (a.out / "pp_sample.json").write_text(json.dumps(sorted(keep)))
_keep = set(a.classes.split(","))
for t in list(ifaces):
    for cls in list(ifaces[t]):
        if cls not in _keep: del ifaces[t][cls]
    if not ifaces[t]: del ifaces[t]
order = {"protein_rna": 0, "protein_dna": 1, "antibody_antigen": 2, "protein_protein": 3}
targets = sorted(ifaces, key=lambda t: (min(order[c] for c in ifaces[t]), t))
targets = [t for i, t in enumerate(targets) if i % a.nshards == a.shard]
if a.limit: targets = targets[: a.limit]
print(f"shard {a.shard}/{a.nshards}: {len(targets)} targets", flush=True)

# ---- models
with initialize_config_dir(str(a.repo / "configs/miniworld"), version_base=None):
    cfg = Config.model_validate(compose(config_name=a.config, overrides=a.override))
nb = cfg.model.shared.n_distogram_bins
dev = torch.device("cuda")
models = {}
for spec_ in a.ckpts:
    name, path = spec_.split("=", 1)
    st = torch.load(path, map_location="cpu", weights_only=False)
    m = apply_precision(MiniSWAModel(cfg.model), "bf16-mixed")
    r = m.load_state_dict(st["ema_state_dict"] if (a.weights == "ema" and "ema_state_dict" in st) else st["model_state_dict"], strict=False)
    print(f"{name}: using {'ema_state_dict' if (a.weights == 'ema' and 'ema_state_dict' in st) else 'model_state_dict (raw)'}", flush=True)
    print(f"{name}: epoch {st.get('epoch')} missing {len(r.missing_keys)} unexpected {len(r.unexpected_keys)}", flush=True)
    models[name] = m.to(dev).eval()
cen = 2.25 + 0.25 * (np.arange(nb) - 0.5); cen[0], cen[-1] = 2.0, 26.0
cen_t = torch.from_numpy(cen).float().to(dev)

PURINE = {"A", "G", "DA", "DG", "I", "DI"}; PYRIM = {"C", "U", "T", "DC", "DT", "DU"}
def gt_reps(gt_path):
    d = MMCIF2Dict(str(gt_path))
    asym, seq, name, comp = d["_atom_site.label_asym_id"], d["_atom_site.label_seq_id"], d["_atom_site.label_atom_id"], d["_atom_site.label_comp_id"]
    x = [np.asarray(d[f"_atom_site.Cartn_{c}"], dtype=np.float64) for c in "xyz"]
    model = d.get("_atom_site.pdbx_PDB_model_num", ["1"] * len(asym)); alt = d.get("_atom_site.label_alt_id", ["."] * len(asym))
    out = {}
    for i in range(len(asym)):
        if model[i] not in ("1", "?", ".") or alt[i] not in (".", "?", "A"): continue
        out.setdefault((asym[i], seq[i], name[i]), (comp[i], x[0][i], x[1][i], x[2][i]))
    return out
def rep_xyz(table, chain, seq, etype):
    s = str(seq)
    if "peptide" in etype:
        g = table.get((chain, s, "CA"))
        if g is None: return None
        atom = "CA" if g[0] == "GLY" else "CB"
        v = table.get((chain, s, atom)); return None if v is None else v[1:]
    g = table.get((chain, s, "C1'")) or table.get((chain, s, "P")) or table.get((chain, s, "C4")) or table.get((chain, s, "C2"))
    if g is None: return None
    comp = g[0]; atom = "C4" if comp in PURINE else ("C2" if comp in PYRIM else None)
    if atom is None: return None
    v = table.get((chain, s, atom)); return None if v is None else v[1:]

t0 = time.time()
for ti, t in enumerate(targets):
    outp = a.out / f"{t}.npz"
    if outp.exists(): continue
    spec = InferenceSpec.from_yaml(a.fb / "inputs" / t / "data.yaml")
    if not spec.no_pairing_msa: spec = spec.model_copy(update={"no_pairing_msa": True})
    ydoc = yaml.safe_load(open(a.fb / "inputs" / t / "data.yaml"))
    name2idx = {v: int(k) for k, v in ydoc["chain_names"].items()}
    try:
        batch = build_inference_batch(spec, max_msa_depth=a.max_msa_depth, missing_policy="query", seed=0, msa_subsample=True)
    except Exception as e:
        print(f"[{ti+1}/{len(targets)}] {t}: build failed {type(e).__name__}", flush=True); np.savez(outp, skipped=str(e)[:200]); continue
    ltok = int(batch.token_length)
    if ltok > a.max_tokens:
        print(f"[{ti+1}/{len(targets)}] {t}: {ltok} tokens > cap, skipped", flush=True); np.savez(outp, skipped=f"tokens {ltok}"); continue
    asym = batch.scheme.token_asym_id[0].cpu().numpy()
    table = gt_reps(a.fb / "ground_truth" / f"{t}.cif")
    chain_cache = {}
    def chain_xyz(c):
        if c in chain_cache: return chain_cache[c]
        et, n_used, off = chain_meta.get((t, c), (None, 0, 0))
        idx = np.nonzero(asym == name2idx[c])[0] if c in name2idx else np.array([], dtype=int)
        if et is None or len(idx) == 0 or len(idx) != n_used or not (("peptide" in et) or ("ribo" in et)):
            chain_cache[c] = None; return None
        pos = np.full((len(idx), 3), np.nan)
        for k in range(len(idx)):
            v = rep_xyz(table, c, k + 1 + off, et)
            if v is not None: pos[k] = v
        chain_cache[c] = (idx, pos); return chain_cache[c]
    padded = _pad_inference_batch(batch, 8).to(device=dev)
    logits = {}
    for mname, m in models.items():
        m._forced_n_recycle = a.recycles
        try:
            with torch.no_grad(), model_autocast(m):
                logits[mname] = m(padded.msa, padded.reference, padded.scheme, padded.sequence, padded.structure, padded.template)[0, :ltok, :ltok].float()
        except torch.cuda.OutOfMemoryError:
            logits = None; break
    if logits is None:
        print(f"[{ti+1}/{len(targets)}] {t}: {ltok} tokens, CUDA OOM, skipped", flush=True); np.savez(outp, skipped=f"oom tokens {ltok}")
        del padded; torch.cuda.empty_cache(); continue
    rec = {}; info = []; raw = {}; ifmeta = []
    for cls, pairs in ifaces[t].items():
        for (c1, c2) in sorted(pairs):
            g1, g2 = chain_xyz(c1), chain_xyz(c2)
            if g1 is None or g2 is None: info.append((cls, c1, c2, "layout")); continue
            (i1, p1), (i2, p2) = g1, g2
            D = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
            valid = ~np.isnan(D)
            if valid.sum() == 0: info.append((cls, c1, c2, "unresolved")); continue
            y = (D < 8.0)[valid]; dtrue = D[valid]
            ii, jj = torch.from_numpy(i1).to(dev), torch.from_numpy(i2).to(dev)
            vt = torch.from_numpy(valid).to(dev)
            kif = len(ifmeta); ifmeta.append(dict(k=kif, cls=cls, c1=c1, c2=c2, n1=len(i1), n2=len(i2)))
            raw[f"if{kif}|tok1"] = i1.astype(np.int32); raw[f"if{kif}|tok2"] = i2.astype(np.int32); raw[f"if{kif}|valid"] = valid
            for mname, lg in logits.items():
                blk = lg[ii][:, jj]
                if not a.light: raw[f"if{kif}|lg|{mname}"] = blk.half().cpu().numpy()
                prob = torch.softmax(blk, -1)
                pc = prob[..., : CB_BIN + 1].sum(-1)[vt].cpu().numpy().astype(np.float32)
                ed = (prob * cen_t).sum(-1)[vt].cpu().numpy().astype(np.float32)
                rec.setdefault(f"{cls}|{mname}|p", []).append(pc); rec.setdefault(f"{cls}|{mname}|e", []).append(ed)
            rec.setdefault(f"{cls}|y", []).append(y); rec.setdefault(f"{cls}|d", []).append(dtrue.astype(np.float32))
            rec.setdefault(f"{cls}|n", []).append(np.full(len(y), len(rec.get(f"{cls}|n", [])), dtype=np.int32))
            info.append((cls, c1, c2, int(y.sum())))
    np.savez_compressed(outp, **{k: np.concatenate(v) for k, v in rec.items()}, **raw, info=json.dumps(info), ifmeta=json.dumps(ifmeta), tokens=ltok)
    # contact-map PNG of the whole assembly: predicted P(contact) for two epochs next to the answer (rep-atom < 8 A) where the chains are scored
    try:
        if a.light: raise RuntimeError('light')
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        gtpos = np.full((ltok, 3), np.nan)
        for c in name2idx:
            g = chain_xyz(c)
            if g is not None: gtpos[g[0]] = g[1]
        Dfull = np.linalg.norm(gtpos[:, None, :] - gtpos[None, :, :], axis=-1)
        gt_map = np.where(np.isnan(Dfull), 0.5, (Dfull < 8.0).astype(float))
        names = [n for n in ("ep300", "ep900") if n in logits] or list(logits)[:2]
        fig, ax = plt.subplots(1, len(names) + 1, figsize=(4.2 * (len(names) + 1), 4.4))
        for a_, n in zip(ax, names):
            a_.imshow(torch.softmax(logits[n], -1)[..., : CB_BIN + 1].sum(-1).cpu().numpy(), cmap="viridis", vmin=0, vmax=1, interpolation="nearest"); a_.set_title(f"{n} P(contact)", fontsize=9)
        ax[-1].imshow(gt_map, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest"); ax[-1].set_title("answer (<8 A; gray = unscored)", fontsize=9)
        bounds = [int(np.nonzero(asym == k)[0][0]) for k in sorted(set(asym.tolist()))][1:]
        for a_ in ax:
            for b_ in bounds: a_.axhline(b_ - 0.5, color="red", lw=0.4); a_.axvline(b_ - 0.5, color="red", lw=0.4)
            a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f"{t}  ({ltok} tokens)", fontsize=10); fig.tight_layout()
        (a.out / "png").mkdir(exist_ok=True); fig.savefig(a.out / "png" / f"{t}.png", dpi=90); plt.close(fig)
    except Exception as e:
        if not a.light: print(f"  png failed: {type(e).__name__} {e}", flush=True)
    print(f"[{ti+1}/{len(targets)}] {t}: {ltok} tok, interfaces {sum(1 for x in info if isinstance(x[3], int))}/{len(info)}  {time.time()-t0:.0f}s", flush=True)
    del padded, logits
print("SHARD_DONE", a.shard, flush=True)
