"""Antibody-antigen contact AP on items drawn from the TRAINING database, whole assemblies (crop budgets far above any assembly).

    python eval_train_abag.py --repo $W/mw-v239 --config phase1b_distogram_medium_v239_b200 --n 160 --out DIR ep100=ckpt ep900=ckpt ...

The dataset is built from the config UNCHANGED (so the cached catalog's fingerprint matches and it is only memory-mapped, never rebuilt or overwritten).
`antibody_protein` rows are then drawn with probability proportional to the catalog's (cluster-balanced) weights = the training sampler's distribution, and
preprocessed one by one (`Preprocessor.process`) with crop_config.max_tokens / max_atoms raised to --cap-tokens / --cap-atoms. Every checkpoint sees the SAME
batch (one process). Metric = FoldBench's: contact = representative-atom distance < 8 A (distogram bin <= 23), score = P(bin <= 23), pooled over the
item's (antibody chain, antigen chain) pair when it has at least one contact (FoldBench lists only interfaces that exist), both tokens resolved.
The training DB tags antibody chains 'P', not 'A', so entity-type masks cannot find them (checked on 24 batches).
Templates are the training batch's own (FoldBench runs used none). Inputs vary run to run (the pipeline draws its MSA subsample / focus unseeded).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--n", type=int, default=160, help="items to draw")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cap-tokens", type=int, default=2400)
ap.add_argument("--cap-atoms", type=int, default=24000)
ap.add_argument("--weights", choices=("ema", "raw"), default="ema")
ap.add_argument("--recycles", type=int, default=4)
ap.add_argument("--rows-cache", type=Path, default=Path("/home/psk6950/MiniWorld-scratch/fb-v240/trainabag_rows.npy"))
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("ckpts", nargs="+", help="name=path (full training checkpoint)")
a = ap.parse_args()
sys.path[:0] = [str(a.repo / "scripts"), str(a.repo / "src"), str(a.repo / "libs/team-gm/src")]
from hydra import compose, initialize_config_dir
from run_miniworld_distogram_train import Config

from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.dataloader.collate import bucketed_collate
from miniworld.data.features import convert as _convert
from miniworld.loss.auxiliary import atom_distogram_target
from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel
from miniworld.training.precision import apply_precision, model_autocast

_orig_tcf = _convert.to_chain_features
LAST_CHAINS: list = []
LAST_SEQ: list = []


def _tcf(cifmol, *args, **kwargs):  # remember the chain-id order of the (cropped) assembly: token_asym_id indexes it
    LAST_CHAINS[:] = [str(c) for c in cifmol.chains.chain_id.value]; LAST_SEQ[:] = [str(s) for s in cifmol.chains.seq_id.value]
    return _orig_tcf(cifmol, *args, **kwargs)


_convert.to_chain_features = _tcf
CB_BIN = 23  # bin 23 closes at 8.0 A (edges 2.25 + 0.25 i)
dev = torch.device("cuda")
a.out.mkdir(parents=True, exist_ok=True)

with initialize_config_dir(str(a.repo / "configs/miniworld"), version_base=None):
    cfg = Config.model_validate(compose(config_name=a.config))
cfg.data.crop.max_tokens = a.cap_tokens
cfg.data.crop.max_atoms = a.cap_atoms
nb = cfg.model.shared.n_distogram_bins

ds = BioMolData(
    BioMolData.BioMolConfig(
        crop_config=cfg.data.crop,
        msa_config=cfg.data.msa,
        DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler,  # unchanged: its fingerprint must match the cached catalog
        tokenizer_config=cfg.data.tokenizer,
    )
)
print(f"catalog: {len(ds.items)} rows; crop budget {cfg.data.crop.max_tokens} tokens / {cfg.data.crop.max_atoms} atoms", flush=True)
tab = ds.items._t
rows_cache = a.rows_cache
if rows_cache.exists():
    sel = np.load(rows_cache)
else:  # the catalog has no bucket column: re-derive the antibody_protein rows from train_item.tsv exactly as reweight_catalog_abag.py does
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv

    from miniworld.data.dataloader.sources import _pdb_edge_type

    tsv = pacsv.read_csv(
        cfg.data.train_db.train_item_path,
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in (
            "source", "cluster1", "cluster2", "pdb_id", "assembly_id", "model_id", "alt_id", "chain_id1", "chain_id2")}),
    )
    sources = set(pc.unique(tab.column("source")).to_pylist())
    tsv = tsv.filter(pc.is_in(tsv.column("source"), pa.array(sorted(sources))))
    assert tsv.num_rows == len(tab), f"train_item rows {tsv.num_rows} != catalog rows {len(tab)}"
    src_col = tsv.column("source")
    is_pdb = pc.equal(src_col, "pdb")
    record = pc.if_else(is_pdb, pc.utf8_lower(tsv.column("pdb_id")), tsv.column("pdb_id"))
    mono = pc.equal(tsv.column("cluster2"), "None")
    chains = pc.if_else(mono, tsv.column("chain_id1"), pc.binary_join_element_wise(tsv.column("chain_id1"), tsv.column("chain_id2"), ":"))
    derived = pc.binary_join_element_wise(src_col, record, tsv.column("assembly_id"), tsv.column("model_id"), tsv.column("alt_id"), chains, ":")
    assert pc.all(pc.equal(derived, tab.column("item_id"))).as_py(), "train_item.tsv order does not match the catalog"
    edge = pc.if_else(mono, tsv.column("cluster1"), pc.binary_join_element_wise(tsv.column("cluster1"), tsv.column("cluster2"), "_"))
    pdb_idx = np.flatnonzero(is_pdb.to_numpy(zero_copy_only=False))
    pdb_edges = edge.take(pa.array(pdb_idx)).to_numpy(zero_copy_only=False)
    uniq, inverse = np.unique(pdb_edges, return_inverse=True)
    is_bucket_edge = np.array([_pdb_edge_type(e) == "antibody_protein" for e in uniq])
    sel = pdb_idx[is_bucket_edge[inverse]]
    np.save(rows_cache, sel)
raw_w = tab.column("weight").to_numpy()
w = raw_w[sel].astype(np.float64)
print(f"antibody_protein rows: {len(sel)} ({100 * len(sel) / len(tab):.2f}% of rows), raw weight share of the catalog {w.sum() / raw_w.sum():.3f}", flush=True)
rng = np.random.default_rng(a.seed)
draw = rng.choice(sel, size=a.n, replace=False, p=w / w.sum())

models = {}
for spec_ in a.ckpts:
    name, path = spec_.split("=", 1)
    mcfg_name = None
    if "@" in path:  # name=path@model_config: the model is built from that config (data always from --config)
        path, mcfg_name = path.split("@", 1)
    if mcfg_name:
        with initialize_config_dir(str(a.repo / "configs/miniworld"), version_base=None):
            mcfg = Config.model_validate(compose(config_name=mcfg_name))
    else:
        mcfg = cfg
    st = torch.load(path, map_location="cpu", weights_only=False)
    m = apply_precision(MiniSWAModel(mcfg.model), "bf16-mixed")
    key = "ema_state_dict" if (a.weights == "ema" and "ema_state_dict" in st) else "model_state_dict"
    r = m.load_state_dict(st[key], strict=False)
    print(f"{name}: {key}, epoch {st.get('epoch')} missing {len(r.missing_keys)} unexpected {len(r.unexpected_keys)}", flush=True)
    models[name] = m.to(dev).eval()
    del st



def ap_score(y, p):
    o = np.argsort(-p, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1) * y).sum() / max(y.sum(), 1))


# ---- ablation: the antibody chains (seq_id in the antibody list) are loaded as if they had NO a3m, exactly what the pipeline does for a chain without an MSA
import miniworld.data.dataloader.loading as _loading
_ORIG = _loading._load_a3m_from_paths
MODE = {"abl": False}
DEPTH = {}
AB = ds.preprocessor.antibody_seq_ids


def _patched(seq_id, paths):
    r = _ORIG(seq_id, paths)
    if seq_id in AB:
        DEPTH[seq_id] = None if r is None else int(r.aligned_sequences.shape[0])
        if MODE["abl"]:
            return None
    return r


_loading._load_a3m_from_paths = _patched
import miniworld.data.dataloader.preprocess as _prep
_TPL_ORIG = _prep.load_record_templates
TPL_SEED = {"s": 0}


def _tpl(**kw):  # the template draw must not depend on how many random numbers the MSA sampling used before it
    kw["rng"] = np.random.default_rng(TPL_SEED["s"])
    return _TPL_ORIG(**kw)


_prep.load_record_templates = _tpl
CONDS = ("full", "no_ab_msa")
pool = {(c, n): ([], []) for c in CONDS for n in models}
per_item = []
t0 = time.time()
kept = skipped = 0
for k, idx in enumerate(draw):
    if kept >= a.n:
        break
    rec = ds.items[int(idx)]
    res, ys, meta, prev = {}, {}, {}, None
    TPL_SEED["s"] = 777 + k
    try:
        for cond in CONDS:
            MODE["abl"] = cond == "no_ab_msa"
            DEPTH.clear()
            batch = ds.preprocessor.process(rec, rng=np.random.default_rng(a.seed * 100003 + k))
            b = bucketed_collate([batch], bucket_msa_multiple=cfg.train.bucket_msa_multiple, bucket_token_multiple=8,
                                 bucket_atom_multiple=64, bucket_template_multiple=4).to(device=dev)
            L = int(b.structure.token_mask[0].sum())
            target, pair_mask = atom_distogram_target(b.structure.atom_pos, b.structure.atom_pos_mask, b.scheme.atom_to_token_idx_map,
                                                      num_bins=nb, token_num=b.scheme.token_asym_id.shape[-1], rep_atom_mask=b.structure.atom_is_rep)
            c1, c2 = (str(c) for c in rec.chain_ids)
            if c1 not in LAST_CHAINS or c2 not in LAST_CHAINS:
                raise RuntimeError("item chains not in assembly")
            asym = b.scheme.token_asym_id[0]
            m1 = (asym == LAST_CHAINS.index(c1)) & b.structure.token_mask[0].bool()
            m2 = (asym == LAST_CHAINS.index(c2)) & b.structure.token_mask[0].bool()
            cand = m1[:, None] & m2[None, :] & pair_mask[0]
            ii, jj = cand.nonzero(as_tuple=True)
            if len(ii) == 0 or not bool((target[0][ii, jj] <= CB_BIN).any()):
                raise RuntimeError("no resolved contact")
            y = (target[0][ii, jj] <= CB_BIN).cpu().numpy().astype(np.float32)
            ys[cond] = y
            if cond == "full":
                meta = {"item": rec.item_id, "tokens": L, "pairs": int(len(y)), "contacts": int(y.sum()), "chains": [c1, c2],
                        "ab_chains_in_assembly": [LAST_CHAINS[i] for i, s in enumerate(LAST_SEQ) if s in AB],
                        "ab_msa_rows": dict(DEPTH), "msa_rows_total": int(b.msa.aligned_sequences.shape[1])}
                prev = b
            else:
                meta["msa_rows_total_abl"] = int(b.msa.aligned_sequences.shape[1])
                meta["same_y"] = bool(len(ys["full"]) == len(y) and np.array_equal(ys["full"], y))
                meta["same_template"] = bool(torch.equal(prev.template.res_type, b.template.res_type) and torch.equal(prev.template.mask, b.template.mask))
            for name, m in models.items():
                m._forced_n_recycle = a.recycles
                with torch.no_grad(), model_autocast(m):
                    lg = m(b.msa, b.reference, b.scheme, b.sequence, b.structure, b.template)[0].float()
                p = torch.softmax(lg[ii, jj], -1)[..., : CB_BIN + 1].sum(-1).cpu().numpy().astype(np.float32)
                res[(cond, name)] = p
        if not (meta.get("same_y") and meta.get("same_template")):
            raise RuntimeError(f"conditions differ beyond the MSA: same_y={meta.get('same_y')} same_template={meta.get('same_template')}")
    except Exception as e:  # noqa: BLE001
        print(f"[{k + 1}] {rec.record_id}: skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
        skipped += 1
        torch.cuda.empty_cache()
        continue
    y = ys["full"]
    for (cond, name), p in res.items():
        meta[f"ap|{cond}|{name}"] = ap_score(y, p)
        pool[(cond, name)][0].append(y)
        pool[(cond, name)][1].append(p)
    kept += 1
    per_item.append(meta)
    print(f"[{k + 1}] kept {kept}: {meta['item']} {meta['tokens']} tok, {meta['contacts']} contacts, antibody chains {meta['ab_chains_in_assembly']} MSA rows {meta['ab_msa_rows']}  "
          + " ".join(f"{n}:{meta[f'ap|full|{n}']:.2f}->{meta[f'ap|no_ab_msa|{n}']:.2f}" for n in models) + f"  {time.time() - t0:.0f}s", flush=True)

lines = [f"{kept} items (skipped {skipped}); pooled AP, full MSA vs antibody MSA removed (same crop, same templates, same antigen MSA)"]
lines.append("model        full    no_ab_msa   delta | mean per-item AP full -> no_ab_msa")
for name in models:
    r = {}
    for cond in CONDS:
        y = np.concatenate(pool[(cond, name)][0]); p = np.concatenate(pool[(cond, name)][1])
        r[cond] = ap_score(y, p)
    mi = {c: np.mean([it[f"ap|{c}|{name}"] for it in per_item]) for c in CONDS}
    lines.append(f"{name:11s} {r['full']:.3f}   {r['no_ab_msa']:.3f}     {r['no_ab_msa'] - r['full']:+.3f} | {mi['full']:.3f} -> {mi['no_ab_msa']:.3f}")
(a.out / "summary.txt").write_text("\n".join(lines) + "\n")
(a.out / "per_item.json").write_text(json.dumps(per_item, default=str))
print("\n".join(lines), flush=True)
print("ABLATE_DONE", flush=True)
