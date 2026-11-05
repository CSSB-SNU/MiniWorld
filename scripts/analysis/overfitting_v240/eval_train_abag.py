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


def _tcf(cifmol, *args, **kwargs):  # remember the chain-id order of the (cropped) assembly: token_asym_id indexes it
    LAST_CHAINS[:] = [str(c) for c in cifmol.chains.chain_id.value]
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


def f1_max(y, p):
    o = np.argsort(-p, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    pr = tp / np.arange(1, len(y) + 1)
    rc = tp / max(y.sum(), 1)
    return float((2 * pr * rc / np.maximum(pr + rc, 1e-9)).max())


def bce(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


pool = {n: ([], []) for n in models}  # name -> (y list, p list) pooled over items
per_item = []
t0 = time.time()
kept = skipped = 0
for k, idx in enumerate(draw):
    rec = ds.items[int(idx)]
    try:
        batch = ds.preprocessor.process(rec, rng=np.random.default_rng(a.seed * 100003 + k))
    except Exception as e:  # noqa: BLE001
        print(f"[{k + 1}/{len(draw)}] {rec.record_id}: preprocess failed {type(e).__name__}", flush=True)
        skipped += 1
        continue
    b = bucketed_collate([batch], bucket_msa_multiple=cfg.train.bucket_msa_multiple, bucket_token_multiple=8,
                         bucket_atom_multiple=64, bucket_template_multiple=4).to(device=dev)
    L = int(b.structure.token_mask[0].sum())
    target, pair_mask = atom_distogram_target(
        b.structure.atom_pos, b.structure.atom_pos_mask, b.scheme.atom_to_token_idx_map,
        num_bins=nb, token_num=b.scheme.token_asym_id.shape[-1], rep_atom_mask=b.structure.atom_is_rep,
    )
    # the training DB tags antibody chains 'P' (no 'A'), so the entity-type ab_ag mask is empty: take the item's own (antibody chain, antigen chain) pair
    c1, c2 = (str(c) for c in rec.chain_ids)
    if c1 not in LAST_CHAINS or c2 not in LAST_CHAINS:
        print(f"[{k + 1}/{len(draw)}] {rec.record_id}: {L} tok, item chains {rec.chain_ids} not in the assembly {LAST_CHAINS[:6]}", flush=True)
        skipped += 1
        continue
    asym = b.scheme.token_asym_id[0]
    m1 = (asym == LAST_CHAINS.index(c1)) & b.structure.token_mask[0].bool()
    m2 = (asym == LAST_CHAINS.index(c2)) & b.structure.token_mask[0].bool()
    cand = m1[:, None] & m2[None, :] & pair_mask[0]
    ii, jj = cand.nonzero(as_tuple=True)
    if len(ii) == 0 or not bool((target[0][ii, jj] <= CB_BIN).any()):
        print(f"[{k + 1}/{len(draw)}] {rec.record_id}: {L} tok, no resolved interface contact between {c1} and {c2}", flush=True)
        skipped += 1
        continue
    y = (target[0][ii, jj] <= CB_BIN).cpu().numpy().astype(np.float32)
    row = {"item": rec.item_id, "tokens": L, "pairs": int(len(y)), "contacts": int(y.sum()), "chains": [c1, c2]}
    ok = True
    for name, m in models.items():
        m._forced_n_recycle = a.recycles
        try:
            with torch.no_grad(), model_autocast(m):
                lg = m(b.msa, b.reference, b.scheme, b.sequence, b.structure, b.template)[0].float()
        except torch.cuda.OutOfMemoryError:
            ok = False
            break
        p = torch.softmax(lg[ii, jj], -1)[..., : CB_BIN + 1].sum(-1).cpu().numpy().astype(np.float32)
        row[f"ap|{name}"] = ap_score(y, p)
        pool[name][0].append(y)
        pool[name][1].append(p)
    if not ok:
        for name in models:  # drop this item from every model (same inputs for all)
            if len(pool[name][0]) > kept:
                pool[name][0].pop()
                pool[name][1].pop()
        print(f"[{k + 1}/{len(draw)}] {rec.record_id}: {L} tok, CUDA OOM, skipped", flush=True)
        skipped += 1
        torch.cuda.empty_cache()
        continue
    kept += 1
    per_item.append(row)
    print(f"[{k + 1}/{len(draw)}] kept {kept}: {rec.item_id} {L} tok, {row['pairs']} pairs / {row['contacts']} contacts  {time.time() - t0:.0f}s", flush=True)

if not per_item:
    print("no usable items", flush=True)
    sys.exit(1)
lines = [f"{a.config}: {kept} training antibody-antigen items (skipped {skipped}), tokens mean {np.mean([r['tokens'] for r in per_item]):.0f} max {max(r['tokens'] for r in per_item)}; "
         f"{sum(r['pairs'] for r in per_item)} pairs, {sum(r['contacts'] for r in per_item)} contacts"]
lines.append("name        pooled AP   F1max    BCE   | mean per-item AP")
for name in models:
    y = np.concatenate(pool[name][0])
    p = np.concatenate(pool[name][1])
    lines.append(f"{name:10s}  {ap_score(y, p):.3f}      {f1_max(y, p):.3f}   {bce(y, p):.4f} | {np.mean([r['ap|' + name] for r in per_item]):.3f}")
(a.out / "summary.txt").write_text("\n".join(lines) + "\n")
(a.out / "per_item.json").write_text(json.dumps(per_item))
print("\n".join(lines), flush=True)
print("TRAINABAG_DONE", flush=True)
