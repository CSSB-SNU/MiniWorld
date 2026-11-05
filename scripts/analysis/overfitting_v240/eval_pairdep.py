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


import csv
def ap_score(y, p):
    o = np.argsort(-p, kind="stable"); y = y[o]; tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1) * y).sum() / max(y.sum(), 1))


MD = "/home/psk6950/data/BioMolDB_20261014/metadata"
CDR = {}
for r in csv.DictReader(open(f"{MD}/cdr_region.csv")):
    try:
        CDR[r["seqid"]] = [int(r[k]) for k in ("cdr1_start", "cdr1_end", "cdr2_start", "cdr2_end", "cdr3_start", "cdr3_end")]
    except ValueError:
        pass  # empty CDR fields: no annotation
EXTRA = {r["seqid"] for r in csv.DictReader(open(f"{MD}/cdr_region_extra_domains.csv"))}
FAILED = {l.strip() for l in open(f"{MD}/cdr_region_failed.txt") if l.strip()}
ABSEQ = {}
with open(f"{MD}/ab_seq_id.tsv") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        ABSEQ[p[0]] = (int(p[3]), p[4], p[2])
print("CDR table", len(CDR), "extra-domain", len(EXTRA), "failed", len(FAILED), flush=True)

import miniworld.data.dataloader.loading as _loading
_ORIG = _loading._load_a3m_from_paths
MODE = {"abl": False}
AB = ds.preprocessor.antibody_seq_ids


def _patched(seq_id, paths):
    r = _ORIG(seq_id, paths)
    if seq_id in AB and MODE["abl"]:
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
BACKBONE = {"N", "CA", "C", "O", "CB", "OXT"}
CONDS = ("full", "cdr_unk", "fw_unk", "epi_unk", "ctrl_unk", "cdr_swap", "epi_swap")
REG = ("CDR", "FR", "const")


def regions(n, cdr):
    s1, e1, s2, e2, s3, e3 = cdr
    lab = np.full(n, 2, dtype=np.int8)           # constant / outside the variable domain
    lab[: min(n, e3 + 11)] = 1                    # variable-domain framework (FR4 ~ 11 residues after CDR3)
    for s, e in ((s1, e1), (s2, e2), (s3, e3)):
        lab[s:e] = 0                              # CDR (0-based half-open, checked against the sequence below)
    return lab


UNK = 20   # ResidueMapping.AA_UNKNOWN (checked against an ALA token below: ALA must be 0 in both token_type and the MSA numbering)
from miniworld.data.pipeline.msa import GAP_IDX
from miniworld.data.constants.mapping import ResidueMapping


def mask_cdr(b, T, with_tpl):
    """T: absolute token indices of the antibody chain to MASK. The antibody MSA is kept for every other column.
    sequence: token_type -> UNK | MSA: query row -> UNK, every other row -> gap at those columns, profile = one-hot UNK, deletions 0
    reference: side-chain atoms of those residues masked | templates (with_tpl): residue type -> UNK, CB / backbone masks off at those residues."""
    ms = b.msa
    b.sequence.token_type[0, T] = UNK
    ms.aligned_sequences[0][:, T] = GAP_IDX
    ms.aligned_sequences[0, 0, T] = UNK
    ms.profile[0, T] = 0
    ms.profile[0, T, UNK] = 1
    ms.deletion_mean[0, T] = 0
    ms.has_deletion[0][:, T] = 0
    ms.deletion_value[0][:, T] = 0
    a2t = b.scheme.atom_to_token_idx_map[0]
    inT = torch.zeros(int(b.scheme.token_idx.shape[-1]), dtype=torch.bool, device=a2t.device); inT[T] = True
    _an = b.atom_ids[0]; _an = _an.value if hasattr(_an, "value") else _an
    _names = np.char.strip(np.asarray(_an).astype(str)); _n = len(_names)
    side = torch.from_numpy(~np.isin(_names, list(BACKBONE))).to(a2t.device)
    am = torch.zeros(len(a2t), dtype=torch.bool, device=a2t.device); am[:_n] = inT[a2t[:_n]] & side
    b.reference.mask[0, am] = 0; b.reference.element[0, am] = 0; b.reference.charge[0, am] = 0
    if with_tpl:
        res = b.scheme.token_residue_idx[0, T].long()
        tp = b.template
        tp.res_type[0][:, res] = UNK
        tp.cb_mask[0][:, res] = False; tp.bb_mask[0][:, res] = False
        tp.cb_xyz[0][:, res] = 0; tp.bb_xyz[0][:, res] = 0
    return int(am.sum())


import copy
AA2NUM = ResidueMapping().AA2NUM
# donors for the CDR swap: antibodies of the same ANARCI type with exactly the same three CDR lengths
DONORS = {}
for sid_, c_ in CDR.items():
    if sid_ in EXTRA or sid_ in FAILED or sid_ not in ABSEQ: continue
    DONORS.setdefault((ABSEQ[sid_][2], c_[1] - c_[0], c_[3] - c_[2], c_[5] - c_[4]), []).append(sid_)
print("donor classes", len(DONORS), "largest", max(len(v) for v in DONORS.values()), flush=True)


def swap_res(b, T, types):
    """Residue identities of tokens T become `types` (a different, real amino acid each): sequence, EVERY MSA row at those columns, profile, template residue type;
    side-chain atoms of those residues masked (identity cannot leak through the atom composition)."""
    ms = b.msa
    b.sequence.token_type[0, T] = types
    ms.aligned_sequences[0][:, T] = types.unsqueeze(0).expand(ms.aligned_sequences.shape[1], -1)
    ms.profile[0, T] = 0
    ms.profile[0, T, types] = 1
    ms.deletion_mean[0, T] = 0; ms.has_deletion[0][:, T] = 0; ms.deletion_value[0][:, T] = 0
    a2t = b.scheme.atom_to_token_idx_map[0]
    inT = torch.zeros(int(b.scheme.token_idx.shape[-1]), dtype=torch.bool, device=a2t.device); inT[T] = True
    _an = b.atom_ids[0]; _an = _an.value if hasattr(_an, "value") else _an
    _names = np.char.strip(np.asarray(_an).astype(str)); _n = len(_names)
    side = torch.from_numpy(~np.isin(_names, list(BACKBONE))).to(a2t.device)
    am = torch.zeros(len(a2t), dtype=torch.bool, device=a2t.device); am[:_n] = inT[a2t[:_n]] & side
    b.reference.mask[0, am] = 0; b.reference.element[0, am] = 0; b.reference.charge[0, am] = 0
    res = b.scheme.token_residue_idx[0, T].long()
    for i in range(len(res)):
        b.template.res_type[0][:, res[i]] = types[i]


def side_stats(p, top, flag, tokens, pos_flag_tokens):
    """mass share / top-k share on pairs whose token on this side is in the flagged set, and recall of the flagged tokens by the top-k pairs."""
    mass = float(p[flag].sum()); tops = int(flag[top].sum())
    rec = len(set(tokens[top].tolist()) & set(pos_flag_tokens.tolist())) / max(len(pos_flag_tokens), 1)
    return mass, tops, rec


pool = {}
per_item = []
t0 = time.time()
kept = skipped = 0
for k, idx in enumerate(draw):
    if kept >= a.n or time.time() - t0 > 330: break   # 10-minute job: stop taking new items after 5.5 min of the loop
    rec = ds.items[int(idx)]
    try:
        TPL_SEED["s"] = 777 + k
        MODE["abl"] = False
        batch = ds.preprocessor.process(rec, rng=np.random.default_rng(a.seed * 100003 + k))
        b0 = bucketed_collate([batch], bucket_msa_multiple=cfg.train.bucket_msa_multiple, bucket_token_multiple=8,
                              bucket_atom_multiple=64, bucket_template_multiple=4).to(device=dev)
        L = int(b0.structure.token_mask[0].sum())
        if L >= a.cap_tokens - 20: raise RuntimeError("assembly hit the token cap (cropped)")
        target, pair_mask = atom_distogram_target(b0.structure.atom_pos, b0.structure.atom_pos_mask, b0.scheme.atom_to_token_idx_map,
                                                  num_bins=nb, token_num=b0.scheme.token_asym_id.shape[-1], rep_atom_mask=b0.structure.atom_is_rep)
        c1, c2 = (str(c) for c in rec.chain_ids)
        asym = b0.scheme.token_asym_id[0]
        seqs = {c: LAST_SEQ[LAST_CHAINS.index(c)] for c in (c1, c2)}
        abc = [c for c in (c1, c2) if seqs[c] in AB]
        if len(abc) != 1: raise RuntimeError(f"need exactly one antibody chain in the pair, got {abc}")
        abchain = abc[0]; agchain = c2 if abchain == c1 else c1; sid = seqs[abchain]
        if seqs[agchain][0] not in "PQ": raise RuntimeError("antigen chain is not a protein")
        if sid not in CDR or sid in EXTRA or sid in FAILED: raise RuntimeError("no single-domain CDR annotation")
        tokmask = b0.structure.token_mask[0].bool()
        tok_ab = ((asym == LAST_CHAINS.index(abchain)) & tokmask).nonzero(as_tuple=True)[0]
        tok_ag = ((asym == LAST_CHAINS.index(agchain)) & tokmask).nonzero(as_tuple=True)[0]
        if len(tok_ab) != ABSEQ[sid][0]: raise RuntimeError("antibody chain length mismatch")
        _exp = np.array([AA2NUM.get(ch, 20) for ch in ABSEQ[sid][1]]); _got = b0.sequence.token_type[0, tok_ab].cpu().numpy()
        if float((_exp == _got).mean()) < 0.9: raise RuntimeError("antibody token_type does not match the chain sequence")
        _g2 = b0.sequence.token_type[0, tok_ag].cpu().numpy()
        if (_g2 > 20).any(): raise RuntimeError("antigen token_type outside 0..20")
        c = CDR[sid]; reg = regions(len(tok_ab), c)
        m1 = (asym == LAST_CHAINS.index(c1)) & tokmask; m2 = (asym == LAST_CHAINS.index(c2)) & tokmask
        cand = m1[:, None] & m2[None, :] & pair_mask[0]
        ii, jj = cand.nonzero(as_tuple=True)
        if len(ii) == 0 or not bool((target[0][ii, jj] <= CB_BIN).any()): raise RuntimeError("no resolved contact")
        y = (target[0][ii, jj] <= CB_BIN).cpu().numpy().astype(np.float32)
        ab_of_pair = (ii if abchain == c1 else jj).cpu().numpy(); ag_of_pair = (jj if abchain == c1 else ii).cpu().numpy()
        epi = np.unique(ag_of_pair[y > 0])
        cdr_tok = tok_ab[torch.from_numpy(np.nonzero(reg == 0)[0]).to(dev)].cpu().numpy()
        in_epi = np.isin(ag_of_pair, epi); in_cdr = np.isin(ab_of_pair, cdr_tok)
        others_ag = np.setdiff1d(tok_ag.cpu().numpy(), epi)
        fw_tok = tok_ab[torch.from_numpy(np.nonzero(reg == 1)[0]).to(dev)].cpu().numpy()
        if len(others_ag) < len(epi) or len(fw_tok) < len(cdr_tok): raise RuntimeError("too small for same-size controls")
        # donor antibody for the CDR swap
        key = (ABSEQ[sid][2], c[1] - c[0], c[3] - c[2], c[5] - c[4]); mine = "".join(ABSEQ[sid][1][s:e] for s, e in ((c[0], c[1]), (c[2], c[3]), (c[4], c[5])))
        don = []
        for d_ in DONORS.get(key, []):
            dc = CDR[d_]; ds_ = "".join(ABSEQ[d_][1][s:e] for s, e in ((dc[0], dc[1]), (dc[2], dc[3]), (dc[4], dc[5])))
            if ds_ != mine: don.append((d_, ds_))
        if not don: raise RuntimeError(f"no length-matched donor for {key}")
        d_id, d_cdr = don[int(np.random.default_rng(2000 + k).integers(len(don)))]
        donor_types = torch.tensor([AA2NUM.get(ch, 20) for ch in d_cdr], device=dev)
        rng_c = np.random.default_rng(1000 + k)
        sets = {
            "cdr_unk": ("unk", cdr_tok),
            "fw_unk": ("unk", rng_c.choice(fw_tok, size=len(cdr_tok), replace=False)),
            "epi_unk": ("unk", epi),
            "ctrl_unk": ("unk", rng_c.choice(others_ag, size=len(epi), replace=False)),
            "cdr_swap": ("swap", cdr_tok),
            "epi_swap": ("swap", epi),
        }
        epi_swap_types = torch.from_numpy(b0.sequence.token_type[0].cpu().numpy()[rng_c.choice(others_ag, size=len(epi), replace=False)]).to(dev)
        out = {}
        for cond in CONDS:
            if cond == "full": b = b0
            else:
                b = copy.deepcopy(b0)
                kind, T = sets[cond]
                Tt = torch.from_numpy(np.asarray(T)).to(dev)
                if kind == "unk": mask_cdr(b, Tt, True)
                elif cond == "cdr_swap": swap_res(b, Tt, donor_types)
                else: swap_res(b, Tt, epi_swap_types)
            ps = {}
            for name, m in models.items():
                m._forced_n_recycle = a.recycles
                with torch.no_grad(), model_autocast(m):
                    lg = m(b.msa, b.reference, b.scheme, b.sequence, b.structure, b.template)[0].float()
                ps[name] = torch.softmax(lg[ii, jj], -1)[..., : CB_BIN + 1].sum(-1).cpu().numpy().astype(np.float32)
            out[cond] = ps
            if b is not b0: del b
    except Exception as e:  # noqa: BLE001
        if skipped == 0:
            import traceback; traceback.print_exc()
        print(f"[{k + 1}] {rec.record_id}: skipped ({type(e).__name__}: {str(e)[:140]})", flush=True)
        skipped += 1; torch.cuda.empty_cache(); continue
    kk = int(y.sum()); row = dict(item=rec.item_id, tokens=L, contacts=kk, n_epi=len(epi), n_cdr=len(cdr_tok), donor=d_id, donor_cdr=d_cdr, my_cdr=mine)
    for name in models:
        base_top = set(np.argsort(-out["full"][name], kind="stable")[:kk].tolist())
        for cond in CONDS:
            p = out[cond][name]; top = np.argsort(-p, kind="stable")[:kk]
            st = pool.setdefault((cond, name), dict(y=[], p=[], hit=0, k=0, em=0.0, et=0, er=[], cm=0.0, ct=0, cr=[], mass=0.0, jac=[]))
            st["y"].append(y); st["p"].append(p); st["hit"] += int(y[top].sum()); st["k"] += kk; st["mass"] += float(p.sum())
            em, et, er = side_stats(p, top, in_epi, ag_of_pair, epi); cm, ct, cr = side_stats(p, top, in_cdr, ab_of_pair, cdr_tok)
            st["em"] += em; st["et"] += et; st["er"].append(er); st["cm"] += cm; st["ct"] += ct; st["cr"].append(cr)
            st["jac"].append(len(base_top & set(top.tolist())) / max(len(base_top | set(top.tolist())), 1))
            row[f"ap|{cond}|{name}"] = ap_score(y, p)
    kept += 1; per_item.append(row)
    print(f"[{k + 1}] kept {kept}: {row['item']} {L} tok contacts {kk} epitope {len(epi)} CDR {len(cdr_tok)} donor {d_id} ({d_cdr[:20]}) ep900 AP " + " ".join(f"{c}:{row[f'ap|{c}|ep900']:.2f}" for c in CONDS) + f"  {time.time() - t0:.0f}s", flush=True)

lines = [f"{kept} items (skipped {skipped}). unk = identity set to UNK (sequence, MSA columns [query UNK, other rows gap], side-chain atoms, template); swap = replaced by REAL other residues (cdr_swap: the CDRs of a different antibody with identical CDR lengths and ANARCI type; epi_swap: identities of random non-epitope antigen residues), in every MSA row; side-chain atoms masked. Controls fw_unk / ctrl_unk = same number of residues elsewhere.",
         "columns: AP | P@k | antigen side: mass share / top-k share on epitope residues, recall of epitope residues by the top-k pairs | antibody side: same for CDR residues | Jaccard of top-k with full"]
for name in models:
    lines.append(f"--- {name}")
    for cond in CONDS:
        st = pool[(cond, name)]; y = np.concatenate(st["y"]); p = np.concatenate(st["p"])
        lines.append(f"  {cond:9s} AP {ap_score(y, p):.3f} | P@k {st['hit'] / max(st['k'], 1):.3f} | epitope: mass {st['em'] / max(st['mass'], 1e-9):.2f} top-k {st['et'] / max(st['k'], 1):.2f} recall {np.mean(st['er']):.2f} | CDR: mass {st['cm'] / max(st['mass'], 1e-9):.2f} top-k {st['ct'] / max(st['k'], 1):.2f} recall {np.mean(st['cr']):.2f} | jaccard {np.mean(st['jac']):.2f}")
(a.out / "summary.txt").write_text("\n".join(lines) + "\n"); (a.out / "per_item.json").write_text(json.dumps(per_item, default=str))
print("\n".join(lines), flush=True)
print("PAIRDEP_DONE", flush=True)
