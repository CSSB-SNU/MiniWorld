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
CONDS = ("full", "cdr_gap_all", "cdr_swap_all", "cdr_swap_rows", "cdr_swap_stats", "fw_swap_all")
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
# donors: same ANARCI type and IDENTICAL CDR boundaries (CDR columns line up 1:1)
DONORS = {}
for sid_, c_ in CDR.items():
    if sid_ in EXTRA or sid_ in FAILED or sid_ not in ABSEQ: continue
    DONORS.setdefault((ABSEQ[sid_][2], *c_), []).append(sid_)
print("donor classes", len(DONORS), "largest", max(len(v) for v in DONORS.values()), flush=True)

SWAP = {"sid": None, "cols": None, "donor": None, "rows": False, "stats": False}   # active MSA swap for the next process() call
_LOAD = _loading._load_a3m_from_paths          # the original loader (the head wrapped it for the no-MSA mode; MODE['abl'] stays False)


def _patched2(seq_id, paths):
    msa = _LOAD(seq_id, paths)
    if SWAP["sid"] is None or seq_id != SWAP["sid"] or msa is None: return msa
    d = SWAP["donor"]; cols = SWAP["cols"]
    hyb = copy.copy(msa)
    A = msa.aligned_sequences.copy(); Dl = msa.deletions.copy(); nA = A.shape[0]; nD = d.aligned_sequences.shape[0]
    if SWAP["rows"]:
        idx = np.arange(nA) % nD                    # donor rows in their stored (best-first) order; cycled only if the donor MSA is shallower
        A[1:, cols] = d.aligned_sequences[idx[1:]][:, cols]; Dl[1:, cols] = d.deletions[idx[1:]][:, cols]     # row 0 = the query keeps its own residues
        hyb.aligned_sequences = A; hyb.deletions = Dl
    if SWAP["stats"]:
        prof = msa.profile.copy(); prof[cols] = d.profile[cols]; hyb.profile = prof
        dm = msa.deletion_mean.copy(); dm[cols] = d.deletion_mean[cols]; hyb.deletion_mean = dm
    return hyb


_loading._load_a3m_from_paths = _patched2


def msa_gap(b, T):
    ms = b.msa
    q = ms.aligned_sequences[0, 0, T].clone()
    ms.aligned_sequences[0][1:, T] = GAP_IDX
    ms.has_deletion[0][1:, T] = 0; ms.deletion_value[0][1:, T] = 0
    ms.profile[0, T] = 0; ms.profile[0, T, q] = 1
    ms.deletion_mean[0, T] = 0; ms.has_deletion[0][0, T] = 0; ms.deletion_value[0][0, T] = 0


def side_stats(p, top, flag, tokens, flag_tokens):
    mass = float(p[flag].sum()); tops = int(flag[top].sum())
    rec = len(set(tokens[top].tolist()) & set(flag_tokens.tolist())) / max(len(flag_tokens), 1)
    return mass, tops, rec


def run_models(b, ii, jj):
    ps = {}
    for name, m in models.items():
        m._forced_n_recycle = a.recycles
        with torch.no_grad(), model_autocast(m):
            lg = m(b.msa, b.reference, b.scheme, b.sequence, b.structure, b.template)[0].float()
        ps[name] = torch.softmax(lg[ii, jj], -1)[..., : CB_BIN + 1].sum(-1).cpu().numpy().astype(np.float32)
    return ps


def collate(batch):
    return bucketed_collate([batch], bucket_msa_multiple=cfg.train.bucket_msa_multiple, bucket_token_multiple=8,
                            bucket_atom_multiple=64, bucket_template_multiple=4).to(device=dev)


def point_mutate(b, T, types):
    """Residues T get the new identities `types`: sequence, the QUERY row of the MSA, template residue type. The rest of the MSA (here: another antibody's CDR columns) is untouched.
    Side-chain atoms of those residues are masked so the old identity cannot leak through the atom composition."""
    b.sequence.token_type[0, T] = types
    b.msa.aligned_sequences[0, 0, T] = types
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


KS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 99)
curves = {}      # (name, kind) -> {k: [ap per item]}
firstdrop = {}   # (name, kind) -> list of the first k where AP < 0.5 * AP at k=0
base_ap = {}
rows_out = []
t0 = time.time(); kept = skipped = 0
for k_it, idx in enumerate(draw):
    if kept >= 16 or time.time() - t0 > 330: break
    rec = ds.items[int(idx)]
    try:
        TPL_SEED["s"] = 777 + k_it; MODE["abl"] = False; SWAP.update(sid=None)
        batch = ds.preprocessor.process(rec, rng=np.random.default_rng(a.seed * 100003 + k_it)); b0 = collate(batch)
        L = int(b0.structure.token_mask[0].sum())
        if L >= a.cap_tokens - 20: raise RuntimeError("assembly hit the token cap (cropped)")
        target, pair_mask = atom_distogram_target(b0.structure.atom_pos, b0.structure.atom_pos_mask, b0.scheme.atom_to_token_idx_map,
                                                  num_bins=nb, token_num=b0.scheme.token_asym_id.shape[-1], rep_atom_mask=b0.structure.atom_is_rep)
        c1, c2 = (str(c) for c in rec.chain_ids)
        asym = b0.scheme.token_asym_id[0]
        seqs = {c: LAST_SEQ[LAST_CHAINS.index(c)] for c in (c1, c2)}
        abc = [c for c in (c1, c2) if seqs[c] in AB]
        if len(abc) != 1: raise RuntimeError("need exactly one antibody chain in the pair")
        abchain = abc[0]; sid = seqs[abchain]
        if sid not in ABSEQ or ABSEQ[sid][2] != "H": raise RuntimeError("the evaluated antibody chain is not a heavy chain")
        if sid not in CDR or sid in EXTRA or sid in FAILED: raise RuntimeError("no single-domain CDR annotation")
        tokmask = b0.structure.token_mask[0].bool()
        tok_ab = ((asym == LAST_CHAINS.index(abchain)) & tokmask).nonzero(as_tuple=True)[0]
        if len(tok_ab) != ABSEQ[sid][0]: raise RuntimeError("antibody chain length mismatch")
        c = CDR[sid]; reg = regions(len(tok_ab), c)
        _exp = np.array([AA2NUM.get(ch, 20) for ch in ABSEQ[sid][1]]); _got = b0.sequence.token_type[0, tok_ab].cpu().numpy()
        if float((_exp == _got).mean()) < 0.9: raise RuntimeError("token_type does not match the chain sequence")
        m1 = (asym == LAST_CHAINS.index(c1)) & tokmask; m2 = (asym == LAST_CHAINS.index(c2)) & tokmask
        cand = m1[:, None] & m2[None, :] & pair_mask[0]
        ii, jj = cand.nonzero(as_tuple=True)
        if len(ii) == 0 or not bool((target[0][ii, jj] <= CB_BIN).any()): raise RuntimeError("no resolved contact")
        y = (target[0][ii, jj] <= CB_BIN).cpu().numpy().astype(np.float32)
        cdr_pos = np.nonzero(reg == 0)[0]; h3_pos = np.arange(c[4], c[5])
        # donor MSA for all CDR columns (random other antibody, identical CDR boundaries)
        mine_msa = _LOAD(sid, ds.preprocessor.resources.msa_paths_for(rec.source, sid))
        if mine_msa is None: raise RuntimeError("the antibody has no a3m of its own")
        cands = [d_ for d_ in DONORS.get((ABSEQ[sid][2], *c), []) if d_ != sid]
        np.random.default_rng(2000 + k_it).shuffle(cands)
        donor = None; d_id = None
        for d_ in cands[:12]:
            dm_ = _LOAD(d_, ds.preprocessor.resources.msa_paths_for(rec.source, d_))
            if dm_ is None or dm_.aligned_sequences.shape[1] <= int(cdr_pos.max()): continue
            if donor is None or dm_.aligned_sequences.shape[0] >= mine_msa.aligned_sequences.shape[0]:
                donor, d_id = dm_, d_
                if dm_.aligned_sequences.shape[0] >= mine_msa.aligned_sequences.shape[0]: break
        if donor is None: raise RuntimeError("no donor with an a3m")
        SWAP.update(sid=sid, cols=cdr_pos, donor=donor, rows=True, stats=True)
        batch = ds.preprocessor.process(rec, rng=np.random.default_rng(a.seed * 100003 + k_it)); bB = collate(batch)
        SWAP.update(sid=None)
        if not (torch.equal(bB.sequence.token_type, b0.sequence.token_type) and torch.equal(bB.template.res_type, b0.template.res_type)):
            raise RuntimeError("sequence/template differ after the CDR MSA swap")
        n3 = len(h3_pos); rngm = np.random.default_rng(3000 + k_it)
        order = rngm.permutation(n3)                                              # mutation order inside CDR-H3 (nested: k mutations are the first k of this order)
        orig_types = b0.sequence.token_type[0, tok_ab[torch.from_numpy(h3_pos).to(dev)]].cpu().numpy()
        new_types = np.array([rngm.choice([x for x in range(20) if x != t]) for t in orig_types])     # a different real amino acid at every position
        full_p = run_models(b0, ii, jj)
        base_p = run_models(bB, ii, jj)
        ks = [kk_ for kk_ in KS if kk_ <= n3] + ([n3] if n3 not in KS else [])
        ks = sorted(set(ks))
        res = {"full": {0: full_p}, "base": {0: base_p}, "mut": {}, "self": {}}
        for kk_ in ks:
            pos = h3_pos[order[:kk_]]; T = tok_ab[torch.from_numpy(pos).to(dev)]
            for kind in ("mut", "self"):
                b = copy.deepcopy(bB)
                types = torch.from_numpy(new_types[order[:kk_]]).to(dev) if kind == "mut" else torch.from_numpy(orig_types[order[:kk_]]).to(dev)
                point_mutate(b, T, types)
                res[kind][kk_] = run_models(b, ii, jj); del b
    except Exception as e:  # noqa: BLE001
        SWAP.update(sid=None)
        if skipped == 0:
            import traceback; traceback.print_exc()
        print(f"[{k_it + 1}] {rec.record_id}: skipped ({type(e).__name__}: {str(e)[:130]})", flush=True)
        skipped += 1; torch.cuda.empty_cache(); continue
    kept += 1; kk_tot = int(y.sum())
    row = dict(item=rec.item_id, tokens=L, contacts=kk_tot, h3_len=n3, h3=ABSEQ[sid][1][c[4]:c[5]], donor=d_id, ks=ks)
    for name in models:
        ap0 = ap_score(y, res["base"][0][name]); apf = ap_score(y, res["full"][0][name]); row[f"ap_full|{name}"] = apf; row[f"ap_base|{name}"] = ap0
        for kind in ("mut", "self"):
            fd = None
            for kk_ in ks:
                v = ap_score(y, res[kind][kk_][name]); row[f"ap_{kind}|{name}|{kk_}"] = v
                curves.setdefault((name, kind), {}).setdefault(kk_, []).append(v)
                if fd is None and v < 0.5 * ap0: fd = kk_
            firstdrop.setdefault((name, kind), []).append(fd if fd is not None else 999)
        curves.setdefault((name, "base"), {}).setdefault(0, []).append(ap0); curves.setdefault((name, "full"), {}).setdefault(0, []).append(apf)
    rows_out.append(row)
    print(f"[{k_it + 1}] kept {kept}: {rec.item_id} {L} tok H3 {ABSEQ[sid][1][c[4]:c[5]]} (len {n3}) ep900 AP full {row['ap_full|ep900']:.2f} base {row['ap_base|ep900']:.2f} mut " + " ".join(f"k{kk_}:{row[f'ap_mut|ep900|{kk_}']:.2f}" for kk_ in ks[:6]) + f"  {time.time() - t0:.0f}s", flush=True)

lines = [f"{kept} items with a heavy-chain antibody (skipped {skipped}). Base = every CDR column of the antibody MSA taken from another antibody (random donor), sequences original. Then k random residues of CDR-H3 mutated to a different random amino acid (query sequence, query MSA row, template residue type; their side-chain atoms masked); 'self' control = the same k residues rewritten with their OWN identity (same processing, no mutation). AP pooled over items that have at least k H3 residues (k = 99 means all H3 residues)."]
for name in models:
    lines.append(f"--- {name}: pooled AP by number of mutated H3 residues")
    y_by_item = None
    f0 = np.mean(curves[(name, "full")][0]); b0_ = np.mean(curves[(name, "base")][0])
    lines.append(f"  original input AP {f0:.3f}; with the donor CDR MSA (k=0) AP {b0_:.3f}")
    lines.append("   k  items |  mutated AP | self-control AP | mutated / base")
    allk = sorted(set(kk_ for r in rows_out for kk_ in r["ks"]))
    for kk_ in allk:
        v = curves[(name, "mut")].get(kk_, []); s = curves[(name, "self")].get(kk_, [])
        if not v: continue
        lines.append(f"  {kk_:3d}  {len(v):4d}  |   {np.mean(v):.3f}     |    {np.mean(s):.3f}        |  {np.mean(v) / max(b0_, 1e-9):.2f}")
    for kind in ("mut", "self"):
        fd = np.array(firstdrop[(name, kind)]); has = fd[fd < 999]
        lines.append(f"  {kind}: AP first falls below half of the k=0 value at k = " + (f"median {np.median(has):.0f} (items where it happens: {len(has)} of {len(fd)})" if len(has) else f"never in the tested range (0 of {len(fd)} items)"))
(a.out / "summary.txt").write_text("\n".join(lines) + "\n"); (a.out / "per_item.json").write_text(json.dumps(rows_out, default=str))
print("\n".join(lines), flush=True)
print("H3MUT_DONE", flush=True)
