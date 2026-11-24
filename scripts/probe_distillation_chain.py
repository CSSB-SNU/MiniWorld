"""Distillation chain-id probe: process() with chain_ids replaced; MSA/template/key diagnostics; per-sample cost."""
import collections, dataclasses, os, sys, time
import numpy as np
from hydra import compose, initialize_config_dir
from run_miniworld_distogram_train import Config
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.dataloader.loading import chain_seq_id, _load_a3m_from_paths
from miniworld.data.io import load_cifmol, load_raw_data
N = int(sys.argv[1]); CHAIN = sys.argv[2]; CFG = os.environ.get("PROBE_CFG", "phase1a_distogram_medium_v240_nr_nt_pf8")
ROOT = os.environ.get("BENCH_ROOT", "/home/psk6950/MiniWorld")
with initialize_config_dir(f"{ROOT}/configs/miniworld", version_base=None):
    cfg = Config.model_validate(compose(config_name=CFG, overrides=[o for o in os.environ.get("PROBE_OVR", "").split() if o]))
ds = BioMolData(BioMolData.BioMolConfig(crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db,
                                        sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer))
n = len(ds); print("catalog", n, flush=True)
ds.preprocessor.load_templates = True
rng0 = np.random.default_rng(11)
SRC = ["protein_monomer", "short_protein_monomer", "pdb", "disordered_pdb"]
idx = rng0.integers(0, n, size=3_000_000)
by = collections.defaultdict(list)
for i in idx:
    s = ds.items[int(i)].source
    if len(by[s]) < N: by[s].append(int(i))
    if all(len(by[x]) >= N for x in SRC): break
print({k: len(v) for k, v in by.items()}, flush=True)
for s in ["protein_monomer", "short_protein_monomer", "disordered_pdb", "pdb"]:
    res = collections.Counter(); t_ok = []; msa_found = 0; msa_len_ok = 0; msa_depth = []; tmpl = collections.Counter(); diag = 0
    for k, i in enumerate(by[s]):
        rec = ds.items[i]
        if s in ("protein_monomer", "short_protein_monomer") and CHAIN != "orig":
            rec = dataclasses.replace(rec, chain_ids=(CHAIN,))
        t0 = time.time()
        try:
            ds.preprocessor.process(rec, rng=np.random.default_rng(k))
            t_ok.append(time.time() - t0); res["ok"] += 1
        except Exception as e:
            res[type(e).__name__ + ": " + str(e)[:70].split(" in cifmol")[0]] += 1
            continue
        if diag < 100 and s != "pdb":
            diag += 1
            try:
                cm = load_cifmol(rec.cif_db_path, rec.record_id, rec.assembly_id, rec.model_id, rec.alt_id)
                cid = list(cm.chains.chain_id.value)[0]
                sid = chain_seq_id(cm, cid)
                paths = ds.resources.msa_paths_for(rec.source, sid)
                msa = _load_a3m_from_paths(sid, paths)
                nres = len(cm.chains.select(chain_id=cid).residues)
                if msa is not None:
                    msa_found += 1; msa_depth.append(len(msa.aligned_sequences))
                    if len(msa.query_sequence) == nres: msa_len_ok += 1
                tp = rec.template_db_paths[0]
                for key in (rec.record_id, f"{rec.record_id}_{cid}", f"{rec.record_id}_1"):
                    tmpl[key.replace(rec.record_id, "<id>")] += int(load_raw_data(key, tp) is not None)
            except Exception as e:
                res["diag_err " + type(e).__name__ + str(e)[:60]] += 1
    print(f"## {s} chain={CHAIN if s in ('protein_monomer','short_protein_monomer') else 'native'} n={len(by[s])} {dict(res)}", flush=True)
    if t_ok: print(f"   ok-sample time mean {np.mean(t_ok):.3f}s median {np.median(t_ok):.3f}s p90 {np.percentile(t_ok,90):.3f}s", flush=True)
    if diag: print(f"   diag({diag}): msa_found {msa_found} len==nres {msa_len_ok} depth median {np.median(msa_depth) if msa_depth else None} template key hits {dict(tmpl)}", flush=True)
