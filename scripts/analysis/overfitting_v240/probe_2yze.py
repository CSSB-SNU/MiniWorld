"""What happens to the 2yze training items in the pipeline? Per chain coordinate availability in the cif LMDB, then preprocess every 2yze train item with several seeds."""
import sys, traceback, collections
import numpy as np
sys.path[:0] = ["/home/psk6950/MiniWorld/scripts", "/home/psk6950/MiniWorld/src", "/home/psk6950/MiniWorld/libs/team-gm/src"]
from hydra import compose, initialize_config_dir
from run_miniworld_diffusion_train import Config
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.dataloader.types import DataRecord
from miniworld.data.io.load import load_cifmol
PID = sys.argv[1].lower()
rows = []
with open("/home/psk6950/data/filtering/edge_node/train_item_20210930.tsv") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[3].lower() == PID: rows.append(p)
print(f"{PID}: {len(rows)} train items", flush=True)
with initialize_config_dir("/home/psk6950/MiniWorld/configs/miniworld", version_base=None):
    cfg = Config.model_validate(compose(config_name="phase2a_diffusion_medium_v240", overrides=["data=BioMolDB_20261014", "data.train_db.catalog_cache_path=/home/psk6950/data/BioMolDB_20261014/catalog_trainitem_no_disordered_unified_v240.arrow"]))
ds = BioMolData(BioMolData.BioMolConfig(crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db, sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer))
dbp = cfg.data.train_db.pdb.cif_db_path
for asm in sorted({(r[4], r[5], r[6]) for r in rows}):
    cm = load_cifmol(db_path=dbp, pdb_id=PID, assembly_id=asm[0], model_id=asm[1], alt_id=asm[2])
    xyz = cm.atoms.xyz.value
    print(f"assembly {asm}: atoms {len(xyz)} finite-xyz {int(np.isfinite(xyz).all(1).sum())} ({100 * np.isfinite(xyz).all(1).mean():.1f}%) residues {len(cm.residues.chem_comp_id.value)}", flush=True)
    for c in cm.chains.chain_id.value:
        sub = cm.chains.select(chain_id=c)
        sx = sub.atoms.xyz.value
        print(f"   chain {c}: seq_id {sub.seq_id.value[0]} residues {len(sub.residues.chem_comp_id.value)} atoms {len(sx)} finite {int(np.isfinite(sx).all(1).sum())}", flush=True)
res = collections.Counter(); first_tb = {}
for r in rows:
    rec = DataRecord(item_id=f"pdb:{PID}:{r[4]}:{r[5]}:{r[6]}:{r[7]}:{r[8]}", source="pdb", record_id=PID, cif_db_path=dbp, assembly_id=r[4], model_id=r[5], alt_id=r[6],
                     chain_ids=tuple(x for x in (r[7], r[8]) if x != "None"), feature_keys=(), seq_ids=(), msa_db_paths=(), template_db_paths=())
    for seed in range(3):
        try:
            b = ds.preprocessor.process(rec, rng=np.random.default_rng(seed))
            m = int(b.structure.atom_pos_mask.sum()); res["ok"] += 1
            if m == 0: res["ok but atom_pos_mask all zero"] += 1
            if seed == 0: print(f"  {rec.item_id} ok: tokens {int(b.structure.token_mask.sum())} atoms {int(b.structure.atom_mask.sum())} atom_pos_mask sum {m}", flush=True)
        except Exception as e:  # noqa: BLE001
            key = f"{type(e).__name__}: {str(e)[:110]}"; res[key] += 1
            if seed == 0: print(f"  FAIL {rec.item_id} chains {rec.chain_ids}: {key}", flush=True)
            if key not in first_tb: first_tb[key] = traceback.format_exc()
print("chains present per assembly:", {a_: sorted(set(str(c) for c in load_cifmol(db_path=dbp, pdb_id=PID, assembly_id=a_, model_id="1", alt_id=".").chains.chain_id.value)) for a_ in ("1", "2")}, flush=True)
print("\nRESULT counts:", dict(res), flush=True)
for k, tb in first_tb.items(): print("\nFIRST TRACEBACK for", k, "\n", tb[-1500:], flush=True)
print("PROBE_DONE", flush=True)
