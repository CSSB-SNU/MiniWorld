"""3-D data for the crop viewer: for ~9 of the 60 items, the backbone (N, CA, C, O, CB) of both Ab/Ag chains plus every residue of any other chain that one of the 8 crops contains, with a per-residue
bitmask of the crops (same seeds as crop_abag.py) and the crop focus point. Output crop_viz_all.json."""
import json, sys, time
import numpy as np
sys.path[:0] = ["/home/psk6950/MiniWorld/scripts", "/home/psk6950/MiniWorld/src", "/home/psk6950/MiniWorld/libs/team-gm/src"]
from hydra import compose, initialize_config_dir
from run_miniworld_distogram_train import Config
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.io.load import load_cifmol
from miniworld.utils.structure.distance import extract_residue_com

K, SEED = 8, 1
d = json.load(open("/home/psk6950/MiniWorld-scratch/fb-v240/crop_abag.json"))
items = d["items"]
for it in items:
    it["lossy"] = any(c["contacts_kept"] < it["n_contacts"] for c in it["crops"])
pool_loss = sorted([i for i in items if i["lossy"] and i["n_res_total"] <= 4000], key=lambda x: x["n_res_total"])
pool_ok = sorted([i for i in items if not i["lossy"] and i["n_res_total"] <= 6000], key=lambda x: x["n_res_total"])
pick = [pool_loss[int(q * (len(pool_loss) - 1))] for q in (0.0, 0.34, 0.67, 1.0)] + [pool_ok[int(q * (len(pool_ok) - 1))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
seen, sel = set(), []
for it in pick:
    if it["item"] not in seen: seen.add(it["item"]); sel.append(it)
with initialize_config_dir("/home/psk6950/MiniWorld/configs/miniworld", version_base=None):
    cfg = Config.model_validate(compose(config_name="phase1a_distogram_medium_v240_local"))
ds = BioMolData(BioMolData.BioMolConfig(crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db,
                                        sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer))
pre = ds.preprocessor
rec_by_id = {}
sel_ids = {s["item"] for s in sel}
sel_pos = {}
tab = ds.items._t
print("selected", [(s["item"], s["n_res_total"], s["lossy"]) for s in sel], flush=True)
# find the catalog row of each selected item (item_id column)
col = tab.column("item_id").to_pylist() if hasattr(tab, "column") else None
rows = np.load("/home/psk6950/MiniWorld-scratch/fb-v240/trainabag_rows.npy")
need = set(sel_ids)
for r in rows:
    rec = ds.items[int(r)]
    if rec.item_id in need:
        rec_by_id[rec.item_id] = rec; need.discard(rec.item_id)
        if not need: break
out = []
for it in sel:
    rec = rec_by_id[it["item"]]
    cm = load_cifmol(db_path=rec.cif_db_path, pdb_id=rec.record_id, assembly_id=rec.assembly_id, model_id=rec.model_id, alt_id=rec.alt_id)
    c1, c2 = (str(c) for c in rec.chain_ids)
    chains = [str(c) for c in cm.chains.chain_id.value]
    def res_of(c): return np.unique(cm.index_table.chains_to_residues(np.where(cm.chains.chain_id == c)[0]))
    com = extract_residue_com(cm)
    r1, r2 = res_of(c1), res_of(c2)
    D = np.linalg.norm(com[r1][:, None, :] - com[r2][None, :, :], axis=-1)
    iface1, iface2 = set(r1[(D < 10).any(1)].tolist()), set(r2[(D < 10).any(0)].tolist())
    masks = np.zeros(len(com), dtype=np.int32); focus = []; kept_check = []
    for k in range(K):
        rng = np.random.default_rng(SEED * 1000 + k)
        res = pre.get_crop_indices(cm, [c1, c2], cfg.data.crop.max_tokens, cfg.data.crop.max_atoms, rng)
        ci, foc = res[0], res[4]
        masks[ci] |= (1 << k); focus.append([round(float(x), 1) for x in np.asarray(foc).reshape(-1)[:3]])
        k1, k2 = np.isin(r1, ci), np.isin(r2, ci)
        kept_check.append(int(((D < 10) & k1[:, None] & k2[None, :]).sum()))
    ok = kept_check == [c["contacts_kept"] for c in it["crops"]]
    keep_res = set(range(len(com)))  # every residue of every chain, so the part outside the crop can be drawn faded too
    a2r = cm.index_table.atom_to_res
    names = cm.atoms.id.value; xyz = cm.atoms.xyz.value; resn = cm.residues.chem_comp_id.value
    chain_of_res = {}
    for c in chains:
        for r in res_of(c): chain_of_res[int(r)] = c
    want = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}
    coords = {}
    for ai in range(len(names)):
        r = int(a2r[ai])
        if r in keep_res and names[ai] in want and np.isfinite(xyz[ai]).all():
            coords.setdefault(r, {})[want[names[ai]]] = [round(float(v), 1) for v in xyz[ai]]
    ch_index = {c: i for i, c in enumerate(chains)}
    residues = []
    first = {c: int(res_of(c)[0]) for c in chains if int(res_of(c)[0]) in keep_res or True}
    for r in sorted(coords):
        if 1 not in coords[r]: continue
        c = chain_of_res[r]
        role = 1 if r in iface1 or r in iface2 else 0
        residues.append([ch_index[c], r - first[c] + 1, str(resn[r]), int(masks[r]), role] + [coords[r].get(i) for i in range(5)])
    out.append(dict(id=it["item"].replace("pdb:", ""), pdb=it["pdb"], chains=chains, c1=c1, c2=c2, tags=it["tags"], total=it["n_res_total"], nc=it["n_contacts"],
                    kept=kept_check, replay_ok=ok, focus=focus, residues=residues))
    print(it["item"], "residues out", len(residues), "replay_ok", ok, flush=True)
json.dump(out, open("/home/psk6950/MiniWorld-scratch/fb-v240/crop_viz_all.json", "w"), separators=(",", ":"))
print("VIZ_DONE", len(out), flush=True)
