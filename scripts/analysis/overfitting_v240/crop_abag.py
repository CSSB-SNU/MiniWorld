"""How ab-ag items are really cropped under the v2.4.0 training config: for N antibody_protein items drawn like the sampler, K crops each with
Preprocessor.get_crop_indices (the training crop), residue level. Records the contacts (residue-centre distance < 10 A, the crop code's own centre-of-mass) between the
item's two chains that survive each crop. Output: crop_abag.json."""
import json, sys, time, traceback
from pathlib import Path
import numpy as np
sys.path[:0] = ["/home/psk6950/MiniWorld/scripts", "/home/psk6950/MiniWorld/src", "/home/psk6950/MiniWorld/libs/team-gm/src"]
from hydra import compose, initialize_config_dir
from run_miniworld_distogram_train import Config
from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.data.io.load import load_cifmol
from miniworld.data.constants.mapping import molecule_tag
from miniworld.utils.structure.distance import extract_residue_com

N, K, SEED = int(sys.argv[1]), int(sys.argv[2]), 1
with initialize_config_dir("/home/psk6950/MiniWorld/configs/miniworld", version_base=None):
    cfg = Config.model_validate(compose(config_name="phase1a_distogram_medium_v240_local"))
print("CROP CONFIG:", cfg.data.crop.model_dump(), flush=True)
ds = BioMolData(BioMolData.BioMolConfig(crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db,
                                        sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer))
pre = ds.preprocessor
print("antibody list size:", len(pre.antibody_seq_ids), "| catalog rows", len(ds.items), flush=True)
sel = np.load("/home/psk6950/MiniWorld-scratch/fb-v240/trainabag_rows.npy")
tab = ds.items._t
w = tab.column("weight").to_numpy()[sel].astype(np.float64)
draw = np.random.default_rng(SEED).choice(sel, size=N * 3, replace=False, p=w / w.sum())
out, nfail = [], 0
t0 = time.time()
for idx in draw:
    if len(out) >= N: break
    rec = ds.items[int(idx)]
    try:
        cm = load_cifmol(db_path=rec.cif_db_path, pdb_id=rec.record_id, assembly_id=rec.assembly_id, model_id=rec.model_id, alt_id=rec.alt_id)
        c1, c2 = (str(c) for c in rec.chain_ids)
        chains = [str(c) for c in cm.chains.chain_id.value]
        if c1 not in chains or c2 not in chains: continue
        def res_of(c):
            r = np.unique(cm.index_table.chains_to_residues(np.where(cm.chains.chain_id == c)[0])); return r
        r1, r2 = res_of(c1), res_of(c2)
        com = extract_residue_com(cm)
        D = np.linalg.norm(com[r1][:, None, :] - com[r2][None, :, :], axis=-1)
        contact = D < 10.0
        if contact.sum() == 0: continue
        tags = [molecule_tag(cm.chains.select(chain_id=c).seq_id.value[0], pre.antibody_seq_ids) for c in (c1, c2)]
        item = dict(item=rec.item_id, pdb=rec.record_id, chains=[c1, c2], tags=tags, n_res_total=int(len(com)), n_chains=len(chains),
                    len=[int(len(r1)), int(len(r2))], n_contacts=int(contact.sum()),
                    contact_pairs=np.argwhere(contact).tolist(), crops=[])
        for k in range(K):
            rng = np.random.default_rng(SEED * 1000 + k)
            ci, cic, *_ = pre.get_crop_indices(cm, [c1, c2], cfg.data.crop.max_tokens, cfg.data.crop.max_atoms, rng)
            k1, k2 = np.isin(r1, ci), np.isin(r2, ci)
            kept = int((contact & k1[:, None] & k2[None, :]).sum())
            item["crops"].append(dict(n=int(len(ci)), kept1=np.flatnonzero(k1).tolist(), kept2=np.flatnonzero(k2).tolist(), contacts_kept=kept,
                                      other_chain_res=int(len(ci) - k1.sum() - k2.sum())))
        out.append(item)
        print(f"[{len(out)}/{N}] {rec.item_id} tags={tags} len={item['len']} total={item['n_res_total']} contacts={item['n_contacts']} "
              f"kept={[c['contacts_kept'] for c in item['crops']]} {time.time()-t0:.0f}s", flush=True)
    except Exception:
        nfail += 1
        if nfail <= 3: traceback.print_exc()
        print("FAILED", rec.record_id, flush=True)
json.dump(dict(crop_config=cfg.data.crop.model_dump(), antibody_list=len(pre.antibody_seq_ids), items=out), open("/home/psk6950/MiniWorld-scratch/fb-v240/crop_abag.json", "w"))
print("CROP_DONE", len(out), "failed", nfail, flush=True)
