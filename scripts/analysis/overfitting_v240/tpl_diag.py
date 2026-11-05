"""For every FoldBench eval target of the v2.4.0 sweep: is any template slot filled? If a chain has a template key in data.yaml but the chain ends up empty, say why:
key_missing / zero_mols (key present, template_mols empty) / too_short (all templates < 4 valid backbone residues) / len_mismatch (template length != chain tokens) / ok."""
import collections, json, sys, yaml, glob, os
sys.path[:0] = ["/home/psk6950/MiniWorld/src", "/home/psk6950/MiniWorld/scripts", "/home/psk6950/MiniWorld/libs/team-gm/src"]
import numpy as np
from miniworld.data.inference import InferenceSpec, build_inference_batch
from miniworld.data.io.load import load_raw_data, load_bytes
fb = "/home/psk6950/data/foldbench_v240eval"
targets = sorted(os.path.basename(f)[:-4] for f in glob.glob("/home/psk6950/MiniWorld-scratch/fb-v240/sweep/*.npz"))
summ = collections.Counter(); chain_reason = collections.Counter(); rows = []
for t in targets:
    spec = InferenceSpec.from_yaml(f"{fb}/inputs/{t}/data.yaml")
    ydoc = yaml.safe_load(open(f"{fb}/inputs/{t}/data.yaml"))
    if not spec.no_pairing_msa: spec = spec.model_copy(update={"no_pairing_msa": True})
    b = build_inference_batch(spec, max_msa_depth=64, missing_policy="query", seed=0, msa_subsample=True)
    asym = b.scheme.token_asym_id[0].cpu().numpy()
    any_slot = bool(b.template.mask.any())
    tcb = (b.template.cb_mask.any(-1)[0] if hasattr(b.template, "cb_mask") else None)
    reasons = {}
    for ci, key in (spec.template or {}).items():
        ntok = int((asym == int(ci)).sum())
        v = load_raw_data(key, spec.template_db)
        if v is None: r = "key_missing"
        else:
            mols = load_bytes(bytes(v)).get("template_mols", {})
            if len(mols) == 0: r = "zero_mols"
            else:
                lens = set()
                for k, it in mols.items():
                    try: lens.add(len(it["residues"]["chain_id"]) if isinstance(it.get("residues"), dict) and "chain_id" in it["residues"] else -1)
                    except Exception: lens.add(-1)
                r = f"has_mols(n={len(mols)})"
        reasons[ci] = (key, ntok, r)
        chain_reason[r.split("(")[0]] += 1
    summ["any_slot" if any_slot else "no_slot"] += 1
    rows.append(dict(t=t, any_slot=any_slot, ntok=int(b.token_length), n_tpl_chains=len(spec.template or {}), reasons=reasons))
    print(t, "any_slot" if any_slot else "NO_SLOT", {k: v[2] for k, v in reasons.items()}, flush=True)
print("SUMMARY targets:", dict(summ), flush=True)
print("SUMMARY template chains in data.yaml by DB status:", dict(chain_reason), flush=True)
nos = [r for r in rows if not r["any_slot"]]
print("NO_SLOT targets with >=1 template chain configured:", sum(1 for r in nos if r["n_tpl_chains"] > 0), "of", len(nos), flush=True)
json.dump(rows, open("/home/psk6950/MiniWorld-scratch/fb-v240/tpl_diag.json", "w"), indent=1, default=str)
