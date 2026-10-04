"""Run Protenix's inference featurizer (its own venv) on the copied examples; save the feature dicts for the engine-env harness."""
import os, sys
sys.path.insert(0, os.environ["PROTENIX_SRC"])        # run in Protenix's own environment
os.environ.setdefault("LAYERNORM_TYPE", "torch")
import torch
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from protenix.config.config import parse_configs
from protenix.data.inference.infer_dataloader import InferenceDataset
out = os.environ.get("PFX_FEATS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "feats"))
os.makedirs(out, exist_ok=True)
cfg = {**configs_base, **{"data": data_configs}, **inference_configs}
cfg = parse_configs(cfg, arg_str=f"--input_json_path {os.environ.get('PFX_INPUT_JSON', os.path.join(os.environ['PROTENIX_SRC'], 'examples/example.json'))} --dump_dir {out}/dump --model_name protenix_base_default_v1.0.0", fill_required_with_null=True)
ds = InferenceDataset(cfg) if "configs" in InferenceDataset.__init__.__code__.co_varnames else InferenceDataset(cfg)
for i in range(len(ds)):
    try:
        data, atom_array, err = ds[i]
    except Exception as e:
        print("item", i, "failed", type(e).__name__, str(e)[:200]); continue
    name = data["sample_name"]
    f = data["input_feature_dict"]
    print(f"{name}: N_token {data['N_token'].item()} N_atom {data['N_atom'].item()} N_msa {data['N_msa'].item()} keys {len(f)}", flush=True)
    torch.save({k: v for k, v in data.items() if k != "atom_array"}, f"{out}/{name}.pt")
