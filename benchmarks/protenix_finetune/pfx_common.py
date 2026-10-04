"""Build the Protenix v1 base model (fine-tuning settings of finetune_demo.sh) from a Protenix checkout, load the released weights.

Environment (README.md): PROTENIX_SRC -- a Protenix checkout (bytedance/Protenix 2475421) with protenix.patch applied;
PROTENIX_CKPT -- protenix_base_default_v1.0.0.pt; PROTENIX_EXTRA_SITE (optional) -- a directory with the packages Protenix
imports that the training env lacks (absl, ml_collections); PFX_FEATS -- the feature dicts (featurize.py; default ./feats)."""
import os, sys
from pathlib import Path
from collections.abc import Mapping


def _need(name):
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"set {name} (see benchmarks/protenix_finetune/README.md)")
    return v


if os.environ.get("PROTENIX_EXTRA_SITE"):
    sys.path.insert(0, os.environ["PROTENIX_EXTRA_SITE"])
sys.path.insert(0, _need("PROTENIX_SRC"))
os.environ.setdefault("LAYERNORM_TYPE", "fast_layernorm")
import torch
CKPT = _need("PROTENIX_CKPT")
FEATS = Path(os.environ.get("PFX_FEATS", Path(__file__).resolve().parent / "feats"))

def build_configs(extra=""):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs
    configs_base["triangle_attention"] = os.environ.get("TRIANGLE_ATTENTION", "cuequivariance")
    configs_base["triangle_multiplicative"] = os.environ.get("TRIANGLE_MULTIPLICATIVE", "cuequivariance")
    arg = ("--model_name protenix_base_default_v1.0.0 --dtype bf16 --diffusion_batch_size 48 --train_crop_size 384 "
           "--model.N_cycle 4 --sample_diffusion.N_step 20 --use_wandb false "
           + ("" if os.environ.get("PFX_CKPT") == "1" else "--blocks_per_ckpt None ")       # activation checkpointing off (default)
           + extra)
    base = {**configs_base, **{"data": data_configs}}
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d
    deep_update(base, model_configs["protenix_base_default_v1.0.0"])
    return parse_configs(configs=base, arg_str=arg, fill_required_with_null=True)

def build_model(device="cpu", load=True, extra=""):
    from protenix.model.protenix import Protenix
    cfg = build_configs(extra)
    m = Protenix(cfg)
    if load:
        sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)["model"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"loaded: missing {len(missing)} unexpected {len(unexpected)}", missing[:5], unexpected[:5])
    return cfg, m.to(device)
