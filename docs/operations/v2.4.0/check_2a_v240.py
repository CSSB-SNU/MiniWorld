"""CPU check: the v2.4.0 2a config composes, its trunk matches the seed tensor-for-tensor, and the diffusion head is new."""
import sys
from pathlib import Path
import torch
sys.path[:0] = ["/home/psk6950/MiniWorld/scripts", "/home/psk6950/MiniWorld/src", "/home/psk6950/MiniWorld/libs/team-gm/src"]
from hydra import compose, initialize_config_dir
from run_miniworld_diffusion_train import Config
from miniworld.models.diffusion import Model

SEED = sys.argv[1]
with initialize_config_dir("/home/psk6950/MiniWorld/configs/miniworld", version_base=None):
    cfg = Config.model_validate(compose(config_name="phase2a_diffusion_medium_v240"))
print("num_epoch", cfg.train.num_epoch, "| crop", cfg.data.crop.max_tokens, cfg.data.crop.max_atoms, "| dit", cfg.model.diffusion.token_dit_kind, cfg.model.diffusion.dtype,
      "| pf blocks", cfg.model.trunk.pairformer.n_block, "| atom encoder", cfg.model.atom_swa.input_features, "| cuda_graph", cfg.train.cuda_graph, "| precision", cfg.train.precision,
      "| grad_accum", cfg.train.grad_accum_steps, "| sampler ab/rna", cfg.data.sampler.antibody_protein, cfg.data.sampler.protein_rna, flush=True)
print("catalog", cfg.data.train_db.catalog_cache_path, "| antibody list", cfg.data.train_db.antibody_seq_id_path, flush=True)
print("crop flags", cfg.data.crop.ab_ag_interface_only, cfg.data.crop.prefer_nonprotein_focus, "| msa", cfg.data.msa.missing_policy, cfg.data.msa.pairing_mode, flush=True)
model = Model(cfg.model)
msd = model.state_dict()
seed = torch.load(SEED, map_location="cpu", weights_only=False)["model_state_dict"]
bad = [k for k, v in seed.items() if k not in msd or msd[k].shape != v.shape]
new = [k for k in msd if k not in seed]
print(f"seed keys {len(seed)} | mismatched/absent in model {len(bad)} {bad[:3]} | model-only keys {len(new)}", flush=True)
print("model-only top modules:", sorted({k.split('.')[0] for k in new}), flush=True)
assert not bad
print("CHECK2A_DONE", flush=True)
