"""Size of one real phase-2a training batch, by group (CPU only)."""
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_miniworld_diffusion_train as R
from miniworld.configs import TemplateConfig
from miniworld.data.dataloader.dataloader import BioMolData


def nbytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if is_dataclass(x):
        return sum(nbytes(getattr(x, f.name)) for f in fields(x))
    return 0


def main():
    root = Path(__file__).resolve().parents[2] / "configs/miniworld"
    with initialize_config_dir(str(root), version_base=None):
        cfg = R.Config.model_validate(compose(config_name="phase2a_diffusion_medium_v120_b200"))
    ds = BioMolData(BioMolData.BioMolConfig(
        crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer))
    t = cfg.train
    dl = ds.create_ddp_dataloader(
        world_size=1, rank=0, seed=0, drop_last=True, batch_size=1, num_workers=2, prefetch_factor=2,
        num_samples_per_rank=3, persistent_workers=False, shuffle=True,
        bucket_msa_multiple=t.bucket_msa_multiple, bucket_token_multiple=t.bucket_token_multiple,
        bucket_atom_multiple=t.bucket_atom_multiple, bucket_template_multiple=TemplateConfig().n_templates)
    dl.sampler.set_epoch(0)
    ds.set_epoch(0)
    for i, b in enumerate(dl):
        groups = {f.name: nbytes(getattr(b, f.name)) for f in fields(b)}
        total = sum(groups.values())
        big = sorted(groups.items(), key=lambda kv: -kv[1])[:4]
        print(f"batch {i}: {total / 2**20:.0f} MiB  " + "  ".join(f"{k}={v / 2**20:.0f}" for k, v in big), flush=True)
        if i == 0:  # biggest tensors of the biggest group
            g = getattr(b, big[0][0])
            ts = sorted(((n, getattr(g, n).numel() * getattr(g, n).element_size(), tuple(getattr(g, n).shape), getattr(g, n).dtype)
                         for n in vars(g) if isinstance(getattr(g, n), torch.Tensor)), key=lambda x: -x[1])[:5]
            for n, by, sh, dt in ts:
                print(f"   {big[0][0]}.{n}: {by / 2**20:.0f} MiB {sh} {dt}")
    print("static shape check: tokens", cfg.data.crop.max_tokens, "atoms", cfg.data.crop.max_atoms, "msa", cfg.data.msa.max_msa_depth)


if __name__ == "__main__":
    main()
