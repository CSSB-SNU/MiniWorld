"""Phase-2 training speed on synthetic batches: the real trainer (DDP, optimizer, EMA, compile) with the dataloader
replaced by one dense on-device batch, so the number is the step time without data loading.

    torchrun --standalone --nproc_per_node=2 scripts/b200/bench_phase2_synthetic.py train \
        --config configs/miniworld/phase2a_diffusion_medium_v120_b200.yaml --ckpt SEED OVERRIDES...

Epoch time / (train_item / world / grad_accum_steps) is the optimizer-step time; the first epoch still compiles.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_miniworld_diffusion_train as R


class _Sampler:
    def set_epoch(self, epoch):
        pass


class _SyntheticLoader:
    sampler = _Sampler()

    def __init__(self, batch, n_batches):
        self.batch, self.n_batches = batch, n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.batch


class _SyntheticData:
    class BioMolConfig:  # the trainer builds BioMolData.BioMolConfig(...) before the dataset
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def __init__(self, config):
        self.config = config

    def set_epoch(self, epoch):
        pass

    def create_ddp_dataloader(self, **kw):
        crop, msa = self.config.crop_config, self.config.msa_config
        batch = R._build_precompile_batch(
            device=torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))),
            msa_depth=R._ceil_to_multiple(msa.max_msa_depth, kw["bucket_msa_multiple"]),
            n_tokens=R._ceil_to_multiple(crop.max_tokens, kw["bucket_token_multiple"]),
            n_atoms=R._ceil_to_multiple(crop.max_atoms, kw["bucket_atom_multiple"]),
            n_templates=kw["bucket_template_multiple"],
            num_res_class=32,  # model.shared.num_res_class of the medium configs
        )
        return _SyntheticLoader(batch, kw["num_samples_per_rank"])


R.BioMolData = _SyntheticData

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    R.cli()
