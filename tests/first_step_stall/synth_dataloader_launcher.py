"""Wrapper that runs the same training entry point but swaps the real
BioMolData dataloader for an infinite synthetic-batch iterator, to
isolate whether the dataloader is the source of the training stall.
"""
from __future__ import annotations
import os, sys, torch
from typing import Iterable
sys.path.insert(0, "/home/snu_hwle/psk/MiniWorld/scripts")

# Import the training module (also gets _build_precompile_batch, cli, etc.)
import run_miniworld_distogram_train as trainer
from miniworld.data.dataloader import BioMolData


class InfiniteSyntheticLoader:
    """Yields synthetic batches on the target device forever."""
    def __init__(self, cfg, device):
        # match bucket max shapes from the config so no recompile fires
        self.msa_depth = int(cfg.train.bucket_msa_multiple)
        self.n_tokens = int(cfg.train.bucket_token_multiple)
        self.n_atoms = int(cfg.train.bucket_atom_multiple)
        self.n_templates = int(cfg.data.template.n_templates)
        self.num_res_class = int(getattr(cfg.data.tokenizer, "num_res_class", 32))
        self.device = device
        self.batch = None
        # give it a dummy sampler for compatibility
        class _Sampler:
            def set_epoch(self, _): pass
        self.sampler = _Sampler()

    def _batch(self):
        if self.batch is None:
            self.batch = trainer._build_precompile_batch(
                device=self.device,
                msa_depth=self.msa_depth,
                n_tokens=self.n_tokens,
                n_atoms=self.n_atoms,
                n_templates=self.n_templates,
                num_res_class=self.num_res_class,
            )
        return self.batch

    def __iter__(self) -> Iterable:
        while True:
            yield self._batch()

    def __len__(self) -> int:
        return 10**9


_orig_create = BioMolData.create_ddp_dataloader

def create_ddp_dataloader_synth(self, rank, *, world_size=1, **kwargs):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    # cfg is not passed in here — pluck bucket dims from kwargs
    class _Cfg:
        class train: pass
        class data:
            class tokenizer: pass
            class template: pass
    _Cfg.train.bucket_msa_multiple = kwargs.get("bucket_msa_multiple") or 2048
    _Cfg.train.bucket_token_multiple = kwargs.get("bucket_token_multiple") or 384
    _Cfg.train.bucket_atom_multiple = kwargs.get("bucket_atom_multiple") or 4096
    _Cfg.data.template.n_templates = 4
    _Cfg.data.tokenizer.num_res_class = 32
    print(f"[synth-loader] rank={rank}/{world_size} msa={_Cfg.train.bucket_msa_multiple} tokens={_Cfg.train.bucket_token_multiple} atoms={_Cfg.train.bucket_atom_multiple}", flush=True)
    return InfiniteSyntheticLoader(_Cfg, device)

BioMolData.create_ddp_dataloader = create_ddp_dataloader_synth

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    trainer.cli()
