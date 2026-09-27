"""Short GPU check of v1.3: real trunk, optimizer updates, target-free sampling.

Run through Slurm. Does not create a W&B run or touch existing checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from miniworld.models.distogram_only.client import Client
from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel
from miniworld.training.engine_backend import configure_engine_backend, configure_fused_msa_train
from run_miniworld_distogram_train import _build_precompile_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/miniworld/distogram_diffusion_medium_v130_bioai.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    with initialize_config_dir(str(args.config.resolve().parent), version_base=None):
        raw = compose(config_name=args.config.stem)
    cfg = Client.Config.model_validate(OmegaConf.to_container(raw, resolve=True))
    configure_engine_backend(cfg.train.engine_backend)
    configure_fused_msa_train(cfg.train.fused_msa_train)
    torch.manual_seed(130)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model = MiniSWAModel(cfg.model).cuda().train()
    assert model.add_pair_recycle is None and model.distogram_head is None
    batch = _build_precompile_batch(
        device=torch.device("cuda"), msa_depth=8192,
        n_tokens=raw.data.crop.max_tokens, n_atoms=raw.data.crop.max_atoms,
        n_templates=4, num_res_class=cfg.model.shared.num_res_class,
    )
    # Include cross-chain pairs and padding in the actual model check.
    length = raw.data.crop.max_tokens
    batch.scheme.token_asym_id[:, length // 2:] = 1
    mapping = batch.scheme.atom_to_token_idx_map
    batch.structure.atom_is_rep[:, 0] = True
    batch.structure.atom_is_rep[:, 1:] = mapping[:, 1:] != mapping[:, :-1]
    batch.structure.token_mask[:, -8:] = False
    batch.structure.atom_mask &= mapping < length - 8
    batch.structure.atom_pos_mask &= batch.structure.atom_mask
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if args.compile:
        model.compile(dynamic=False)
    raw_model = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
    torch.manual_seed(130 + rank)
    if rank % 2:
        batch.template.mask.zero_()
    rows = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        loss, stats = model(
            msa=batch.msa, reference=batch.reference, scheme=batch.scheme,
            sequence=batch.sequence, structure=batch.structure, template=batch.template,
            interchain_weight=cfg.loss.distogram_interchain_weight,
        )
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        assert torch.isfinite(loss)
        grads = {n: p.grad for n, p in raw_model.named_parameters() if p.grad is not None}
        assert grads and all(torch.isfinite(g).all() for g in grads.values())
        # Zero-initialized decoder blocks upstream gradients only on update zero.
        if step:
            assert raw_model.diffusion.encoder.weight.grad.abs().sum() > 0
            assert any(g.abs().sum() > 0 for n, g in grads.items() if n.startswith("pairformer_blocks."))
        optimizer.step()
        row = {"step": step, "loss": loss.item(), "fwd_bwd_ms": start.elapsed_time(end),
               "gradient_tensors": len(grads), **{k: v.item() for k, v in stats.items()}}
        rows.append(row)
        print(json.dumps(row), flush=True)
    model.eval()
    # Sampling must not use ground truth coordinates / representative labels.
    batch.structure.atom_pos.fill_(float("nan"))
    batch.structure.atom_pos_mask.zero_()
    batch.structure.atom_is_rep = None
    bins = raw_model.sample_distogram(
        batch.msa, batch.reference, batch.scheme, batch.sequence,
        batch.structure, batch.template, num_steps=2,
    )
    assert bins.shape == (1, length, length)
    assert torch.equal(bins, bins.transpose(-1, -2))
    assert bins.min() >= 0 and bins.max() < 96
    result = {"passed": True, "compile": args.compile, "rows": rows,
              "peak_memory_bytes": torch.cuda.max_memory_allocated(), "sampling_passed": True,
              "torch": torch.__version__, "world_size": world, "rank": rank}
    output = args.output.with_name(f"{args.output.stem}-rank{rank}{args.output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
