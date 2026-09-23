"""Supplement the engine matrix with MiniWorld's actual mixed-dtype training path.

Run after build all, against an isolated engine package. Three independent GPU
workers build L128/384/768. Only the parent merges caches; fresh processes then
verify cache hits. Repeated stacks are shortened to two blocks, preserving the
first/repeated layer layouts, widths, dtypes, dropout and template/MSA paths.
This is workload coverage, not a model accuracy or speed benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import torch
    from miniworld_engine import settings
    from miniworld_engine.autotune import cache, capture
    from miniworld_engine.autotune.configs import CONFIG_ROOT, use_config_dir

    tuning = args.mode == "tune"
    settings.configure(engine_backend="triton", run_autotune=tuning,
                       capture=tuning, fill_gaps=True, compile_jobs=12)
    use_config_dir(CONFIG_ROOT / "grid", require_all=False)
    if tuning:
        capture.install()
        capture.set_incremental(True)

    # Imports follow the explicit engine/config setup (kernel decorators read it).
    from omegaconf import OmegaConf
    from run_miniworld_distogram_train import Config, _build_precompile_batch
    from miniworld.models.distogram_only import MiniSWAModel
    from miniworld.loss.auxiliary import cal_atom_distogram_loss

    cfg = Config.model_validate(OmegaConf.to_container(
        OmegaConf.load(args.config), resolve=True))
    cfg.model.trunk.pairformer.n_block = 2
    cfg.model.trunk.msa_module.n_block = 2
    cfg.model.input_feat_embbeder.n_block = 2
    cfg.model.trunk.template_embedder.n_block = 2
    torch.manual_seed(17)
    torch.set_num_threads(4)
    model = MiniSWAModel(cfg.model).cuda().train()
    model._forced_n_recycle = 1
    backends = {m._backend.value for m in model.modules() if hasattr(m, "_backend")}
    assert backends == {"triton"}, backends
    batch = _build_precompile_batch(
        device=torch.device("cuda"), msa_depth=2048, n_tokens=args.length,
        n_atoms=4096, n_templates=4, num_res_class=cfg.model.shared.num_res_class)
    mapping = batch.scheme.atom_to_token_idx_map
    batch.structure.atom_is_rep = torch.cat(
        (torch.ones_like(mapping[:, :1], dtype=torch.bool),
         mapping[:, 1:] != mapping[:, :-1]), dim=1)
    batch.scheme.token_asym_id[:, args.length // 2:] = 1
    # Match the active Fabric job: explicit model dtypes, no blanket autocast.
    model.compile(dynamic=False, options={"triton.cudagraphs": False})
    cache.clear_cache_misses()
    report = {"mode": args.mode, "length": args.length, "msa_depth": 2048,
              "atoms": 4096, "templates": 4, "blocks_per_stack": 2,
              "recycles": 1, "compiled": True, "complete": False}
    start = time.monotonic()
    try:
        logits = model(msa=batch.msa, reference=batch.reference, scheme=batch.scheme,
                       sequence=batch.sequence, structure=batch.structure,
                       template=batch.template)
        loss = cfg.loss.distogram_loss * cal_atom_distogram_loss(
            logits, batch.structure.atom_pos, batch.structure.atom_pos_mask, mapping,
            rep_atom_mask=batch.structure.atom_is_rep,
            token_asym_id=batch.scheme.token_asym_id,
            interchain_weight=cfg.loss.distogram_interchain_weight)
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss), "nonfinite loss"
        assert all(torch.isfinite(p.grad).all() for p in model.parameters()
                   if p.grad is not None), "nonfinite gradients"
        misses = sorted(cache.cache_misses())
        report.update(loss=loss.item(), misses=misses,
                      peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        if tuning:
            assert not capture.record_errors(), capture.record_errors()
            capture.dump_shard(str(args.output.with_suffix(".shard.json")),
                               unit_complete=True)
        else:
            assert not misses, f"{len(misses)} unresolved runtime cache misses"
        report["complete"] = True
    finally:
        report["seconds"] = time.monotonic() - start
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if tuning:
            capture.shutdown_precompile()
        print(json.dumps(report), flush=True)


def orchestrate(args):
    import torch
    from miniworld_engine.autotune import capture

    assert torch.cuda.device_count() == 3, "requires three Slurm-allocated GPUs"
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2").split(",")
    lengths = (128, 384, 768)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    stages = {}
    for mode in ("tune", "verify"):
        processes = []
        for device, length in zip(devices, lengths):
            log = (out / f"{mode}-L{length}.log").open("w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device,
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--mode", mode,
                 "--length", str(length), "--config", str(args.config),
                 "--output", str(out / f"{mode}-L{length}.json")],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
            processes.append((length, proc))
        # All children are launched first; wait does not serialize GPU work.
        results = {str(length): proc.wait() for length, proc in processes}
        stages[mode] = results
        if mode == "tune":
            shards = [out / f"tune-L{length}.shard.json" for length in lengths
                      if results[str(length)] == 0]
            capture.merge_shards(shards, gpu="NVIDIA H100 80GB HBM3 (sm90)")
            stages["merge_rejected"] = list(capture._MERGE_SKIPPED)
        (out / "status.json").write_text(json.dumps(stages, indent=2) + "\n")
        if any(results.values()) or stages.get("merge_rejected"):
            return 1

    # Exact keys observed in the real medium job must also exist after merging.
    import miniworld_engine
    data = Path(miniworld_engine.__file__).parent / "autotune/data"
    missing = []
    if args.required_keys:
        required = json.loads(args.required_keys.read_text())["kernels"]
        for op, keys in required.items():
            path = data / op / "NVIDIA H100 80GB HBM3 (sm90).json"
            entries = json.loads(path.read_text()).get("entries", {}) if path.exists() else {}
            missing.extend([op, key] for key in keys if not entries.get(key))
    stages["required_keys_missing"] = missing
    stages["complete"] = not missing
    (out / "status.json").write_text(json.dumps(stages, indent=2) + "\n")
    return int(bool(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-keys", type=Path)
    parser.add_argument("--mode", choices=("tune", "verify"))
    parser.add_argument("--length", type=int)
    args = parser.parse_args()
    if args.mode:
        worker(args)
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
