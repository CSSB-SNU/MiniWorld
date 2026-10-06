"""Short check of real distogram generation: sample bins with the Heun solver and compare them with the target.

    python scripts/b200/eval_distogram_generation.py CKPT --weights ema|raw --samples 8 [--steps 24]

The sampler gets no target coordinates (``sample_distogram``); the target bins are only read afterwards for the metrics.
Metrics are over valid token pairs (both tokens real, off-diagonal, target distance known). Bins are 0.25 A wide over
2.25-25.75 A; bin 0 is "<2.25" and the last bin ">25.75" (most pairs of a large crop), so the accuracy is also given
on the near pairs alone (target <= 25.75 A) and as 8 A contacts (bin <= 23) and far-pair recall.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_miniworld_distogram_train import Config

from miniworld.data.dataloader.dataloader import BioMolData
from miniworld.loss.auxiliary import atom_distogram_target
from miniworld.models.distogram_only.model_mini_swa import MiniSWAModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--config", default="distogram_diffusion_medium_v130_b200_L768")
    ap.add_argument("--weights", choices=["ema", "raw"], default="ema")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(str(root / "configs/miniworld"), version_base=None):
        cfg = Config.model_validate(compose(config_name=args.config))

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = MiniSWAModel(cfg.model)
    weights = state["ema_state_dict"] if args.weights == "ema" else state["model_state_dict"]
    model.load_state_dict({**state["model_state_dict"], **weights}, strict=True)
    device = torch.device("cuda")
    model = model.to(device).eval()
    print(f"loaded {args.weights} weights of epoch {state['epoch']} step {state['global_step']}", flush=True)

    ds = BioMolData(BioMolData.BioMolConfig(
        crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer,
    ))
    train = cfg.train
    loader = ds.create_ddp_dataloader(
        world_size=1, rank=0, seed=args.seed, drop_last=True, batch_size=1, num_workers=4, prefetch_factor=2,
        num_samples_per_rank=args.samples, persistent_workers=False, shuffle=True,
        bucket_msa_multiple=train.bucket_msa_multiple, bucket_token_multiple=train.bucket_token_multiple,
        bucket_atom_multiple=train.bucket_atom_multiple, bucket_template_multiple=1,
    )
    loader.sampler.set_epoch(0)
    ds.set_epoch(0)

    nb = cfg.model.shared.n_distogram_bins
    tot = dict(pairs=0, exact=0, within1=0, abs_bin=0.0, mode=0, tokens=[],
               near=0, near_exact=0, near_within4=0, near_abs_bin=0.0,
               tc=0, pc=0, tp=0, far=0, far_hit=0)
    for i, batch in enumerate(loader):
        if i >= args.samples:
            break
        batch = batch.to(device=device)
        target, pair_mask = atom_distogram_target(
            atom_pos=batch.structure.atom_pos, atom_pos_mask=batch.structure.atom_pos_mask,
            atom_to_token_idx_map=batch.scheme.atom_to_token_idx_map,
            num_bins=cfg.model.shared.n_distogram_bins, token_num=batch.structure.token_mask.shape[1],
            rep_atom_mask=batch.structure.atom_is_rep,
        )
        tm = batch.structure.token_mask
        valid = pair_mask & tm[:, :, None] & tm[:, None, :]
        valid &= ~torch.eye(tm.shape[1], device=device, dtype=torch.bool)
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            pred = model.sample_distogram(
                batch.msa, batch.reference, batch.scheme, batch.sequence, batch.structure, batch.template,
                num_steps=args.steps,
            )
        d = (pred - target).abs()[valid]
        n = int(valid.sum())
        mode_bin = torch.bincount(target[valid]).argmax()
        tot["pairs"] += n
        tot["exact"] += int((d == 0).sum())
        tot["within1"] += int((d <= 1).sum())
        tot["abs_bin"] += float(d.float().sum())
        tot["mode"] += int((target[valid] == mode_bin).sum())
        tot["tokens"].append(int(tm.sum()))
        near = valid & (target < nb - 1)
        dn = (pred - target).abs()[near]
        tot["near"] += int(near.sum())
        tot["near_exact"] += int((dn == 0).sum())
        tot["near_within4"] += int((dn <= 4).sum())
        tot["near_abs_bin"] += float(dn.float().sum())
        tc, pc = valid & (target <= 23), valid & (pred <= 23)
        tot["tc"] += int(tc.sum())
        tot["pc"] += int(pc.sum())
        tot["tp"] += int((tc & pc).sum())
        far = valid & (target == nb - 1)
        tot["far"] += int(far.sum())
        tot["far_hit"] += int((far & (pred == nb - 1)).sum())
        print(f"sample {i}: tokens={int(tm.sum())} pairs={n} exact={float((d == 0).float().mean()):.3f} "
              f"within1={float((d <= 1).float().mean()):.3f} mean|dbin|={float(d.float().mean()):.2f}", flush=True)

    p = tot["pairs"]
    print(json.dumps({
        "weights": args.weights, "epoch": state["epoch"],
        "steps": args.steps or cfg.model.trunk.diffusion.num_sampling_steps,
        "samples": len(tot["tokens"]), "tokens": tot["tokens"], "pairs": p,
        "exact_acc": tot["exact"] / p, "within1_acc": tot["within1"] / p, "mean_abs_bin_err": tot["abs_bin"] / p,
        "baseline_most_frequent_bin_acc": tot["mode"] / p,
        "far_pair_fraction": tot["far"] / p,
        "far_pair_recall": tot["far_hit"] / max(tot["far"], 1),
        "near_pairs": tot["near"],
        "near_exact_acc": tot["near_exact"] / max(tot["near"], 1),
        "near_within_1A_acc": tot["near_within4"] / max(tot["near"], 1),
        "near_mean_abs_err_A": 0.25 * tot["near_abs_bin"] / max(tot["near"], 1),
        "contact8A_precision": tot["tp"] / max(tot["pc"], 1),
        "contact8A_recall": tot["tp"] / max(tot["tc"], 1),
        "contact8A_pairs": tot["tc"],
    }, indent=1))


if __name__ == "__main__":
    main()
