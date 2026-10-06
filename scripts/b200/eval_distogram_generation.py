"""Distogram generation check: sample bins with the Heun solver and compare them with the target.

    python scripts/b200/eval_distogram_generation.py CKPT --weights ema --samples 48            # accuracy, all / interface pairs
    python scripts/b200/eval_distogram_generation.py CKPT --weights ema --samples 8 --seeds 8   # + diversity across seeds

The sampler gets no target coordinates (``sample_distogram``); the target bins are only read afterwards.
Bins are 0.25 A wide over 2.25-25.75 A (bin 0 "<2.25", last bin ">25.75"). A contact is a bin <= 23 (< 8 A).
"Interface" pairs are token pairs of two different chains (token_asym_id differs).
"""

import argparse
import itertools
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

CONTACT_BIN = 23


def bins_to_angstrom(b, nb):
    d = 2.25 + 0.25 * (b.float() - 0.5)
    return torch.where(b == nb - 1, torch.full_like(d, 26.0), d.clamp(min=2.0))


class Counts(dict):
    def add(self, key, value):
        self[key] = self.get(key, 0) + value


def accumulate(c, pred, target, mask, nb):
    """pairs, exact / near-pair stats, contact counts and far-pair recall on ``mask``."""
    d = (pred - target).abs()[mask]
    near = mask & (target < nb - 1)
    dn = (pred - target).abs()[near]
    tc, pc = mask & (target <= CONTACT_BIN), mask & (pred <= CONTACT_BIN)
    far = mask & (target == nb - 1)
    c.add("pairs", int(mask.sum()))
    c.add("exact", int((d == 0).sum()))
    c.add("near", int(near.sum()))
    c.add("near_within_1A", int((dn <= 4).sum()))
    c.add("near_abs_bin", float(dn.float().sum()))
    c.add("contact_true", int(tc.sum()))
    c.add("contact_pred", int(pc.sum()))
    c.add("contact_hit", int((tc & pc).sum()))
    c.add("far", int(far.sum()))
    c.add("far_hit", int((far & (pred == nb - 1)).sum()))


def summarize(c):
    g = lambda k: c.get(k, 0)  # noqa: E731
    prec = g("contact_hit") / max(g("contact_pred"), 1)
    rec = g("contact_hit") / max(g("contact_true"), 1)
    return {
        "pairs": g("pairs"), "contact_pairs_true": g("contact_true"), "contact_pairs_pred": g("contact_pred"),
        "contact8A_precision": round(prec, 4), "contact8A_recall": round(rec, 4),
        "contact8A_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "near_mean_abs_err_A": round(0.25 * g("near_abs_bin") / max(g("near"), 1), 3),
        "near_within_1A": round(g("near_within_1A") / max(g("near"), 1), 4),
        "far_pair_recall": round(g("far_hit") / max(g("far"), 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--config", default="distogram_diffusion_medium_v130_b200_L768")
    ap.add_argument("--weights", choices=["ema", "raw"], default="ema")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=1, help="generations per sample (>1 adds the diversity analysis)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-interface", type=int, default=0,
                    help="skip samples with fewer than this many true interface contacts (still counts toward --samples kept)")
    ap.add_argument("--max-load", type=int, default=400, help="samples to draw while looking for --samples kept ones")
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
    nb = cfg.model.shared.n_distogram_bins
    print(f"loaded {args.weights} weights of epoch {state['epoch']} step {state['global_step']}", flush=True)

    ds = BioMolData(BioMolData.BioMolConfig(
        crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db,
        sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer,
    ))
    train = cfg.train
    loader = ds.create_ddp_dataloader(
        world_size=1, rank=0, seed=args.seed, drop_last=True, batch_size=1, num_workers=4, prefetch_factor=2,
        num_samples_per_rank=args.max_load if args.require_interface else args.samples, persistent_workers=False, shuffle=True,
        bucket_msa_multiple=train.bucket_msa_multiple, bucket_token_multiple=train.bucket_token_multiple,
        bucket_atom_multiple=train.bucket_atom_multiple, bucket_template_multiple=1,
    )
    loader.sampler.set_epoch(0)
    ds.set_epoch(0)

    acc = {"all": Counts(), "interface": Counts(), "intra_chain": Counts()}
    div = {"all": Counts(), "interface": Counts()}
    n_chains, n_samples, with_interface = [], 0, 0
    per_sample = []  # best-of-K bookkeeping: per sample, per seed, per mask: (hit, pred, true)
    for i, batch in enumerate(loader):
        if n_samples >= args.samples:
            break
        batch = batch.to(device=device)
        target, pair_mask = atom_distogram_target(
            atom_pos=batch.structure.atom_pos, atom_pos_mask=batch.structure.atom_pos_mask,
            atom_to_token_idx_map=batch.scheme.atom_to_token_idx_map,
            num_bins=nb, token_num=batch.structure.token_mask.shape[1], rep_atom_mask=batch.structure.atom_is_rep,
        )
        tm = batch.structure.token_mask
        asym = batch.scheme.token_asym_id
        valid = pair_mask & tm[:, :, None] & tm[:, None, :]
        valid &= ~torch.eye(tm.shape[1], device=device, dtype=torch.bool)
        interface = valid & (asym[:, :, None] != asym[:, None, :])
        masks = {"all": valid, "interface": interface, "intra_chain": valid & ~interface}
        if args.require_interface and int((interface & (target <= CONTACT_BIN)).sum()) < args.require_interface:
            continue
        preds = []
        for s in range(args.seeds):
            torch.manual_seed(1000 * args.seed + 17 * i + s)
            with torch.no_grad():
                preds.append(model.sample_distogram(
                    batch.msa, batch.reference, batch.scheme, batch.sequence, batch.structure, batch.template,
                    num_steps=args.steps,
                ))
        for p in preds:
            for name, m in masks.items():
                accumulate(acc[name], p, target, m, nb)
        if args.seeds > 1:
            rec = {}
            for name, m in masks.items():
                tc = m & (target <= CONTACT_BIN)
                pcs = [m & (p <= CONTACT_BIN) for p in preds]
                vote = torch.stack(pcs).float().mean(0) >= 0.5
                rec[name] = {
                    "true": int(tc.sum()),
                    "seeds": [(int((tc & pc).sum()), int(pc.sum())) for pc in pcs],
                    "consensus": (int((tc & vote).sum()), int(vote.sum())),
                }
            per_sample.append(rec)
        n_samples += 1
        n_chains.append(int(asym[tm].unique().numel()))
        with_interface += int(interface.any())
        # diversity: how much do generations of ONE sample differ from each other
        if args.seeds > 1:
            a_all = [bins_to_angstrom(p, nb) for p in preds]
            for name in ("all", "interface"):
                m = masks[name]
                if not m.any():
                    continue
                for p, q, ap_, aq in ((preds[x], preds[y], a_all[x], a_all[y])
                                      for x, y in itertools.combinations(range(args.seeds), 2)):
                    both_near = m & (p < nb - 1) & (q < nb - 1)
                    pc, qc = m & (p <= CONTACT_BIN), m & (q <= CONTACT_BIN)
                    div[name].add("pairs", int(m.sum()))
                    div[name].add("bin_differs", int(((p != q) & m).sum()))
                    div[name].add("both_near", int(both_near.sum()))
                    div[name].add("abs_diff_A_near", float((ap_ - aq).abs()[both_near].sum()))
                    div[name].add("contact_inter", int((pc & qc).sum()))
                    div[name].add("contact_union", int((pc | qc).sum()))
        print(f"sample {i}: tokens={int(tm.sum())} chains={n_chains[-1]} interface_pairs={int(interface.sum())} "
              f"true_interface_contacts={int((interface & (target <= CONTACT_BIN)).sum())}", flush=True)

    out = {"weights": args.weights, "epoch": state["epoch"],
           "steps": args.steps or cfg.model.trunk.diffusion.num_sampling_steps,
           "samples": n_samples, "samples_with_interface": with_interface, "generations_per_sample": args.seeds,
           "accuracy": {k: summarize(v) for k, v in acc.items()}}
    if args.seeds > 1:
        out["diversity_across_seeds_of_one_sample"] = {
            k: {
                "pairs_compared": v.get("pairs", 0),
                "fraction_of_pairs_with_different_bin": round(v.get("bin_differs", 0) / max(v.get("pairs", 1), 1), 4),
                "mean_abs_distance_diff_A_on_near_pairs": round(
                    v.get("abs_diff_A_near", 0) / max(v.get("both_near", 1), 1), 3),
                "contact_set_jaccard_between_seeds": round(v.get("contact_inter", 0) / max(v.get("contact_union", 1), 1), 4),
            } for k, v in div.items()
        }
    if args.seeds > 1 and per_sample:
        def f1(hit, pred, true):
            p, r = hit / max(pred, 1), hit / max(true, 1)
            return 2 * p * r / max(p + r, 1e-9)

        best = {}
        for name in ("all", "interface", "intra_chain"):
            rows = [r[name] for r in per_sample if r[name]["true"] > 0]
            if not rows:
                continue
            curve = {}
            for k in [x for x in (1, 2, 4, 8, 16, 32) if x <= args.seeds]:
                singles = [sum(f1(h, p, r_["true"]) for h, p in r_["seeds"][:k]) / k for r_ in rows]
                bests = [max(f1(h, p, r_["true"]) for h, p in r_["seeds"][:k]) for r_ in rows]
                curve[f"K={k}"] = {"mean_single_seed_f1": round(sum(singles) / len(rows), 4),
                                   "best_of_K_f1": round(sum(bests) / len(rows), 4)}
            cons = [f1(*r_["consensus"], r_["true"]) for r_ in rows]
            best[name] = {"samples_with_true_contacts": len(rows), "curve": curve,
                          f"majority_vote_over_{args.seeds}_seeds_f1": round(sum(cons) / len(rows), 4)}
        out["best_of_K_per_sample_f1"] = best
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
