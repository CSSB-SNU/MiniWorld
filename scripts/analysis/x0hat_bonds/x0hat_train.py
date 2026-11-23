"""Training-setup x0-hat: does the x0-hat the model predicts at TRAINING noise levels keep its bonds?

Uses what the 2a training run uses: the phase-2a config, its training DB / sampler / crop (BioMolData, same bucketing), the checkpoint's RAW weights (training weights,
not EMA), the diffuser's own noise sampling (``diffuser.sample`` with num_augment augments, each with its own sigma ~ the training log-normal), bf16-mixed autocast, and the same
forward as ``Client.loss_fn``:  x_pred = diffuser.get_x_pred(x_input, model(x_input), sigma).  Bond lengths of x_pred are scored against the CCD ideal lengths
(x0hat_common) and against the same bonds in the ground truth x0, binned by sigma. The GT row is the floor (crystal vs ideal).
usage: x0hat_train.py --ckpt CKPT --config CFG --items N --augment A --out JSON
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src"), str(Path(__file__).resolve().parent)]
import run_miniworld_diffusion_inference as inf  # noqa: E402
from x0hat_common import bond_metrics, ideal_bonds  # noqa: E402
from miniworld.training.precision import model_autocast  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=Path, required=True)
ap.add_argument("--config", type=Path, default=REPO / "configs/miniworld/phase2a_diffusion_medium_v240.yaml")
ap.add_argument("--items", type=int, default=32)
ap.add_argument("--augment", type=int, default=48)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ema", action="store_true", help="EMA weights instead of the raw training weights")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--overrides", nargs="*", default=[], help="hydra overrides, e.g. train.precision=native model.diffusion.dtype=fp32")
ap.add_argument("--quant-input", action="store_true", help="round ONLY the network input x_t to bf16 (the skip path stays exact)")
ap.add_argument("--highest", action="store_true", help="exact fp32 matmuls (torch.set_float32_matmul_precision('highest'))")
args = ap.parse_args()
if args.highest:
    torch.set_float32_matmul_precision("highest")

with inf.initialize_config_dir(str(args.config.parent.absolute()), version_base=None):
    cfg = inf.compose(config_name=args.config.name, overrides=list(args.overrides))
cfg = inf.Config.model_validate(cfg)
fabric = inf._fabric_from_torchrun()
fabric.launch()
fabric.seed_everything(args.seed)
cfg.train.use_ema = bool(args.ema)
cfg.train.param_policy.enabled = False
client = inf.Client(inf.Client.Config(train=cfg.train, model=cfg.model, diffuser=cfg.diffuser, loss=cfg.loss))
client.setup(fabric=fabric)
client.load_state_dict(torch.load(args.ckpt, map_location="cpu"), model_only=True, strict=False)
client.model.eval()
print(f"loaded {args.ckpt} (epoch={client.epoch}) weights={'EMA' if args.ema else 'RAW'} precision={cfg.train.precision}", flush=True)

data_config = inf.BioMolData.BioMolConfig(
    crop_config=cfg.data.crop, msa_config=cfg.data.msa, DB_config=cfg.data.train_db, sampler_config=cfg.data.sampler, tokenizer_config=cfg.data.tokenizer
)
dataset = inf.BioMolData(data_config)
dataloader = dataset.create_ddp_dataloader(
    world_size=1, rank=0, seed=args.seed, drop_last=False, batch_size=1, num_workers=0, num_samples_per_rank=args.items, shuffle=False,
    bucket_msa_multiple=cfg.train.bucket_msa_multiple, bucket_token_multiple=cfg.train.bucket_token_multiple, bucket_atom_multiple=cfg.train.bucket_atom_multiple,
)
dataset.set_epoch(0)

EDGES = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 1e9]
rows = {k: [] for k in range(len(EDGES) - 1)}  # per sigma bin: list of per-(item,augment) metric dicts
floor = []  # the GT itself vs ideal lengths, per item
t0 = time.time()
for it, raw in enumerate(dataloader):
    if it >= args.items:
        break
    batch = raw.to(device=client.device)
    ref = batch.reference
    pos_mask = batch.structure.atom_pos_mask[0] > 0
    valid = pos_mask & (ref.mask[0] > 0)
    uid = ref.space_uid[0].long()
    i, j, ideal = ideal_bonds(ref.pos[0], uid, valid)
    if len(i) == 0:
        continue
    with torch.no_grad():
        x0, x_input, x_mask, t_emb, sigma = client.diffuser.sample(batch.structure.atom_pos, num_augment=args.augment, mask=batch.structure.atom_pos_mask)
        x_net = x_input.to(torch.bfloat16).to(x_input.dtype) if args.quant_input else x_input  # network input only; get_x_pred below keeps the exact x_input
        with model_autocast(client.model):
            x_update = client.model.forward(
                msa=batch.msa, template=batch.template, reference=batch.reference, scheme=batch.scheme, sequence=batch.sequence,
                structure=batch.structure, x_t=x_net, x_mask=x_mask, t_emb=t_emb,
            )
        x_pred = client.diffuser.get_x_pred(x_input=x_input, x_update=x_update.float(), sigma=sigma)  # [A, B=1, L, 3]
        X0 = x0.reshape(x0.shape[0], -1, 3)  # [A, L, 3] whether the diffuser keeps a B axis or not
        XP = x_pred.reshape(x_pred.shape[0], -1, 3)
        floor.append(bond_metrics(X0[0], i, j, ideal, uid, valid))
        sig = sigma.reshape(sigma.shape[0], -1)[:, 0].float().cpu().numpy()
        d_gt = (X0[0][i] - X0[0][j]).norm(dim=-1)
        # inter-residue covalent bonds (peptide C-N etc.): atom pairs of DIFFERENT residues that are < 1.6 A apart in the ground truth
        dgt_all = torch.cdist(X0[0], X0[0])
        inter = (uid[:, None] != uid[None, :]) & valid[:, None] & valid[None, :] & (dgt_all < 1.6) & torch.triu(torch.ones_like(dgt_all, dtype=torch.bool), diagonal=1)
        pi, pj = inter.nonzero(as_tuple=True)
        pdg = dgt_all[pi, pj]
        for a in range(XP.shape[0]):
            m = bond_metrics(XP[a], i, j, ideal, uid, valid)
            dd = (XP[a][i] - XP[a][j]).norm(dim=-1)
            m["mean_abs_dev_vs_gt"] = float((dd - d_gt).abs().mean())
            if len(pi):
                pd = ((XP[a][pi] - XP[a][pj]).norm(dim=-1) - pdg).abs()
                m["pep_mean_dev"] = float(pd.mean())
                m["pep_gt_0.1"] = float((pd > 0.1).float().mean())
            m["sigma"] = float(sig[a])
            b = int(np.searchsorted(EDGES, sig[a], side="right") - 1)
            rows[b].append(m)
    print(f"item {it} {batch.name[0]} atoms={int(valid.sum())} bonds={len(i)}  ({time.time() - t0:.0f}s)", flush=True)

def mean(ms, k):
    v = [m[k] for m in ms if k in m and m[k] == m[k]]
    return float(np.mean(v)) if v else float("nan")

summary = {"ckpt": str(args.ckpt), "weights": "EMA" if args.ema else "RAW", "items": len(floor), "augment": args.augment, "bins": []}
print(f"\n=== x0-hat bond geometry at TRAINING noise levels ({len(floor)} items x {args.augment} noise draws, {summary['weights']} weights, {cfg.train.precision}) ===")
print(f"{'sigma bin':>14}{'n':>6}{'mean|d| vs ideal':>18}{'>0.1A':>8}{'>0.3A':>8}{'|d| vs GT':>11}{'clash':>8}{'peptide|d|':>12}{'pep>0.1A':>10}")
fm = {k: mean(floor, k) for k in ("mean_abs_dev", "frac_gt_0.1", "frac_gt_0.3", "n_clash")}
print(f"{'GT (floor)':>14}{len(floor):6d}{fm['mean_abs_dev']:18.3f}{fm['frac_gt_0.1']:8.2f}{fm['frac_gt_0.3']:8.2f}{0.0:11.3f}{fm['n_clash']:8.1f}")
for b in range(len(EDGES) - 1):
    ms = rows[b]
    if not ms:
        continue
    r = {k: mean(ms, k) for k in ("mean_abs_dev", "frac_gt_0.1", "frac_gt_0.3", "mean_abs_dev_vs_gt", "n_clash", "pep_mean_dev", "pep_gt_0.1")}
    label = f"{EDGES[b]:g}-{EDGES[b + 1]:g}" if EDGES[b + 1] < 1e8 else f">{EDGES[b]:g}"
    print(f"{label:>14}{len(ms):6d}{r['mean_abs_dev']:18.3f}{r['frac_gt_0.1']:8.2f}{r['frac_gt_0.3']:8.2f}{r['mean_abs_dev_vs_gt']:11.3f}{r['n_clash']:8.1f}{r['pep_mean_dev']:12.3f}{r['pep_gt_0.1']:10.2f}")
    summary["bins"].append({"sigma_lo": EDGES[b], "sigma_hi": EDGES[b + 1], "n": len(ms), **r})
summary["gt_floor"] = fm
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(summary, indent=1))
print("X0HAT_TRAIN_DONE", flush=True)
