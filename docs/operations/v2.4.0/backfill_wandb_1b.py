"""Backfill a phase-1b train.log into a NEW W&B run (the live run shares the phase-1a W&B id, so every step < 66625 was dropped).
usage: backfill_wandb_1b.py <train.log> <run name>. Last line per step wins (a resume can re-log a step). Re-runnable at any time."""
import re, sys
from collections import defaultdict
import wandb

LOG, NAME = sys.argv[1], sys.argv[2]
pat = re.compile(r"Step\s+(\d+) \(Epoch\s+(\d+)\) \| train/distogram_loss=([0-9.]+) step_time_ms=([0-9.]+)")
rows = {}
for line in open(LOG):
    m = pat.search(line)
    if m:
        rows[int(m[1])] = (int(m[2]), float(m[3]), float(m[4]))
steps = sorted(rows)
print(f"{len(steps)} steps {steps[0]}..{steps[-1]}", flush=True)
run = wandb.init(project="MiniWorld", entity="team_gm", name=NAME, config={"source": LOG, "note": "backfill of phase 1b, resumed from phase-1a epoch 600"})
wandb.define_metric("global_step")
wandb.define_metric("epoch")
wandb.define_metric("train/distogram_loss_step", step_metric="global_step")
wandb.define_metric("train/distogram_loss_epoch", step_metric="epoch")
acc = defaultdict(list)
for s in steps:
    ep, loss, ms = rows[s]
    acc[ep].append(loss)
    wandb.log({"train/distogram_loss_step": loss, "train/step_time_ms": ms, "global_step": s}, step=s)
    if len(acc[ep]) == 100:  # the epoch is complete
        wandb.log({"train/distogram_loss_epoch": sum(acc[ep]) / 100, "epoch": ep}, step=s)
print(f"url {run.url}", flush=True)
run.finish()
print("BACKFILL_DONE", flush=True)
