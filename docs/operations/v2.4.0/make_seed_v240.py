"""Phase-2a trunk seed for v2.4.0: the online EMA weights of a phase-1a checkpoint, weights only, epoch 0 (the same layout as v1.2.0's
make_seed.py). usage: make_seed_v240.py <src checkpoint> <expected epoch> <out file>"""
import sys
from pathlib import Path
import torch

SRC, EP, OUT = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
sd = torch.load(SRC, map_location="cpu", weights_only=False)
assert sd["epoch"] == EP, (sd["epoch"], EP)
live, ema = sd["model_state_dict"], sd["ema_state_dict"]
missing = [k for k, v in live.items() if v.is_floating_point() and k not in ema]
extra = [k for k in ema if k not in live]
print(f"live {len(live)} | ema {len(ema)} | float live keys without EMA {len(missing)} | EMA keys not in model {len(extra)}", flush=True)
assert not extra, extra[:3]
seed, rel = {}, []
for k, v in live.items():
    if k in ema:
        e = ema[k].float()
        rel.append(((e - v.float()).norm() / v.float().norm().clamp_min(1e-12)).item())
        seed[k] = e.to(v.dtype)
    else:
        seed[k] = v.clone()
rel.sort()
print(f"EMA vs live relative L2 per tensor: median {rel[len(rel)//2]:.3e}  max {rel[-1]:.3e}", flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({"config": sd["config"], "model_state_dict": seed, "epoch": 0,
            "ema": {"source": str(SRC), "source_epoch": EP, "decay": "train.ema_decay"}, "miniworld_version": "2.4.0"}, OUT)
print(f"saved {OUT} ({OUT.stat().st_size / 1024**2:.0f} MiB)", flush=True)
print("SEED_DONE", flush=True)
