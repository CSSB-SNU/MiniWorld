"""Build the phase-2a trunk initialisation: a uniform average of the last N phase-1b
checkpoints (SWA).

Phase 2a starts from SWA_last12 (epochs 845-900) rather than raw epoch-900: it is 3.45%
better on held-out distogram loss, and a full LCSC coefficient search could not beat it.
See docs/pipeline.md.

Averaging is per parameter in fp32, cast back to each parameter's ORIGINAL dtype (the trunk
is mixed bf16/fp32). Non-float entries (integer buffers) must be identical across the
checkpoints and are copied through. Optimizer/EMA state is intentionally dropped — phase 2a
loads this as a weights-only seed via ``--ckpt`` and freezes the trunk.

Usage:
    python scripts/make_swa_checkpoint.py \
        --run-root logs/autoscale/large_H100_full_d16k \
        --last 12 \
        --out runs/v1.0.0/phase1b/swa_last12/epoch=0900_swa12.pt
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import torch


def discover(run_root: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for p in glob.glob(f"{run_root}/*/*/checkpoints/epoch=*.pt"):
        m = re.search(r"epoch=(\d+)\.pt$", p)
        if m is None:
            continue
        ep = int(m.group(1))
        if ep in out:
            msg = f"duplicate checkpoint for epoch {ep}: {out[ep]} and {p}"
            raise RuntimeError(msg)
        out[ep] = p
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="logs/autoscale/large_H100_full_d16k")
    ap.add_argument("--last", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpts = discover(args.run_root)
    if not ckpts:
        msg = f"no checkpoints under {args.run_root}"
        raise SystemExit(msg)
    epochs = sorted(ckpts)[-args.last :]
    print(f"averaging {len(epochs)} checkpoints: {epochs[0]}..{epochs[-1]} {epochs}")

    acc: dict[str, torch.Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    nonfloat: dict[str, torch.Tensor] = {}
    cfg = None
    for i, ep in enumerate(epochs):
        sd = torch.load(ckpts[ep], map_location="cpu", weights_only=False)
        if ep == epochs[-1]:
            cfg = sd["config"]
        msd = sd["model_state_dict"]
        if i == 0:
            for k, v in msd.items():
                if v.is_floating_point():
                    acc[k] = v.to(torch.float32).clone()
                    dtypes[k] = v.dtype
                else:
                    nonfloat[k] = v.clone()
        else:
            if set(msd) != set(acc) | set(nonfloat):
                msg = f"key mismatch at epoch {ep}"
                raise RuntimeError(msg)
            for k, v in msd.items():
                if k in acc:
                    acc[k] += v.to(torch.float32)
                elif not torch.equal(v, nonfloat[k]):
                    msg = f"non-float entry {k} differs at epoch {ep}"
                    raise RuntimeError(msg)
        del sd, msd
        print(f"  [{i + 1}/{len(epochs)}] epoch {ep}")

    n = float(len(epochs))
    merged = {k: (v / n).to(dtypes[k]) for k, v in acc.items()}
    merged.update(nonfloat)

    ref = torch.load(ckpts[epochs[-1]], map_location="cpu", weights_only=False)
    delta = max(
        (merged[k].to(torch.float32) - ref["model_state_dict"][k].to(torch.float32))
        .abs()
        .max()
        .item()
        for k in acc
    )
    print(f"max |SWA - epoch{epochs[-1]}| = {delta:.4g}  ({len(acc)} float + {len(nonfloat)} other tensors)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg,
            "model_state_dict": merged,
            # epoch=0 ON PURPOSE. The trainer seeds its counter from this field
            # (diffusion/client.py: self._epoch = state_dict.get("epoch", 0)) and loops
            # `while client.epoch < num_epoch`. Carrying 900 over would make phase 2a's
            # `num_epoch: 50` a no-op. Phase 2 runs its own counter from 0; provenance is
            # kept in `swa.source_epoch`.
            "epoch": 0,
            "swa": {
                "epochs": epochs,
                "kind": f"uniform_last{len(epochs)}",
                "source_epoch": epochs[-1],
            },
            "miniworld_version": "1.0.0",
        },
        out,
    )
    print(f"saved {out} ({out.stat().st_size / 1024**2:.0f} MiB)")


if __name__ == "__main__":
    main()
