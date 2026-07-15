"""Minimal MiniPairformer-only test: is compile+autotune the stall culprit?

Runs N forward+backward passes on a synthetic pair tensor. Reports:
- first-step time (compile + autotune)
- steady-state avg (after warmup)

Args: --compile 0|1 --n_block N --token L --dim D --steps K --gpu ID
"""
from __future__ import annotations
import argparse, os, time, torch
from team_gm.modules import MiniPairformer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--n_block", type=int, default=48)
    p.add_argument("--token", type=int, default=384)
    p.add_argument("--d_pair", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16

    cfg = MiniPairformer.Config(d_pair=args.d_pair, n_block=args.n_block, p_drop=0.0)
    model = MiniPairformer(cfg).to(device=device, dtype=dtype)
    if args.compile:
        model = torch.compile(model, dynamic=False)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f"[cfg] compile={args.compile} n_block={args.n_block} L={args.token} d_pair={args.d_pair} "
          f"batch={args.batch} steps={args.steps} gpu={args.gpu} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)

    pair = torch.randn(args.batch, args.token, args.token, args.d_pair, device=device, dtype=dtype)
    mask = torch.ones(args.batch, args.token, device=device, dtype=torch.bool)

    # step loop
    times = []
    total_start = time.perf_counter()
    for i in range(args.steps):
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(pair, mask)
        loss = out.float().mean()
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  step {i:2d}: {dt*1000:8.1f} ms   loss={loss.item():+.4f}", flush=True)

    total = time.perf_counter() - total_start
    print(f"\n[summary]")
    print(f"  total : {total:.2f}s  ({args.steps} steps)")
    print(f"  step0 : {times[0]*1000:8.1f} ms   (compile+autotune)")
    if len(times) > 3:
        steady = times[3:]
        print(f"  steady: {sum(steady)/len(steady)*1000:8.1f} ms/step (mean of steps 3..{args.steps-1})")

if __name__ == "__main__":
    main()
