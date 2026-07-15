#!/usr/bin/env python3
"""Microbenchmark: FA4 varlen (unpad/pack) vs FA4 dense padded+window for the
SWA atom-attention core in MiniWorld.

Motivation
----------
`build_attention_params` (swa_atom_attention.py:437) builds a varlen unpadding
layout via `torch.nonzero(valid)` -> `indices`/`cu_seqlens`. That length is the
number of VALID atoms in the batch, so it changes every step -> torch.compile
recompiles, and it is fundamentally incompatible with a whole-model CUDA graph
(data-dependent shape + nonzero sync).

The candidate replacement is a STATIC-shape path: run FA4 dense on the padded
[N, S, H, D] tensor with a sliding window (and mask padding). This script
measures the wall-clock cost of the two so we know what we trade for CUDA-graph
compatibility.

Operating point defaults come from the running config
(config_exp_msa3_24_3_no_single_ropeswa_af3_mpfull_b200_8gpu_edm.yaml):
  n_head=4, swa_window_size=128 (-> half_window=64), max_atoms(S)=4096, n_block=3.
N = num_aug * batch_per_gpu; num_aug/batch not in the yaml, so parametrized
(defaults num_aug=4, batch=8 -> N=32). Fill fraction estimated ~0.85 from the
logged valid-atom counts (114k-139k over N*S).

Usage
-----
  cd ~/psk/MiniWorld
  # pick a FREE gpu (do NOT steal one from the live 8-GPU training):
  CUDA_VISIBLE_DEVICES=0 pixi run --frozen python scratchpad_logs/bench_swa_varlen_vs_dense.py \
      --N 32 --S 4096 --heads 4 --dim 32 --half-window 64 --fill 0.85 --blocks 3
"""
import argparse
import statistics as stats

import torch
import torch.nn.functional as F

# ---- FA4 (CuTeDSL) probing ------------------------------------------------
_fa_varlen = None
_fa_dense = None
_fa_err = None
try:
    from flash_attn.cute import flash_attn_varlen_func as _fa_varlen  # type: ignore
    try:
        from flash_attn.cute import flash_attn_func as _fa_dense  # type: ignore
    except Exception as e:  # dense entry point may differ across builds
        _fa_dense = None
        _fa_err = f"flash_attn.cute.flash_attn_func import failed: {e!r}"
except Exception as e:  # pragma: no cover
    _fa_err = f"flash_attn.cute import failed: {e!r}"


def index_first_axis(x, indices):
    return x[indices]


def pad_input(out_u, indices, batch, seqlen):
    h, d = out_u.shape[1:]
    out = out_u.new_zeros(batch * seqlen, h, d)
    out[indices] = out_u
    return out.view(batch, seqlen, h, d)


def make_inputs(N, S, H, D, fill, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(N, S, H, D, generator=g).to(device=device, dtype=torch.bfloat16)
    k = torch.randn(N, S, H, D, generator=g).to(device=device, dtype=torch.bfloat16)
    v = torch.randn(N, S, H, D, generator=g).to(device=device, dtype=torch.bfloat16)
    # Valid atoms packed at the FRONT of each row (matches atom_mask layout);
    # per-row seqlen jittered around `fill`*S so cu_seqlens is nontrivial.
    base = int(round(fill * S))
    jit = torch.randint(-S // 20, S // 20 + 1, (N,), generator=g)
    seqlens = (torch.full((N,), base) + jit).clamp_(1, S).to(torch.int32)
    ar = torch.arange(S).view(1, S)
    valid = ar < seqlens.view(N, 1)  # [N, S] bool, front-packed
    return q, k, v, valid.to(device)


def build_varlen_params(valid):
    """Mirror of build_attention_params (the varlen prep under test)."""
    seqlens = valid.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
    cu = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = valid.shape[-1]
    return indices, cu, max_seqlen


def run_varlen(q, k, v, valid, scale, half_window, include_prep):
    """Full current path: (optional prep) + unpad gather + FA4 varlen + scatter."""
    N, S, H, D = q.shape
    if include_prep:
        indices, cu, max_seqlen = build_varlen_params(valid)
    else:
        indices, cu, max_seqlen = run_varlen.cache
    q_u = index_first_axis(q.reshape(N * S, H, D), indices)
    k_u = index_first_axis(k.reshape(N * S, H, D), indices)
    v_u = index_first_axis(v.reshape(N * S, H, D), indices)
    window = (-1, -1) if half_window < 0 else (half_window, half_window)
    out_u = _fa_varlen(
        q_u, k_u, v_u,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=scale, window_size=window,
    )
    if isinstance(out_u, tuple):
        out_u = out_u[0]
    return pad_input(out_u, indices, N, S)


def run_dense(q, k, v, valid, scale, half_window):
    """Static-shape candidate: FA4 dense on padded [N,S,H,D] with sliding window.

    Padding atoms sit at the tail; we zero padded-query outputs afterwards. (A
    boundary query can still see a few padding KEYS within its window -- for a
    faithful version those keys would be masked; that costs ~nothing in time and
    does not change the speed comparison, which is the point here.)
    """
    N, S, H, D = q.shape
    window = (-1, -1) if half_window < 0 else (half_window, half_window)
    out = _fa_dense(q, k, v, softmax_scale=scale, window_size=window)
    if isinstance(out, tuple):
        out = out[0]
    return out * valid.unsqueeze(-1).unsqueeze(-1)


def run_sdpa(q, k, v, valid, scale, half_window):
    """Reference: SDPA with band+padding mask (the eager fallback in the repo)."""
    N, S, H, D = q.shape
    qk = q.transpose(1, 2)  # [N,H,S,D]
    kk = k.transpose(1, 2)
    vk = v.transpose(1, 2)
    idx = torch.arange(S, device=q.device)
    band = (idx.view(S, 1) - idx.view(1, S)).abs() <= half_window if half_window >= 0 else torch.ones(S, S, dtype=torch.bool, device=q.device)
    keymask = valid.view(N, 1, 1, S)
    allowed = band.view(1, 1, S, S) & keymask
    out = F.scaled_dot_product_attention(qk, kk, vk, attn_mask=allowed, scale=scale)
    return (out.transpose(1, 2)) * valid.unsqueeze(-1).unsqueeze(-1)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return stats.mean(times), stats.median(times), min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32, help="num_aug*batch (flattened)")
    ap.add_argument("--S", type=int, default=4096, help="padded atom length")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=32, help="head dim (c_atom/heads)")
    ap.add_argument("--half-window", type=int, default=64,
                    help="window half-width; swa_window_size=128 -> 64. <0 = global")
    ap.add_argument("--fill", type=float, default=0.85, help="valid-atom fraction")
    ap.add_argument("--blocks", type=int, default=3, help="n_block (repeat per fwd)")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device visible"
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    util_hint = "(check `nvidia-smi` -- do NOT run on a GPU busy with training)"
    print(f"# device: {name}  free={free/1e9:.1f}GB/{total/1e9:.1f}GB {util_hint}")
    print(f"# FA4 varlen: {'OK' if _fa_varlen else 'MISSING'}   "
          f"FA4 dense: {'OK' if _fa_dense else 'MISSING'}")
    if _fa_err:
        print(f"# note: {_fa_err}")
    print(f"# shape N={args.N} S={args.S} H={args.heads} D={args.dim} "
          f"half_window={args.half_window} fill={args.fill} blocks={args.blocks}")

    q, k, v, valid = make_inputs(args.N, args.S, args.heads, args.dim, args.fill, dev)
    scale = 1.0 / (args.dim ** 0.5)
    nnz = int(valid.sum().item())
    print(f"# valid atoms (nnz) = {nnz}  of N*S={args.N*args.S}  "
          f"({100*nnz/(args.N*args.S):.1f}% fill)")
    run_varlen.cache = build_varlen_params(valid)

    B = args.blocks

    def rep(f):  # emulate n_block repeats within one forward
        def g():
            o = None
            for _ in range(B):
                o = f()
            return o
        return g

    results = {}
    if _fa_varlen is not None:
        results["varlen (prep+gather+FA+scatter)"] = bench(
            rep(lambda: run_varlen(q, k, v, valid, scale, args.half_window, True)), args.iters)
        results["varlen (FA+gather+scatter, prep cached)"] = bench(
            rep(lambda: run_varlen(q, k, v, valid, scale, args.half_window, False)), args.iters)
    if _fa_dense is not None:
        results["dense padded+window (static shape)"] = bench(
            rep(lambda: run_dense(q, k, v, valid, scale, args.half_window)), args.iters)
    results["SDPA band+mask (eager reference)"] = bench(
        rep(lambda: run_sdpa(q, k, v, valid, scale, args.half_window)), args.iters)

    print(f"\n{'method':44s} {'mean ms':>9s} {'median':>9s} {'min':>9s}")
    print("-" * 76)
    base = None
    for klab, (m, med, mn) in results.items():
        if base is None:
            base = med
        print(f"{klab:44s} {m:9.3f} {med:9.3f} {mn:9.3f}   ({med/base:4.2f}x)")

    # correctness sanity: varlen vs SDPA on valid region
    if _fa_varlen is not None:
        ov = run_varlen(q, k, v, valid, scale, args.half_window, True).float()
        os_ = run_sdpa(q, k, v, valid, scale, args.half_window).float()
        m = valid.unsqueeze(-1).unsqueeze(-1)
        err = ((ov - os_).abs() * m).max().item()
        print(f"\n# max|varlen - SDPA| on valid atoms = {err:.4f} (bf16, expect <~0.05)")


if __name__ == "__main__":
    main()
