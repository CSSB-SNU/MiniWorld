"""MSA + Pairformer compile test — approximates the real MiniSWA trunk."""
from __future__ import annotations
import argparse, time, torch
from team_gm.modules import MiniMSAModule, MiniPairformer
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.layers import MSAPairWeightedAveraging, OuterProductMean
from team_gm.modules.primitives import LayerNorm as MwkLayerNorm


def force_pytorch_ln_in_msa(module):
    """Walk the model; replace the fused Triton LN in every OPM/MPWA with
    a plain PYTORCH LayerNorm so those two ops skip Triton autotune."""
    replaced = 0
    for m in module.modules():
        if isinstance(m, (OuterProductMean, MSAPairWeightedAveraging)):
            for name in ("ln_msa", "ln_pair"):
                if hasattr(m, name):
                    old = getattr(m, name)
                    if getattr(old, "implementation", None) == ImplementationType.PYTORCH:
                        continue
                    new = MwkLayerNorm(old.normalized_shape[0], implementation=ImplementationType.PYTORCH)
                    new = new.to(next(old.parameters()).device, next(old.parameters()).dtype) if any(p.numel() > 0 for p in old.parameters()) else new
                    setattr(m, name, new)
                    replaced += 1
    print(f"[patch] replaced {replaced} LN modules with PYTORCH impl", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--n_msa_block", type=int, default=4)
    p.add_argument("--n_pf_block", type=int, default=48)
    p.add_argument("--N", type=int, default=2048, help="MSA depth")
    p.add_argument("--L", type=int, default=384, help="tokens")
    p.add_argument("--d_pair", type=int, default=128)
    p.add_argument("--d_msa", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--impl", type=str, default="triton", choices=["pytorch", "triton"])
    p.add_argument("--msa_ln_pytorch", type=int, default=0,
                   help="If 1, force pytorch LN inside OPM+MPWA only (keep triton for the rest)")
    args = p.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16
    impl = {"pytorch": ImplementationType.PYTORCH, "triton": ImplementationType.TRITON}[args.impl]

    msa_cfg = MiniMSAModule.Config(
        d_msa=args.d_msa, d_pair=args.d_pair, d_single_token_input=449,
        n_block=args.n_msa_block, implementation=impl,
        d_hidden_msa=32, p_drop=0.0, p_drop_msa=0.0,
    )
    pf_cfg = MiniPairformer.Config(
        d_pair=args.d_pair, n_block=args.n_pf_block, p_drop=0.0, implementation=impl,
    )
    msa_mod = MiniMSAModule(msa_cfg).to(device=device, dtype=dtype)
    pf_mod = MiniPairformer(pf_cfg).to(device=device, dtype=dtype)

    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.msa = msa_mod
            self.pf = pf_mod
        def forward(self, msa_in, msa_mask, pair, single, mask):
            pair = self.msa(msa_in, msa_mask, pair, single=single, mask=mask)
            pair = self.pf(pair, mask=mask)
            return pair

    trunk = Trunk()
    if args.msa_ln_pytorch:
        force_pytorch_ln_in_msa(trunk)
    if args.compile:
        trunk = torch.compile(trunk, dynamic=False)
    optim = torch.optim.AdamW(trunk.parameters(), lr=1e-4)

    total_params = sum(p.numel() for p in trunk.parameters())
    print(f"[cfg] compile={args.compile} impl={args.impl} n_msa={args.n_msa_block} n_pf={args.n_pf_block} "
          f"N={args.N} L={args.L} d_msa={args.d_msa} d_pair={args.d_pair} steps={args.steps} gpu={args.gpu} "
          f"params={total_params/1e6:.2f}M", flush=True)

    # inputs
    B, N, L = args.batch, args.N, args.L
    num_res_class = msa_cfg.num_res_class  # 32
    msa_in = torch.randn(B, N, L, num_res_class + 2, device=device, dtype=dtype)
    msa_mask = torch.ones(B, N, device=device, dtype=torch.bool)
    pair = torch.randn(B, L, L, args.d_pair, device=device, dtype=dtype)
    single = torch.randn(B, L, msa_cfg.d_single_token_input, device=device, dtype=dtype)
    mask = torch.ones(B, L, device=device, dtype=torch.bool)

    times = []
    for i in range(args.steps):
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = trunk(msa_in, msa_mask, pair, single, mask)
        loss = out.float().mean()
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  step {i:2d}: {dt*1000:9.1f} ms   loss={loss.item():+.4f}", flush=True)

    print("[summary]")
    print(f"  step0 : {times[0]*1000:9.1f} ms   (compile+autotune)")
    if len(times) > 2:
        steady = times[2:]
        print(f"  steady: {sum(steady)/len(steady)*1000:9.1f} ms/step (mean of steps 2..{args.steps-1})")


if __name__ == "__main__":
    main()
