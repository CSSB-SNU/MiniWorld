"""InputFeatureEmbedderESMFold2Style + Pairformer compile/eager test.

Runs the SWA/3D-RoPE atom embedder + Pairformer with a synthetic batch to
isolate whether input-embedder Triton kernels autotune-stall the first step.
"""
from __future__ import annotations
import argparse, sys, time, torch
sys.path.insert(0, "/home/snu_hwle/psk/MiniWorld/scripts")
from run_miniworld_distogram_train import _build_precompile_batch

from team_gm.modules import DiffusionTransformer, MiniPairformer
from team_gm.modules.exceptions import ImplementationType
from miniworld.configs import SharedConfig
from miniworld.configs.models import AtomSWAConfig
from miniworld.modules.input_feature_embedder_esmfold2_style import (
    InputFeatureEmbedderESMFold2Style,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--n_pf_block", type=int, default=48)
    p.add_argument("--n_embed_block", type=int, default=3)
    p.add_argument("--L", type=int, default=384, help="tokens")
    p.add_argument("--N_atom", type=int, default=4096)
    p.add_argument("--N_msa", type=int, default=2048)
    p.add_argument("--n_templates", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--impl", type=str, default="triton", choices=["pytorch", "triton"])
    p.add_argument("--swa_backend", type=str, default="flash", choices=["flex", "sdpa", "flash"])
    p.add_argument("--n_recycle", type=int, default=1,
                   help="If >1, loop the PF branch that many times per step (simulates recycling).")
    args = p.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16
    impl = {"pytorch": ImplementationType.PYTORCH, "triton": ImplementationType.TRITON}[args.impl]

    # Match yaml config: d_single_token=768 (atom encoder out concats with msa init)
    # d_single_token_input must be > d_single_token; the diff is the msa-init width.
    shared = SharedConfig(
        d_single=384, d_single_atom=128,
        d_single_token=384, d_single_token_input=449,
        d_pair=128, d_pair_atom=16, num_res_class=32, implementation=impl,
    )
    diff_cfg = DiffusionTransformer.Config(n_block=args.n_embed_block, n_head=4, d_atom=128)
    atom_swa = AtomSWAConfig(enabled=True, swa_window_size=1_000_000, backend=args.swa_backend,
                             expansion_ratio=2)

    # Real MiniSWAModel keeps the input embedder in fp32 (no .to(bfloat16)),
    # only MSA/PF are cast. RelPos.forward does .float() internally; feeding
    # into a bf16 Linear would then crash. Match the real dtype layout.
    embedder = InputFeatureEmbedderESMFold2Style(
        shared, diff_cfg, atom_swa_config=atom_swa, produce_single_init=False,
    ).to(device=device)

    pf_cfg = MiniPairformer.Config(d_pair=128, n_block=args.n_pf_block, p_drop=0.0, implementation=impl)
    pf = MiniPairformer(pf_cfg).to(device=device, dtype=dtype)

    # embedder stays fp32; MSA/PF are bf16. The SWA path inside the embedder
    # runs in fp32 (params fp32, inputs fp32) — no dtype crossover.

    class Model(torch.nn.Module):
        def __init__(self, n_recycle):
            super().__init__()
            self.emb = embedder
            self.pf = pf
            self.n_recycle = n_recycle
        def forward(self, token_single_msa, ref, sch, struct):
            _, _, pair = self.emb(token_single_msa, ref, sch, struct)
            pair = pair.to(dtype)
            # Recycling: run the PF branch n_recycle times, feeding the output
            # back in as the next-round pair init (matches MiniSWAModel's
            # recycle loop where pair_prev + LN + Linear routes back to PF).
            for _ in range(self.n_recycle):
                pair = self.pf(pair, mask=struct.token_mask)
            return pair

    model = Model(args.n_recycle)
    if args.compile:
        model = torch.compile(model, dynamic=False)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f"[cfg] compile={args.compile} impl={args.impl} swa={args.swa_backend} "
          f"n_embed={args.n_embed_block} n_pf={args.n_pf_block} n_recycle={args.n_recycle} "
          f"L={args.L} N_atom={args.N_atom} N_msa={args.N_msa} steps={args.steps} gpu={args.gpu} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    batch = _build_precompile_batch(
        device=device, msa_depth=args.N_msa, n_tokens=args.L, n_atoms=args.N_atom,
        n_templates=args.n_templates, num_res_class=32,
    )
    # Keep batch tensors in whatever dtype _build_precompile_batch produced
    # (mostly fp32) — embedder is fp32 in real training.
    # token_single_msa: [B, L, d_single_token_input]. init_token_single_msa fuses MSA into a per-token init.
    # For the test just create a random one of the right shape.
    B = 1
    # d_single_token_init = d_single_token_input - d_single_token (concat with atom encoder output)
    # Embedder runs in fp32 in real training; keep token_single_msa in fp32 too.
    d_msa_init = shared.d_single_token_input - shared.d_single_token
    token_single_msa = torch.randn(B, args.L, d_msa_init, device=device)  # fp32 default

    times = []
    for i in range(args.steps):
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(token_single_msa, batch.reference, batch.scheme, batch.structure)
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
