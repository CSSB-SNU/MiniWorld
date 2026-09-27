"""Memory of the token DiT fwd+bwd, engine bf16 core vs tdt core (hoist + cublas AdaLN), same model and inputs.
Reports memory held after the forward (what autograd saved) and the fwd+bwd peak, both above the pre-step baseline."""
import argparse, sys
import torch
from team_gm.modules import DiffusionTransformer
from team_gm.modules.exceptions import ImplementationType
p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--blocks", type=int, default=24)
p.add_argument("--keep-attn", action="store_true")
p.add_argument("--no-ckpt", action="store_true")
a = p.parse_args()
torch.set_float32_matmul_precision("medium")
from miniworld_engine.kernels.adaln.triton import training as _adaln_tr
_adaln_tr.set_forward_mode("cublas")
dev, L, A, NB = "cuda", a.length, 48, a.blocks
GB = 2 ** 30
for core in (("bf16_sm90",) if a.keep_attn else ("bf16", "bf16_sm90")):
    torch.manual_seed(0)
    cfg = DiffusionTransformer.Config(d_single=768, d_cond=384, d_pair=128, n_head=16, n_block=NB, use_qk_norm=True,
                                      n_checkpoint_segments=None if a.no_ckpt else NB,
                                      implementation=ImplementationType.MINIWORLD_ENGINE, hoist_pair_bias=True,
                                      attention_core_dtype=core, checkpoint_keep_attention=a.keep_attn)
    m = DiffusionTransformer(cfg).to(dev)
    single = torch.randn(A, 1, L, 768, device=dev, requires_grad=True)
    cond = torch.randn(A, 1, L, 384, device=dev, requires_grad=True)
    pair = torch.randn(1, L, L, 128, device=dev, requires_grad=True)
    for it in range(2):                                             # 2nd iteration: kernels built, caches warm
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = m(single, cond, pair, None)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - base
        out.float().square().mean().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        for t in (single, cond, pair): t.grad = None
        m.zero_grad(set_to_none=True)
        del out
    print(f"mem: L{L} NB{NB} ckpt={'off' if a.no_ckpt else 'per block'} core={core:9s} keep={int(a.keep_attn)}  held after fwd {held / GB:6.2f} GB"
          f"  fwd+bwd peak {peak / GB:6.2f} GB", flush=True)
    del m, single, cond, pair
