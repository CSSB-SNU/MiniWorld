"""no_grad forward of the token DiT (24 blocks, hoist): what stays allocated after the forward besides the output,
engine bf16 core vs tdt core. 0 means nothing was kept for a backward."""
import argparse
import torch
from team_gm.modules import DiffusionTransformer
from team_gm.modules.exceptions import ImplementationType
p = argparse.ArgumentParser(); p.add_argument("--length", type=int, default=768); a = p.parse_args()
torch.set_float32_matmul_precision("medium")
from miniworld_engine.kernels.adaln.triton import training as _adaln_tr
_adaln_tr.set_forward_mode("cublas")
dev, L, A, NB, MB = "cuda", a.length, 48, 24, 2 ** 20
for core in ("bf16", "tdt"):
    torch.manual_seed(0)
    cfg = DiffusionTransformer.Config(d_single=768, d_cond=384, d_pair=128, n_head=16, n_block=NB, use_qk_norm=True,
                                      n_checkpoint_segments=NB, implementation=ImplementationType.MINIWORLD_ENGINE,
                                      hoist_pair_bias=True, attention_core_dtype=core)
    m = DiffusionTransformer(cfg).to(dev).eval()
    single = torch.randn(A, 1, L, 768, device=dev)
    cond = torch.randn(A, 1, L, 384, device=dev)
    pair = torch.randn(1, L, L, 128, device=dev)
    with torch.no_grad():
        for _ in range(2):
            torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            out = m(single, cond, pair, None)
            torch.cuda.synchronize()
            held = torch.cuda.memory_allocated() - base - out.numel() * out.element_size()
            peak = torch.cuda.max_memory_allocated() - base
            del out
    print(f"nograd: L{L} core={core:4s}  kept besides the output {held / MB:8.1f} MB   forward peak {peak / MB:8.1f} MB",
          flush=True)
    del m, single, cond, pair
