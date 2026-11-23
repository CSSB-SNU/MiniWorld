"""Strict check that an inference ran on PyTorch / cuBLAS kernels only. Sets the engine settings that switch its hand kernels off
(swa_dit_fused=False, swa_dit_h100_native=False), prints the effective settings, profiles one short inference (1 seed, 1 sample, 5 steps) and lists EVERY
CUDA kernel name that is not on the allow-list (PyTorch aten / cuBLAS+cuBLASLt GEMM / PyTorch mem-efficient SDPA / memcpy+memset). Empty list = pure PyTorch.
MW_PURE_RUN=1 skips the profiler (use the same settings for the real 5x5 run). No repo code is modified."""
import atexit
import collections
import os
import re
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import run_miniworld_diffusion_inference as inf  # noqa: E402
from miniworld_engine import settings  # noqa: E402

torch.set_float32_matmul_precision("highest")
settings.configure(swa_dit_fused=False, swa_dit_h100_native=False, token_pair_init_fused=False)
# The engine's input atom encoder hard-codes fused_triton=True (miniworld_engine/modules/input_feature_embedder/module.py:137), so no config reaches it:
# make every SWAAtomTransformer refuse the fused block -> the per-block PyTorch statements. Runtime patch only; no file is modified.
from miniworld_engine.modules.swa_atom_transformer import module as _swa_mod  # noqa: E402
_swa_mod.SWAAtomTransformer._fused_ok = lambda self, q, c, attention_params: False
if os.environ.get("MW_ROPE_COMPAT") == "1":   # the pre-2026-10-06 bf16 RoPE angles (docs/known-issues.md); the native H100 kernels refuse them, the PyTorch path takes them
    settings.configure(rope_angles_bf16_compat=True)
    print("rope_angles_bf16_compat = True", flush=True)
_Client = inf.Client
class Client(_Client):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        s = settings.current()
        print("engine settings:", {n: getattr(s, n, "?") for n in ("swa_dit_fused", "swa_dit_h100_native", "force_portable_kernels")}, flush=True)
inf.Client = Client

ALLOW = re.compile(r"^(memcpy|memset|void at::|at::|void \(anonymous namespace\)::|void cunn_|void cub::|void thrust::|ampere_|sm\d+_xmma|nvjet_|cutlass_\d+|"
                   r"fmha_cutlassF|void gemv|void splitK|void dot_kernel|void reduce_1Block|cublas|void flash::|void pytorch_flash|void cutlass::Kernel)", re.I)
if os.environ.get("MW_PURE_RUN") != "1":
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]); prof.__enter__()
    def report():
        prof.__exit__(None, None, None)
        kern = collections.Counter(e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
        bad = {n: c for n, c in kern.items() if not ALLOW.match(n.strip())}
        print(f"\n=== kernels NOT on the PyTorch/cuBLAS allow-list: {len(bad)} names, {sum(bad.values())} launches (total launches {sum(kern.values())}) ===")
        for n, c in sorted(bad.items(), key=lambda x: -x[1]):
            print(f"{c:8d}  {n[:170]}")
        print("\n=== CALLERS of the non-allow-listed engine kernels (innermost miniworld/foldforge frames) ===")
        shown = set()
        for e in prof.events():
            ks = [k.name for k in getattr(e, "kernels", []) if k.name in bad and re.search(r"swa_native|flash_attn|tpi_|cutlass_kernel", k.name)]
            if not ks:
                continue
            frames, p = [], e
            while p is not None and len(frames) < 8:   # the CPU-op parent chain of the launching op (no python stack: with_stack breaks kineto here)
                frames.append(p.name)
                p = p.cpu_parent
            key = (tuple(sorted({k[:40] for k in ks})), tuple(frames))
            if key in shown:
                continue
            shown.add(key)
            print(f"kernels={sorted({k[:48] for k in ks})}\n    op={e.name}\n    " + "\n    ".join(frames))
        print("=== end ===")
    atexit.register(report)
inf.cli()
