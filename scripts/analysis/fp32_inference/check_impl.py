"""Which implementation REALLY runs: (1) the class + `implementation` attribute of every diffusion_module block, (2) the CUDA kernels torch.profiler saw during one
short inference (1 seed, 1 sample, 5 steps), grouped by name. Same settings as pytorch_dfp32_8oxk.sbatch. No repo code is modified."""
import atexit
import collections
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import run_miniworld_diffusion_inference as inf  # noqa: E402

torch.set_float32_matmul_precision("highest")
_Client = inf.Client
class Client(_Client):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        print("\n=== diffusion_module: class + implementation attr of each block type ===", flush=True)
        seen = collections.Counter()
        for n, m in self.model.diffusion_module.named_modules():
            impl = getattr(m, "implementation", None) or getattr(getattr(m, "config", None), "implementation", None)
            key = (type(m).__module__.split(".")[-1], type(m).__name__, str(impl), getattr(m, "fused_triton", getattr(getattr(m, "config", None), "fused_triton", "-")))
            seen[key] += 1
        for k, v in sorted(seen.items()):
            if k[2] != "None" or "Block" in k[1] or "Transformer" in k[1]:
                print(f"{v:4d}  module={k[0]:<28} class={k[1]:<30} implementation={k[2]:<24} fused_triton={k[3]}", flush=True)
inf.Client = Client

prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
prof.__enter__()

def report():
    prof.__exit__(None, None, None)
    kern = collections.Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kern[e.name] += 1
    groups = collections.Counter()
    for name, n in kern.items():
        low = name.lower()
        g = ("triton" if low.startswith("triton") else "cublas/cutlass gemm" if any(s in low for s in ("gemm", "cutlass", "cublas", "sgemm", "xmma", "nvjet")) else
             "flash/attention" if any(s in low for s in ("flash", "fmha", "attn", "attention")) else
             "engine custom (swa/dit/cuda ext)" if any(s in low for s in ("swa", "dit", "miniworld", "rope", "wgmma", "sm90")) else
             "elementwise/reduce/other (aten)")
        groups[g] += n
    print("\n=== CUDA kernel launches by group ===")
    for g, n in groups.most_common():
        print(f"{n:8d}  {g}")
    print("\n=== ALL non-aten/non-gemm kernel names ===")
    for name, n in kern.most_common():
        low = name.lower()
        if low.startswith("triton") or any(s in low for s in ("swa", "dit", "miniworld", "rope", "wgmma", "sm90", "fmha", "flash")):
            print(f"{n:8d}  {name[:160]}")
    print("\n=== top kernel names (by launches) ===")
    for name, n in kern.most_common(40):
        print(f"{n:8d}  {name[:140]}")
atexit.register(report)
inf.cli()
