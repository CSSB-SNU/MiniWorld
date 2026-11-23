"""Measure what dtype the diffusion module REALLY runs in under the dfp32 settings: parameter dtypes of every diffusion_module leaf class and the
output dtype of every submodule's first forward (forward hooks, one short inference: 1 seed, 1 sample, 5 steps). No repo code is modified."""
import atexit
import collections
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import run_miniworld_diffusion_inference as inf  # noqa: E402

torch.set_float32_matmul_precision("highest")
par = collections.defaultdict(collections.Counter)   # (group, class) -> {param dtype: n tensors}
act = collections.defaultdict(collections.Counter)   # (group, class) -> {output dtype: n calls}
seen = set()

def group(name):
    p = name.split(".")
    return ".".join(p[:3]) if p and p[0] else "root"

_Client = inf.Client
class Client(_Client):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        dm = self.model.diffusion_module
        for n, m in dm.named_modules():
            for _, p in m.named_parameters(recurse=False):
                par[(group(n), type(m).__name__)][str(p.dtype)] += 1
            def hook(mod, inp, out, key=(group(n), type(m).__name__), n=n):
                if n in seen:
                    return
                seen.add(n)
                outs = [out] if torch.is_tensor(out) else [o for o in (out if isinstance(out, (tuple, list)) else []) if torch.is_tensor(o)]
                for o in outs:
                    act[key][str(o.dtype)] += 1
            m.register_forward_hook(hook)
        print("dit_dtype:", getattr(self.model, "dit_dtype", None), flush=True)
inf.Client = Client

def report():
    print("\n=== PARAMETER dtypes (diffusion_module, per group/class) ===")
    for k in sorted(par):
        print(f"{k[0]:<40}{k[1]:<28}{dict(par[k])}")
    print("\n=== OUTPUT dtypes of each submodule's first forward ===")
    for k in sorted(act):
        print(f"{k[0]:<40}{k[1]:<28}{dict(act[k])}")
atexit.register(report)
inf.cli()
