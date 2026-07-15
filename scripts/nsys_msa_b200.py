"""nsys component breakdown of the 4-block MiniMSAModule (depth=2048, L=384, MSA
self-update on). NVTX ranges (forward hooks) wrap each sub-component summed across
blocks: opm / mpwa / msa_transition / trimul / pair_transition. Eager so kernels
attribute cleanly. Grabs the real msa_module inputs from one model forward."""
from __future__ import annotations
import os, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path("scripts").resolve()))
from miniworld.configs.models import AtomSWAConfig
from miniworld.data.features.batch import Batch  # noqa
from miniworld.models.distogram_only import MiniSWAModel
from run_miniworld_distogram_train import _build_precompile_batch

torch.manual_seed(0); dev = torch.device("cuda")
L = int(os.environ.get("L_TOK", "384")); DEPTH = int(os.environ.get("MSA_DEPTH", "2048"))
ITERS = int(os.environ.get("ITERS", "3"))
cfg = OmegaConf.load("configs/miniworld/model/medium_distogram.yaml")
m = OmegaConf.to_container(cfg, resolve=True); m["trunk"]["pairformer"]["n_block"] = 48
model = MiniSWAModel(MiniSWAModel.Config(
    shared=m["shared"], input_feat_embbeder=m["input_feat_embbeder"],
    atom_swa=AtomSWAConfig(enabled=True, backend="flash", swa_window_size=1_000_000),
    trunk=m["trunk"])).to(dev)
model._forced_n_recycle = 1
batch = _build_precompile_batch(device=dev, msa_depth=DEPTH, n_tokens=L, n_atoms=L*4,
                                n_templates=1, num_res_class=int(m["shared"]["num_res_class"]))
grabbed = {}
def _grab(mod, args): grabbed["a"] = tuple(a.clone().detach() if torch.is_tensor(a) else a for a in args)
h = model.msa_module.register_forward_pre_hook(_grab)
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    model.forward(msa=batch.msa, reference=batch.reference, scheme=batch.scheme,
                  sequence=batch.sequence, structure=batch.structure)
h.remove()
args = list(grabbed["a"])

PHASE = "infer"
_COMPS = ["outer_product_mean", "msa_pair_weighted_averaging", "transition_msa",
          "tri_multi", "transition_pair"]
for blk in model.msa_module.blocks:
    for cname in _COMPS:
        sub = getattr(blk, cname, None)
        if sub is None:
            continue
        def _pre(m_, inp, _c=cname):
            torch.cuda.nvtx.range_push(f"{PHASE}/{_c}")
        def _post(m_, inp, out, _c=cname):
            torch.cuda.nvtx.range_pop()
        sub.register_forward_pre_hook(_pre)
        sub.register_forward_hook(_post)

msa = model.msa_module
# warmup
msa.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    msa(*args); msa(*args)
targs = list(args); targs[2] = args[2].clone().requires_grad_(True)
msa.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    msa(*targs).float().pow(2).mean().backward()
msa.zero_grad(set_to_none=True)
torch.cuda.synchronize()

PHASE = "infer"; msa.eval()
torch.cuda.nvtx.range_push("INFER")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(ITERS):
        msa(*args)
torch.cuda.synchronize(); torch.cuda.nvtx.range_pop()

PHASE = "train"; msa.train()
torch.cuda.nvtx.range_push("TRAIN")
for _ in range(ITERS):
    msa.zero_grad(set_to_none=True)
    if targs[2].grad is not None: targs[2].grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        msa(*targs).float().pow(2).mean().backward()
torch.cuda.synchronize(); torch.cuda.nvtx.range_pop()
print("NSYS MSA RUN DONE depth=%d L=%d" % (DEPTH, L))
