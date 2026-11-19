"""run_miniworld_diffusion_inference.py in its ORIGINAL settings (EMA, default precision, the engine's fused bf16 kernels) with ONE change, switched by env
MW_LOGIT_FP32=1: the pair-bias attention logits of the bias-only token DiT are made in fp32.

Why: engine `FusedBiasOnlyDiT.hoist` (bf16 runner) does  zh = LN(pair) -> bf16;  P = mm(pw_bf16, zh^T) -> a bf16 BUFFER (the logits, values in the thousands:
bf16 resolution 32..1024);  softmax_rows(P) in place.  The logits are rounded to bf16 before the softmax ever sees them.
Here, for a bf16 runner only, `hoist` is replaced by: LayerNorm(pair) in fp32, logits = LN(pair) @ (to_bias.weight * ln_pair.weight)^T in exact fp32, the key mask,
an fp32 softmax, and only the finished probabilities P are rounded to bf16 for the fused block (P is shared by every solver step and sample: built once per pair).
Nothing else changes; the engine files are not modified (runtime patch). Prints how many times each path ran. MW_LOGIT_FP32=0 (or unset) = the original code path."""
import atexit
import collections
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import run_miniworld_diffusion_inference as inf  # noqa: E402
from miniworld_engine.kernels.bias_only_dit.cuda import runner as R  # noqa: E402

ON = os.environ.get("MW_LOGIT_FP32") == "1"
if ON:
    torch.set_float32_matmul_precision("highest")  # the logit GEMM must be exact fp32 (the script default "medium" would use TF32)
calls = collections.Counter()
_init, _hoist = R.FusedBiasOnlyDiT.__init__, R.FusedBiasOnlyDiT.hoist
EPS = 1e-5


def _new_init(self, blocks, dtype=torch.bfloat16):
    blocks = list(blocks)
    _init(self, blocks, dtype)
    # the runner keeps pw = (to_bias.weight * ln_pair.weight) ROUNDED to bf16: keep the fp32 original too
    self.pw32 = torch.cat([b.attention.to_bias.weight.detach().float() * b.attention.ln_pair.weight.detach().float() for b in blocks], 0).contiguous()


def _new_hoist(self, pair, mask=None):
    if self.fp32 or not ON:
        calls["original hoist (" + ("fp32 runner" if self.fp32 else "bf16 runner, patch off") + ")"] += 1
        return _hoist(self, pair, mask)
    calls["PATCHED hoist (bf16 runner, logits + softmax in fp32)"] += 1
    L = pair.shape[1]
    # the model runs under bf16 autocast: an fp32 matmul in here would silently be cast to bf16 -- the very rounding this patch removes
    with torch.autocast("cuda", enabled=False):
        z = F.layer_norm(pair.reshape(L * L, self.dp).float(), (self.dp,), eps=EPS)  # the ln_pair weight is folded into pw32
        m = None if mask is None else mask.reshape(L).to(torch.bool)
        outs = []
        for b in range(self.nb):
            w = self.pw32[b * self.h : (b + 1) * self.h]  # [H, dp]
            lg = (z @ w.t()).t().reshape(self.h, L, L)  # [H, query m, key n], the layout of the original mm output
            assert lg.dtype is torch.float32, lg.dtype
            if m is not None:
                lg = lg.masked_fill(~m[None, None, :], torch.finfo(torch.float32).min)
            outs.append(torch.softmax(lg, dim=-1))
        return torch.cat(outs, 0).to(self.dtype).contiguous()


R.FusedBiasOnlyDiT.__init__ = _new_init
R.FusedBiasOnlyDiT.hoist = _new_hoist
atexit.register(lambda: print("\n=== hoist calls:", dict(calls), "| MW_LOGIT_FP32 =", ON, flush=True))
inf.cli()
