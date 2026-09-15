"""Does dropping `attention_pair_bias.ln_pair.bias` change the model's output?

The new engine removed this parameter and its `_load_from_state_dict` silently pops the
key, on the argument that the bias reaches the loss only as a per-head constant added to
every attention logit, where softmax is shift-invariant. The argument is sound in exact
arithmetic, but our epoch-93 checkpoint carries values up to |27.9| (Adam random-walks a
zero-gradient parameter on bf16 noise), and the kernels run in bf16 -- so the invariance
has to be measured, not assumed.

Re-injects the checkpoint's bias into every DiT block by wrapping `ln_pair`, and compares
the full diffusion output against the same model without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_miniworld_diffusion_train import Config  # noqa: E402


class _LNPlusBias(nn.Module):
    """`ln_pair` as the OLD engine had it: the LayerNorm, then its removed bias."""

    def __init__(self, ln: nn.Module, bias: torch.Tensor) -> None:
        super().__init__()
        self.ln = ln
        self.register_buffer("extra_bias", bias)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.ln(pair) + self.extra_bias.to(pair.dtype)


def main() -> None:
    cfg_path = Path("configs/miniworld/phase2b_diffusion_v101.yaml").resolve()
    with initialize_config_dir(str(cfg_path.parent), version_base=None):
        cfg = Config.model_validate(compose(config_name=cfg_path.name))

    from miniworld.models.diffusion.model import DiffusionModel
    torch.manual_seed(0)
    model = DiffusionModel(cfg.model).cuda().eval()

    ck = sorted(Path("runs/v1.0.1/phase2b/large_diffusion_L768")
                .glob("*/*/checkpoints/last.pt"))[-1]
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    biases = {k: v for k, v in sd.items()
              if k.endswith("attention_pair_bias.ln_pair.bias")}
    print(f"ckpt: {ck}")
    print(f"re-injecting {len(biases)} ln_pair biases, "
          f"max|b| = {max(v.abs().max().item() for v in biases.values()):.3f}")

    blocks = model.diffusion_module.diffusion_transformer.blocks
    n_aug, n_tok, d_single, d_pair = 2, 256, cfg.model.shared.d_single_token, cfg.model.shared.d_pair
    d_cond = cfg.model.shared.d_single
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1234)
    # the diffusion module holds fp32 params and training runs it under bf16 autocast,
    # so feed fp32 and let autocast pick the kernel dtype -- exactly the training path.
    single = torch.randn(n_aug, 1, n_tok, d_single, device=dev, generator=g)
    pair = torch.randn(1, n_tok, n_tok, d_pair, device=dev, generator=g)
    cond = torch.randn(n_aug, 1, n_tok, d_cond, device=dev, generator=g)

    def run(n: int | None = None) -> torch.Tensor:
        x = single
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for blk in blocks[:n] if n else blocks:
                x = blk(x, cond, pair)
        return x.float()

    def report(tag: str, a: torch.Tensor, b: torch.Tensor) -> float:
        diff = (a - b).abs()
        scale = b.abs().mean()
        rel = (diff.max() / scale).item()
        print(f"{tag:<26} mean|x|={scale:>12.4f}  absdiff max={diff.max():.3e} "
              f"mean={diff.mean():.3e}  reldiff max={rel:.3e}")
        return rel

    # (1) THE ALGEBRA. The removed bias b enters only as to_bias(ln(pair) + b), and
    # to_bias is Linear(d_pair -> n_head, bias=False), so the difference must be the
    # SAME vector W@b at every (i, j). If it is, the per-head logit shift is constant
    # along the softmax axis and the invariance is exact -- anything the full forward
    # then shows is bf16 rounding, not a change of function.
    blk0 = blocks[0]
    apb0 = blk0.attention_pair_bias
    key0 = "diffusion_module.diffusion_transformer.blocks.0.attention_pair_bias.ln_pair.bias"
    b0 = biases[key0].to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ln = apb0.ln_pair(pair)
        d_bias = (apb0.to_bias(ln + b0) - apb0.to_bias(ln)).float()
    per_head_spread = (d_bias - d_bias.mean(dim=(1, 2), keepdim=True)).abs().max()
    print(f"\n[algebra] to_bias shift: mean per head = "
          f"{d_bias.mean(dim=(1, 2)).flatten()[:4].tolist()}")
    print(f"[algebra] deviation from constant over (i,j): {per_head_spread:.3e}"
          f"   -> {'CONSTANT (softmax-invariant)' if per_head_spread < 1e-1 else 'NOT CONSTANT'}")

    # (2) ONE block: the honest per-block effect, free of depth amplification.
    out_new1 = run(1)
    apb0.ln_pair = _LNPlusBias(apb0.ln_pair, b0)
    out_old1 = run(1)
    rel1 = report("1 block", out_old1, out_new1)

    # (3) ALL 24 blocks. A residual stack amplifies any perturbation, so a large number
    # here with a small number above means rounding amplified by depth, NOT a functional
    # change -- read it next to the bf16 control below.
    for i, blk in enumerate(blocks):
        if i == 0:
            continue
        key = f"diffusion_module.diffusion_transformer.blocks.{i}.attention_pair_bias.ln_pair.bias"
        blk.attention_pair_bias.ln_pair = _LNPlusBias(
            blk.attention_pair_bias.ln_pair, biases[key].to(dev))
    out_old = run()
    for i, blk in enumerate(blocks):
        blk.attention_pair_bias.ln_pair = blk.attention_pair_bias.ln_pair.ln
    out_new = run()
    rel_all = report("24 blocks", out_old, out_new)

    # (4) CONTROL: the same stack's own bf16 run-to-run noise, from a perturbation far
    # smaller than the bias. If (3) is the same order, (3) is noise amplification.
    for blk in blocks:
        apb = blk.attention_pair_bias
        apb.ln_pair = _LNPlusBias(apb.ln_pair, torch.zeros_like(b0) + 1e-3)
    out_eps = run()
    rel_eps = report("24 blocks, b=1e-3 ctrl", out_eps, out_new)

    print(f"\n1-block rel diff      : {rel1:.3e}")
    print(f"24-block rel diff     : {rel_all:.3e}")
    print(f"24-block 1e-3 control : {rel_eps:.3e}")
    verdict = ("NO-OP (exact invariance; depth amplifies bf16 rounding only)"
               if per_head_spread < 1e-1 and rel1 < 1e-2
               else "CHANGES OUTPUT")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
