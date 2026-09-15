"""Check removed LayerNorm offsets on actual diffusion inputs, with per-head evidence.

Restores the offset INSIDE the original LayerNorm operation. No checkpoint is
modified. The float32 attention distributions are a diagnostic reference for the
Triton kernel's float32 logit accumulation, not a structure-quality evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_phase2_cudagraph import difference, load_model_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/miniworld/phase2b_diffusion_v101.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weights", choices=("model", "ema"), default="model")
    ap.add_argument("--catalog-snapshot", type=Path)
    args = ap.parse_args()
    from miniworld_engine import settings

    settings.configure()
    model, batch, cfg, ck = load_model_batch(
        args.config, args.ckpt, args.weights, args.catalog_snapshot
    )
    model.eval()
    model._forced_n_recycle = 2
    sd = dict(ck["model_state_dict"])
    if args.weights == "ema":
        sd.update(ck["ema_state_dict"])
    blocks = model.diffusion_module.diffusion_transformer.blocks
    biases = [
        sd[
            f"diffusion_module.diffusion_transformer.blocks.{i}.attention_pair_bias.ln_pair.bias"
        ]
        .cuda()
        .float()
        for i in range(len(blocks))
    ]
    record = {
        "checkpoint": args.ckpt,
        "weights": args.weights,
        "catalog_snapshot": str(args.catalog_snapshot)
        if args.catalog_snapshot
        else None,
        "epoch": ck.get("epoch"),
        "global_step": ck.get("global_step"),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "heads": [],
    }
    # Hooks capture real token-DiT inputs after the diffusion atom encoder.
    captured = []

    def save_inputs(module, inputs, kwargs):
        values = (*inputs, kwargs.get("mask"))
        captured.append(
            tuple(
                t.detach().clone() if isinstance(t, torch.Tensor) else t for t in values
            )
        )

    hook = model.diffusion_module.diffusion_transformer.register_forward_pre_hook(
        save_inputs, with_kwargs=True
    )
    from team_gm.diffusion import EDMScheduler, EuclideanDiffuser

    diffuser = EuclideanDiffuser(
        config=EuclideanDiffuser.EuclideanConfig(seed=0),
        scheduler=EDMScheduler(cfg.diffuser.scheduler),
    )
    _, x, mask, t, _ = diffuser.sample(
        batch.structure.atom_pos, num_augment=1, mask=batch.structure.atom_pos_mask
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        print("Real batch trunk and diffusion forward", flush=True)
        tsi, pair_trunk = model.condition_forward(
            batch.msa,
            batch.reference,
            batch.scheme,
            batch.sequence,
            batch.structure,
            batch.template,
        )
        out_new = model.diffusion_forward(
            batch.reference, batch.scheme, batch.structure, x, mask, t, tsi, pair_trunk
        )
        hook.remove()
        single, cond, pair, *extra = captured[0]
        record["input_shapes"] = {
            "single": list(single.shape),
            "cond": list(cond.shape),
            "pair": list(pair.shape),
        }
        record["sigma_embedding"] = t.cpu().tolist()
        # Exact operation restoration: LayerNorm(input, weight, bias), not LN(input)+bias.
        for blk, bias in zip(blocks, biases):
            blk.attention_pair_bias.ln_pair.bias = torch.nn.Parameter(
                bias, requires_grad=False
            )
        out_old = model.diffusion_forward(
            batch.reference, batch.scheme, batch.structure, x, mask, t, tsi, pair_trunk
        )
        record["denoising_output_old_vs_new"] = difference(out_old, out_new)
        valid = mask.bool().unsqueeze(-1).expand_as(out_old)
        record["valid_atom_output_old_vs_new"] = difference(
            out_old[valid], out_new[valid]
        )
        record["valid_atoms"] = int(mask.bool().sum())
        record["valid_tokens"] = int(batch.structure.token_mask.sum())
        print(f"OUTPUT {record['denoising_output_old_vs_new']}", flush=True)
        # Inspect each block along the restored-old trajectory. Both candidate
        # biases are compared with the SAME Q/K and pair input for each block.
        current = single
        for i, (blk, offset) in enumerate(zip(blocks, biases)):
            apb = blk.attention_pair_bias
            normalized = apb.ada_ln_in(current, cond)
            q, k, value = (
                apb.to_query(normalized),
                apb.to_key(normalized),
                apb.to_value(normalized),
            )
            shape = (*q.shape[:3], apb.n_head, q.shape[-1] // apb.n_head)
            q, k, value = q.view(shape), k.view(shape), value.view(shape)
            if apb.use_qk_norm:
                q, k = apb.norm_query(q), apb.norm_key(k)
            old_ln = apb.ln_pair(pair)
            old_bias = apb.to_bias(old_ln)
            apb.ln_pair.bias = None
            new_ln = apb.ln_pair(pair)
            new_bias = apb.to_bias(new_ln)
            apb.ln_pair.bias = torch.nn.Parameter(offset, requires_grad=False)
            with torch.autocast("cuda", enabled=False):
                # Same head layout and scale as the actual attention core.
                qk = torch.einsum(
                    "abihd,abjhd->abhij", q.float(), k.float()
                ) / math.sqrt(q.shape[-1])
                logits_old = qk + old_bias.float().permute(0, 3, 1, 2)[None]
                logits_new = qk + new_bias.float().permute(0, 3, 1, 2)[None]
                if extra and extra[0] is not None:
                    amask = extra[0]
                    if amask.ndim == 2:
                        amask = amask[None]
                    logits_old.masked_fill_(~amask[:, :, None, None, :], -float("inf"))
                    logits_new.masked_fill_(~amask[:, :, None, None, :], -float("inf"))
                po, pn = logits_old.softmax(-1), logits_new.softmax(-1)
                query_mask = (
                    extra[0]
                    if extra and extra[0] is not None
                    else torch.ones(q.shape[:3], device=q.device, dtype=torch.bool)
                )
                if query_mask.ndim == 2:
                    query_mask = query_mask[None]
                query_mask = query_mask.expand(q.shape[:3])

                def valid_query_mean(rows):
                    return (rows * query_mask[:, :, None, :]).sum(
                        (0, 1, 3)
                    ) / query_mask.sum()

                uniform = query_mask[:, :, None, None, :].float()
                uniform = uniform / uniform.sum(-1, keepdim=True).clamp_min(1)
                entropy_old = valid_query_mean(-(po * po.clamp_min(1e-30).log()).sum(-1))
                entropy_new = valid_query_mean(-(pn * pn.clamp_min(1e-30).log()).sum(-1))
                tv = valid_query_mean((po - pn).abs().sum(-1).mul(0.5))
                uniform_tv = valid_query_mean((po - uniform).abs().sum(-1).mul(0.5))
                c = F.linear(offset, apb.to_bias.weight.float())
                if i in (0, 6, 7):
                    # Recover probabilities from the actual kernel: V consists
                    # of successive columns of the identity. P @ V returns those
                    # columns of P, including the kernel's own arithmetic/rounding.
                    kernel_p = torch.empty_like(po)
                    eye = torch.eye(q.shape[2], device=q.device, dtype=value.dtype)
                    for start in range(0, q.shape[2], q.shape[-1]):
                        width = min(q.shape[-1], q.shape[2] - start)
                        basis = F.pad(
                            eye[:, start : start + width], (0, q.shape[-1] - width)
                        )
                        probe_v = (
                            basis[None, None, :, None, :].expand_as(value).contiguous()
                        )
                        actual = apb._kernel_attention_pair_bias(
                            q, k, probe_v, old_bias, extra[0] if extra else None
                        )
                        kernel_p[..., start : start + width] = actual.permute(
                            0, 1, 3, 2, 4
                        )[..., :width]
                    kernel_p /= kernel_p.sum(-1, keepdim=True).clamp_min(1e-30)
                    kernel_entropy = valid_query_mean(
                        -(kernel_p * kernel_p.clamp_min(1e-30).log()).sum(-1)
                    )
                    kernel_uniform = valid_query_mean(
                        (kernel_p - uniform).abs().sum(-1).mul(0.5)
                    )
                # Derive actual BF16 spacing, avoiding the old off-by-one exponent.
                cbf = c.abs().to(torch.bfloat16)
                ulp = (
                    torch.nextafter(cbf, torch.full_like(cbf, float("inf"))) - cbf
                ).float()
                # FP64 verifies the algebra independently of low-precision projections.
                p64 = pair[:, :16, :16].double()
                ln64 = F.layer_norm(
                    p64,
                    apb.ln_pair.normalized_shape,
                    apb.ln_pair.weight.double(),
                    None,
                    apb.ln_pair.eps,
                )
                w64, b64 = apb.to_bias.weight.double(), offset.double()
                algebra_error = (
                    (
                        (F.linear(ln64 + b64, w64) - F.linear(ln64, w64))
                        - F.linear(b64, w64)
                    )
                    .abs()
                    .max()
                    .item()
                )
                for h in range(apb.n_head):
                    row = {
                        "block": i,
                        "head": h,
                        "offset": c[h].item(),
                        "bf16_ulp": ulp[h].item(),
                        "entropy_old": entropy_old[h].item(),
                        "entropy_new": entropy_new[h].item(),
                        "effective_tokens_old": entropy_old[h].exp().item(),
                        "effective_tokens_new": entropy_new[h].exp().item(),
                        "total_variation": tv[h].item(),
                        "old_vs_uniform_tv": uniform_tv[h].item(),
                        "qk_std": qk[:, :, h].std().item(),
                        "old_bias_std": old_bias[..., h].float().std().item(),
                        "new_bias_std": new_bias[..., h].float().std().item(),
                        "fp64_algebra_max_abs": algebra_error,
                    }
                    record["heads"].append(row)
                    if i in (0, 6, 7):
                        row["kernel_entropy_old"] = kernel_entropy[h].item()
                        row["kernel_effective_tokens_old"] = (
                            kernel_entropy[h].exp().item()
                        )
                        row["kernel_old_vs_uniform_tv"] = kernel_uniform[h].item()
                    if abs(row["offset"]) > 1000:
                        print(f"HEAD {json.dumps(row)}", flush=True)
            current = blk(current, cond, pair, *extra)
        record["token_dit_old_finite"] = bool(torch.isfinite(current).all())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
