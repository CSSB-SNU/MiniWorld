"""AF3TemplateEmbedder: the four combinations of templates-as-one-batch and the fused small projections, in speed and accuracy.

The two switches are the module's own (``_batch_templates``, ``_fuse_projections``), so the "off, off" row is the original code path.
Inputs are the test inputs (L384, 4 templates, 3 valid). The training model builds the embedder in bf16 and runs under the "medium"
matmul precision, so ``--dtype bf16 --precision medium`` (the default) is the case that matters.

    python -m benchmarks.template_embedder.variants
    python -m benchmarks.template_embedder.variants --dtype fp32 --precision highest
    python -m benchmarks.template_embedder.variants --randomize --fp32-reference --no-timing   # accuracy against fp32 truth

``--randomize`` gives every matrix weights of a trained-like scale. At initialisation the pair stack's output projections are zero,
so the stack is the identity and a batched stack that mixed up its samples would go unnoticed.

Measured on one B200, bf16, ms (inference / training forward + backward), engine ``5d8bb030`` | engine with batched TriMul:
original 2.27 / 6.11 | 3.22 / 7.81; fused projections 2.16 / 5.57 | 3.05 / 7.23; batch 3.00 / 8.23 | 3.11 / 6.90; both 3.10 / 7.59 |
3.02 / 6.52. Without batched TriMul in the engine the batch switch stays off (``batch_limit`` is 1) because a B > 1 call is slower there.
"""

from __future__ import annotations

import argparse
import itertools

import torch

from benchmarks.common import apply_engine_settings, graph_ms, header, use_repo_paths

use_repo_paths()


def randomize(module: torch.nn.Module, seed: int = 1234) -> None:
    """Give every matrix weights of scale ``fan_in ** -0.5`` (the pair stack is the identity at initialisation)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim >= 2:
                noise = torch.randn(parameter.shape, device=parameter.device, generator=generator)
                parameter.copy_((noise * parameter.shape[-1] ** -0.5).to(parameter.dtype))


def relative(x: torch.Tensor, reference: torch.Tensor) -> float:
    return float((x.float() - reference.float()).norm() / reference.float().norm())


def gradients(module, call, upstream) -> dict[str, torch.Tensor]:
    module.zero_grad(set_to_none=True)
    torch.autograd.backward(call(), upstream)
    return {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--precision", choices=("highest", "high", "medium"), default="medium")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--no-timing", action="store_true", help="accuracy only (use when the GPU is shared)")
    parser.add_argument("--fp32-reference", action="store_true", help="also compare each variant with the same weights in fp32 (bf16 runs)")
    parser.add_argument("--engine-setting", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    from miniworld.modules.template_embedder_af3 import AF3TemplateEmbedder
    from tests.test_af3_template import D_PAIR, make_inputs, ref_loop

    apply_engine_settings(args.engine_setting)
    torch.set_float32_matmul_precision(args.precision)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    embedder = AF3TemplateEmbedder(d_pair=D_PAIR).cuda().to(dtype)
    pair, template, asym, token_mask = make_inputs()
    pair = pair.to(dtype)
    if args.randomize:
        randomize(embedder)
    upstream = torch.randn_like(pair)
    print(header("template"), f"dtype={args.dtype}, randomized={args.randomize}, batch_limit={embedder._batch_limit}", flush=True)  # noqa: SLF001

    def call():
        return embedder(pair, template, asym, token_mask)

    def set_variant(batch: bool, fuse: bool) -> None:
        embedder._batch_templates, embedder._fuse_projections = batch, fuse  # noqa: SLF001

    variants = list(itertools.product((False, True), (False, True)))
    label = lambda batch, fuse: f"batch={int(batch)} fuse={int(fuse)}"  # noqa: E731

    embedder.eval()
    with torch.no_grad():
        set_variant(False, False)
        base = call().clone()
        reference = ref_loop(embedder, pair, template, asym, token_mask) if args.dtype == "fp32" else base
        for variant in variants:
            set_variant(*variant)
            out = call()
            print(f"TEMPLATE accuracy {label(*variant)}: max|out-reference| {(out - reference).abs().max().item():.2e}, "
                  f"relative {relative(out, reference):.2e}, relative to the original path {relative(out, base):.2e}", flush=True)

    if not args.no_timing:
        times = {}
        for variant in variants:
            set_variant(*variant)
            embedder.eval()

            def inference():
                with torch.no_grad():
                    return call()

            infer_ms = graph_ms(inference)
            embedder.train()

            def training():
                torch.autograd.backward(call(), upstream)

            train_ms = graph_ms(training)
            times[variant] = (infer_ms, train_ms)
            print(f"TEMPLATE time {label(*variant)}: inference {infer_ms:.3f} ms, training forward+backward {train_ms:.3f} ms", flush=True)
        base_infer, base_train = times[(False, False)]
        for variant, (infer_ms, train_ms) in times.items():
            print(f"TEMPLATE speedup {label(*variant)}: inference {base_infer / infer_ms:.2f}x, training {base_train / train_ms:.2f}x", flush=True)

    embedder.eval()  # parameter gradients per variant against the original path (eval: no dropout)
    grads = {}
    for variant in variants:
        set_variant(*variant)
        grads[variant] = gradients(embedder, call, upstream)
    original = grads[(False, False)]
    for variant, grad in grads.items():
        worst = max(float((grad[n] - original[n]).norm() / original[n].norm().clamp_min(1e-30)) for n in original)
        print(f"TEMPLATE grads {label(*variant)}: worst per-parameter relative error against the original path {worst:.2e}", flush=True)

    if args.fp32_reference and args.dtype == "bf16":
        truth = AF3TemplateEmbedder(d_pair=D_PAIR).cuda()
        truth.load_state_dict({k: v.float() for k, v in embedder.state_dict().items()})
        truth.eval()
        truth._batch_templates = truth._fuse_projections = False  # noqa: SLF001
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        truth_out = truth(pair.float(), template, asym, token_mask)
        truth_grads = gradients(truth, lambda: truth(pair.float(), template, asym, token_mask), upstream.float())
        torch.set_float32_matmul_precision(previous)
        for variant, grad in grads.items():
            set_variant(*variant)
            with torch.no_grad():
                forward_error = relative(call(), truth_out)
            errors = sorted(relative(grad[n], truth_grads[n]) for n in truth_grads if truth_grads[n].norm() > 0)
            print(f"TEMPLATE vs fp32 truth {label(*variant)}: forward relative {forward_error:.2e}; parameter gradient error "
                  f"worst {errors[-1]:.2e}, median {errors[len(errors) // 2]:.2e}", flush=True)


if __name__ == "__main__":
    main()
