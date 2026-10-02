"""B200 TriMul, D64 native path: B samples in one call against B calls of one sample, inference and training, as CUDA graphs.

The batched call is what ``AF3TemplateEmbedder`` makes when it folds its templates into the batch. Needs the engine's
``integrations.trimul_b200.MAX_BATCH`` (engine main from ``75693b41``) on a B200.

    python -m benchmarks.engine_ops.trimul_batch
    python -m benchmarks.engine_ops.trimul_batch --lengths 384 --batches 4

Measured on one B200 (batch call vs the B calls): 1.1-2.1x in inference and 1.2-2.6x in training at L128-384, B 2 and 4; the incoming
direction matches the outgoing one to within 2%. Directions: 0 bidirectional, 1 outgoing, 2 incoming.
"""

from __future__ import annotations

import argparse

import torch

from benchmarks.common import apply_engine_settings, graph_ms, header

D = 64


def make_leaves(direction: int, x: torch.Tensor, seed: int = 0) -> list[torch.Tensor]:
    """Input, the six bf16 weights and the four fp32 LayerNorm vectors of one TriMul, at D64."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    bf16 = torch.bfloat16
    planes = (4 if direction == 0 else 2) * D
    half = planes // 2

    def normal(*shape, scale):
        return torch.randn(*shape, device="cuda", generator=generator) * scale

    wl, wlg, wr, wrg = (normal(half, D, scale=D**-0.5).to(bf16) for _ in range(4))
    wg = normal(D, D, scale=D**-0.5).to(bf16)
    wp = normal(D, half, scale=half**-0.5).to(bf16)
    gamma_in, beta_in = 1 + 0.1 * normal(D, scale=1.0), 0.1 * normal(D, scale=1.0)
    gamma_out, beta_out = 1 + 0.1 * normal(half, scale=1.0), 0.1 * normal(half, scale=1.0)
    return [x, wl, wlg, wr, wrg, wg, wp, gamma_in.float(), beta_in.float(), gamma_out.float(), beta_out.float()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--engine-setting", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    from miniworld_engine.kernels.trimul_inproj.cuda import b200_infer, b200_train

    apply_engine_settings(args.engine_setting)
    print(header("trimul_batch"), flush=True)
    for length in args.lengths:
        for batch in args.batches:
            for direction in (0, 1, 2):
                torch.manual_seed(1)
                x = torch.randn(batch, length, length, D, device="cuda", dtype=torch.bfloat16)
                leaves = make_leaves(direction, x)
                mask = torch.rand(batch, length, device="cuda") > 0.15
                dropscale = (torch.rand(batch, length, D, device="cuda") > 0.25).to(torch.bfloat16) / 0.75
                flat_mask, flat_scale = mask.reshape(-1), dropscale.reshape(-1, D).contiguous()
                train_leaves = [t.detach().clone().requires_grad_(t.is_floating_point()) for t in leaves]
                upstream = torch.randn(batch, length, length, D, device="cuda", dtype=torch.bfloat16)

                def inference_batch():
                    return b200_infer.inference([x, *leaves[1:]], flat_mask, flat_scale, direction)

                def inference_loop():
                    return [b200_infer.inference([x[i : i + 1], *leaves[1:]], mask[i], dropscale[i].contiguous(), direction) for i in range(batch)]

                def training_batch():
                    out = b200_train.trimul_train([x.detach().clone().requires_grad_(), *train_leaves[1:]], flat_mask, flat_scale, direction)
                    out.backward(upstream)

                def training_loop():
                    for i in range(batch):
                        out = b200_train.trimul_train(
                            [x[i : i + 1].detach().clone().requires_grad_(), *train_leaves[1:]], mask[i], dropscale[i].contiguous(), direction,
                        )
                        out.backward(upstream[i : i + 1])

                try:
                    inf_batch, inf_loop = graph_ms(inference_batch), graph_ms(inference_loop)
                    trn_batch, trn_loop = graph_ms(training_batch), graph_ms(training_loop)
                    print(f"TRIMUL_BATCH L{length} B{batch} dir{direction}: inference batch {inf_batch:.3f} vs {batch} calls {inf_loop:.3f} ms "
                          f"({inf_loop / inf_batch:.2f}x) | training forward+backward batch {trn_batch:.3f} vs {batch} calls {trn_loop:.3f} ms "
                          f"({trn_loop / trn_batch:.2f}x)", flush=True)
                except Exception as error:  # noqa: BLE001  (an unsupported shape or card is an answer, print it)
                    print(f"TRIMUL_BATCH L{length} B{batch} dir{direction}: {type(error).__name__}: {str(error)[:120]}", flush=True)


if __name__ == "__main__":
    main()
