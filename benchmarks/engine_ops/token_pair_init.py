"""``kernels/token_pair_init`` (fused token-pair initialisation, sm_100a) against its dense fp32 reference, as CUDA graphs.

Forward and forward + backward at several lengths, d_pair 128, batch 1. Needs an engine that has ``kernels.token_pair_init`` (engine
main from ``75693b41``) on a B200; elsewhere the op refuses and this prints why.

    python -m benchmarks.engine_ops.token_pair_init

Measured on one B200 (ms, ours / reference): L384 forward 0.045 / 0.549, forward + backward 0.151 / 0.810.
"""

from __future__ import annotations

import argparse

import torch

from benchmarks.common import apply_engine_settings, graph_ms, header

DEVICE, D_PAIR = "cuda", 128


def make_case(batch: int, length: int, *, seed: int = 0, r_max: int = 32, s_max: int = 2):
    """Left / right pair streams, the two projection weights (requiring grad), the token ids and a sparse bond matrix."""
    generator = torch.Generator().manual_seed(seed)
    n_rel = 2 * (2 * r_max + 2) + (2 * s_max + 2) + 1

    def ids(high, dtype=torch.int64):
        return torch.randint(0, high, (batch, length), generator=generator).to(DEVICE, dtype)

    token_ids = (ids(3), ids(90), ids(60), ids(2), ids(6, torch.int32))
    bond = (torch.rand(batch, length, length, generator=generator) < 0.05).to(DEVICE)
    leaves = [
        torch.randn(batch, length, D_PAIR, generator=generator).to(DEVICE).requires_grad_(),
        torch.randn(batch, length, D_PAIR, generator=generator).to(DEVICE).requires_grad_(),
        (torch.randn(D_PAIR, n_rel, generator=generator) * 0.3).to(DEVICE).requires_grad_(),
        (torch.randn(D_PAIR, 2, generator=generator) * 0.3).to(DEVICE).requires_grad_(),
    ]
    return leaves, token_ids, bond


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 384, 768])
    parser.add_argument("--engine-setting", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    from miniworld_engine.kernels.token_pair_init import token_pair_init, token_pair_init_reference

    apply_engine_settings(args.engine_setting)
    print(header("token_pair_init"), flush=True)
    for length in args.lengths:
        leaves, token_ids, bond = make_case(1, length)
        upstream = torch.randn(1, length, length, D_PAIR, device=DEVICE)
        times = {}
        for name, op in (("ours", token_pair_init), ("reference", token_pair_init_reference)):

            def forward(op=op):
                return op(*leaves, *token_ids, bond)

            def forward_backward(op=op):
                torch.autograd.backward(op(*leaves, *token_ids, bond), upstream)

            try:
                times[name] = (graph_ms(forward), graph_ms(forward_backward))
            except Exception as error:  # noqa: BLE001  (a refusal or an unsupported card is an answer, print it)
                times[name] = (float("nan"), float("nan"))
                print(f"TOKEN_PAIR_INIT L{length} {name}: {type(error).__name__}: {str(error)[:120]}")
        (fwd, fwd_bwd), (ref_fwd, ref_fwd_bwd) = times["ours"], times["reference"]
        print(f"TOKEN_PAIR_INIT L{length}: forward {fwd:.4f} / {ref_fwd:.4f} ms ({ref_fwd / fwd:.1f}x) | "
              f"forward+backward {fwd_bwd:.4f} / {ref_fwd_bwd:.4f} ms ({ref_fwd_bwd / fwd_bwd:.1f}x)   [ours / reference]", flush=True)


if __name__ == "__main__":
    main()
