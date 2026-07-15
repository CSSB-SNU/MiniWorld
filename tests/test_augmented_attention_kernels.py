"""Temporary accuracy verification script for augmented attention kernels.

Compares the memory-efficient and compute-efficient Triton kernels against the
PyTorch reference across several shapes and dtypes (forward + backward).

Run:
    python tests/test_augmented_attention_kernels.py
"""

from __future__ import annotations

import itertools
import sys
import traceback

import torch
import torch.nn.functional as F

from team_gm.modules.kernels import (
    triton_augmented_attention_pair_bias,
    triton_augmented_attention_pair_bias_compute_efficient,
)


# (A, B, L, H, D)  -- HEAD_DIM (D) must be <= 64 for the kernels
SHAPES = [
    (1, 1, 32, 4, 32),
    (1, 1, 128, 4, 64),
    (4, 1, 128, 8, 32),
    (8, 1, 256, 8, 64),
    (1, 2, 384, 4, 32),
    (4, 1, 512, 16, 32),
    (8, 1, 1024, 4, 64),
]

DTYPES = [
    # (dtype, atol_fwd, atol_bwd, rtol)
    # Triton flash-attention-style kernels use reordered reductions and exp2,
    # so they don't bit-match naive PyTorch even in fp32 — set tolerances to
    # the actual achievable accuracy of the kernel, not float-precision ideal.
    (torch.float32, 1e-2, 2e-2, 1e-2),
    (torch.bfloat16, 5e-2, 8e-2, 5e-2),
]

MASK_MODES = ["none", "random"]


def pytorch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reference: matches AugmentedAttentionPairBias._kernel_attention_pair_bias."""
    scale = q.shape[-1] ** -0.5
    q = q * scale
    attn = torch.einsum("abihd,abjhd->abhij", q, k)
    bias_p = bias.permute(0, 3, 1, 2).contiguous()  # (B, H, L, L)
    attn = attn + bias_p[None]
    if mask is not None:
        attn = attn.masked_fill(~mask[:, :, None, None, :], float("-inf"))
    attn = F.softmax(attn, dim=-1)
    return torch.einsum("abhij,abjhd->abihd", attn, v)


def make_inputs(
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
    mask_mode: str,
    seed: int = 0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None,
]:
    A, B, L, H, D = shape
    g = torch.Generator(device="cuda").manual_seed(seed)

    # Always create master inputs in fp32; cast per-impl below
    q = torch.randn(A, B, L, H, D, device="cuda", generator=g, dtype=torch.float32)
    k = torch.randn(A, B, L, H, D, device="cuda", generator=g, dtype=torch.float32)
    v = torch.randn(A, B, L, H, D, device="cuda", generator=g, dtype=torch.float32)
    bias = torch.randn(B, L, L, H, device="cuda", generator=g, dtype=torch.float32)

    if mask_mode == "none":
        mask = None
    else:
        # Keep at least one valid key per row to avoid all -inf rows.
        mask = torch.rand(A, B, L, device="cuda", generator=g) > 0.2
        mask[..., 0] = True

    def cast(t):
        return t.detach().to(dtype).requires_grad_()

    return cast(q), cast(k), cast(v), cast(bias), mask


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def compare(
    name: str,
    out_ref: torch.Tensor,
    out_kernel: torch.Tensor,
    atol: float,
    rtol: float,
    mask: torch.Tensor | None,
) -> tuple[bool, float]:
    """Compare under mask if present (masked positions are undefined for kernel)."""
    if mask is not None:
        # mask: (A, B, L) -> broadcast over (A, B, L, H, D)
        m = mask[..., None, None]
        out_ref = out_ref * m
        out_kernel = out_kernel * m
    diff = max_abs_diff(out_ref, out_kernel)
    tol = atol + rtol * out_ref.float().abs().max().item()
    ok = diff <= tol
    return ok, diff


def run_one(
    kernel_name: str,
    kernel_fn,
    shape,
    dtype,
    atol_fwd,
    atol_bwd,
    rtol,
    mask_mode: str,
) -> dict:
    A, B, L, H, D = shape
    # Reference inputs
    q_ref, k_ref, v_ref, bias_ref, mask = make_inputs(shape, dtype, mask_mode)
    # Kernel inputs (same values, independent grads)
    q_ker, k_ker, v_ker, bias_ker, _ = make_inputs(shape, dtype, mask_mode)

    # Forward
    out_ref = pytorch_reference(q_ref, k_ref, v_ref, bias_ref, mask)
    out_ker = kernel_fn(q_ker, k_ker, v_ker, bias_ker, mask)

    fwd_ok, fwd_diff = compare(
        "fwd", out_ref, out_ker, atol_fwd, rtol, mask,
    )

    # Backward
    g = torch.Generator(device="cuda").manual_seed(123)
    dy = torch.randn(
        A, B, L, H, D, device="cuda", generator=g, dtype=out_ref.dtype,
    )
    if mask is not None:
        dy = dy * mask[..., None, None]

    out_ref.backward(dy)
    out_ker.backward(dy.clone())

    results = {
        "kernel": kernel_name,
        "shape": shape,
        "dtype": str(dtype).replace("torch.", ""),
        "mask": mask_mode,
        "fwd_ok": fwd_ok,
        "fwd_diff": fwd_diff,
    }

    for label, t_ref, t_ker in [
        ("dQ", q_ref.grad, q_ker.grad),
        ("dK", k_ref.grad, k_ker.grad),
        ("dV", v_ref.grad, v_ker.grad),
        ("dBias", bias_ref.grad, bias_ker.grad),
    ]:
        # dBias has no mask broadcast; pass None
        m = mask if label in ("dQ", "dK", "dV") else None
        ok, diff = compare(label, t_ref, t_ker, atol_bwd, rtol, m)
        results[f"{label}_ok"] = ok
        results[f"{label}_diff"] = diff

    return results


def fmt_row(r: dict) -> str:
    def tag(ok):
        return "OK " if ok else "FAIL"

    return (
        f"[{tag(r['fwd_ok'])}] fwd={r['fwd_diff']:.2e}  "
        f"[{tag(r['dQ_ok'])}] dQ={r['dQ_diff']:.2e}  "
        f"[{tag(r['dK_ok'])}] dK={r['dK_diff']:.2e}  "
        f"[{tag(r['dV_ok'])}] dV={r['dV_diff']:.2e}  "
        f"[{tag(r['dBias_ok'])}] dBias={r['dBias_diff']:.2e}  "
        f"| {r['kernel']:>17s} | shape={r['shape']} {r['dtype']} mask={r['mask']}"
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        return 2

    kernels = {
        "memory_efficient": triton_augmented_attention_pair_bias,
        "compute_efficient": triton_augmented_attention_pair_bias_compute_efficient,
    }

    n_total = 0
    n_failed = 0
    rows: list[dict] = []
    errors: list[str] = []

    for shape, (dtype, atol_fwd, atol_bwd, rtol), mask_mode, (kname, kfn) in (
        itertools.product(SHAPES, DTYPES, MASK_MODES, kernels.items())
    ):
        n_total += 1
        try:
            r = run_one(
                kname, kfn, shape, dtype, atol_fwd, atol_bwd, rtol, mask_mode,
            )
            rows.append(r)
            checks = [
                r["fwd_ok"], r["dQ_ok"], r["dK_ok"], r["dV_ok"], r["dBias_ok"],
            ]
            if not all(checks):
                n_failed += 1
            print(fmt_row(r))
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            tb = traceback.format_exc()
            errors.append(
                f"ERROR | {kname} shape={shape} dtype={dtype} mask={mask_mode}\n{tb}"
            )
            print(f"[ERR ] {kname} shape={shape} dtype={dtype} mask={mask_mode}: {e}")
        finally:
            torch.cuda.empty_cache()

    print()
    print(f"{n_total - n_failed}/{n_total} configurations passed.")
    if errors:
        print("\n--- Errors ---")
        for e in errors:
            print(e)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
