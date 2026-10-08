"""A hard "no engine Triton" switch for a training process: a Triton kernel the engine (or any user code) launches is an error.

``train.forbid_triton: true`` installs this once, before the model is built. The engine serves a shape with its hand-written CUDA
kernels when it qualifies and falls back to its Triton family otherwise; this switch turns such a fallback into a stop instead of a
silent slow path. What it closes:

* every launch of a Triton kernel from Python (``triton.runtime.jit.JITFunction.run``: the engine's kernels and their autotuners)
  raises :class:`TritonForbidden` naming the kernel;
* the same kernels inside a ``torch.compile`` region: dynamo registers a user Triton kernel in its side table when it traces the call, and
  that registration raises.

What it does NOT touch: the kernels inductor generates for ``torch.compile`` itself (``triton_*`` in a profile) are allowed, and so
are the engine's CUDA kernels, cuBLAS / cutlass, flash attention and ATen.
"""

from __future__ import annotations

_ACTIVE = False
_ORIGINALS: dict = {}


class TritonForbidden(RuntimeError):
    """An engine / user Triton kernel was launched (or traced into a compiled graph) while ``forbid_triton`` is on."""


def active() -> bool:
    """True once :func:`install` ran in this process."""
    return _ACTIVE


def install() -> None:
    """Forbid engine / user Triton kernels in this process (idempotent)."""
    global _ACTIVE  # noqa: PLW0603
    if _ACTIVE:
        return
    import triton.runtime.jit as triton_jit
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    original_run = triton_jit.JITFunction.run
    original_add = kernel_side_table.add_kernel

    def name_of(kernel) -> str:  # noqa: ANN001
        return getattr(kernel, "__name__", None) or getattr(getattr(kernel, "fn", None), "__name__", "?")

    def forbidden_run(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        msg = f"Triton kernel '{name_of(self)}' launched while train.forbid_triton is on"
        raise TritonForbidden(msg)

    def forbidden_add(kernel):  # noqa: ANN001, ANN202
        msg = f"Triton kernel '{name_of(kernel)}' traced into a compiled graph while train.forbid_triton is on"
        raise TritonForbidden(msg)

    _ORIGINALS.update(run=original_run, add_kernel=original_add)
    triton_jit.JITFunction.run = forbidden_run
    kernel_side_table.add_kernel = forbidden_add
    _ACTIVE = True


def uninstall() -> None:
    """Undo :func:`install` (tests only: a training process installs it once and keeps it)."""
    global _ACTIVE  # noqa: PLW0603
    if not _ACTIVE:
        return
    import triton.runtime.jit as triton_jit
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    triton_jit.JITFunction.run = _ORIGINALS["run"]
    kernel_side_table.add_kernel = _ORIGINALS["add_kernel"]
    _ORIGINALS.clear()
    _ACTIVE = False
