"""A hard "no Triton" switch for a training process: any Triton kernel launch, or any inductor compile, is an error.

``train.forbid_triton: true`` installs this once, before the model is built. What it closes:

* every launch of a Triton kernel from Python (``triton.runtime.jit.JITFunction.run``, which the engine's Triton kernels and their
  autotuners go through) raises :class:`TritonForbidden` naming the kernel. The engine's modules route a shape to the hand-written CUDA
  kernels when it qualifies and fall back to the Triton family otherwise; with this switch such a fallback stops the run
  instead of silently running Triton.
* inductor (the ``torch.compile`` backend: its GPU kernels are Triton) raises on compile, ``torch.compile`` itself is left
  alone so decorators still import. Functions of this repo that are ``torch.compile`` decorated check :func:`active` and run eagerly.

Nothing here makes a kernel faster or slower; it only forbids. Fast paths that remain: the engine CUDA kernels, cuBLAS / cutlass,
flash attention, plain ATen kernels (and CUDA graphs around them).
"""

from __future__ import annotations

_ACTIVE = False
_ORIGINALS: dict = {}


class TritonForbidden(RuntimeError):
    """A Triton kernel launch or an inductor compile happened while ``forbid_triton`` is on."""


def active() -> bool:
    """True once :func:`install` ran in this process."""
    return _ACTIVE


def install() -> None:
    """Forbid Triton launches and inductor compiles in this process (idempotent)."""
    global _ACTIVE  # noqa: PLW0603
    if _ACTIVE:
        return
    import triton.runtime.jit as triton_jit
    from torch._inductor import compile_fx as inductor

    original_run = triton_jit.JITFunction.run

    def forbidden_run(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        name = getattr(self, "__name__", None) or getattr(getattr(self, "fn", None), "__name__", "?")
        msg = f"Triton kernel '{name}' launched while train.forbid_triton is on"
        raise TritonForbidden(msg)

    def forbidden_compile(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        msg = "inductor compile (its GPU kernels are Triton) while train.forbid_triton is on"
        raise TritonForbidden(msg)

    _ORIGINALS.update(run=original_run, compile_fx=inductor.compile_fx)
    triton_jit.JITFunction.run = forbidden_run
    inductor.compile_fx = forbidden_compile
    _ACTIVE = True


def uninstall() -> None:
    """Undo :func:`install` (tests only: a training process installs it once and keeps it)."""
    global _ACTIVE  # noqa: PLW0603
    if not _ACTIVE:
        return
    import triton.runtime.jit as triton_jit
    from torch._inductor import compile_fx as inductor

    triton_jit.JITFunction.run = _ORIGINALS["run"]
    inductor.compile_fx = _ORIGINALS["compile_fx"]
    _ORIGINALS.clear()
    _ACTIVE = False
