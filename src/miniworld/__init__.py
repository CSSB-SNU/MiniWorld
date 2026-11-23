"""MiniWorld."""

# Keep in sync with pyproject.toml. See docs/pipeline.md for what a version means:
# 1.0.x = no architecture change, 1.1.x = architecture change.
__version__ = "1.0.0"


def engine_jit_root():
    """This environment's directory for JIT-built extensions (a ``Path``), or ``None`` without an engine.

    miniworld-engine 3.x caches its JIT-built kernels under ``$XDG_CACHE_HOME/miniworld_engine_jit`` (no ``settings`` to relocate it;
    prebuilt kernels ship in the wheel and are looked up first). Torch-extension ``.so`` files, which depend on the interpreter and
    torch, must not be shared between environments of one account ("Python version mismatch"), so they live in a sub-directory
    keyed by the interpreter prefix (see :func:`_separate_engine_jit_cache_per_environment`).
    """
    import hashlib
    import sys

    try:
        from miniworld_engine.kernels._nvcc import engine_jit_root as _engine_jit_root
    except ImportError:  # no engine installed: nothing builds kernels
        return None
    tag = hashlib.sha1(sys.prefix.encode()).hexdigest()[:10]  # noqa: S324 - a directory name, not security
    py = f"py{sys.version_info.major}{sys.version_info.minor}"
    return _engine_jit_root() / f"{py}-{tag}"


def _separate_engine_jit_cache_per_environment() -> None:
    """Point torch's extension builder at this environment's own cache (unless the run set ``TORCH_EXTENSIONS_DIR``).

    Part of the engine's extensions load without a ``build_directory`` and therefore build under torch's
    ``TORCH_EXTENSIONS_DIR`` (default ``~/.cache/torch_extensions/py312_cu129``), which every environment with the same Python and
    CUDA shares: two checkouts on different engine pins rebuilt each other's extensions on every switch. torch reads the variable at
    every build, so setting it here, at the first ``import miniworld``, is early enough for every extension the engine loads later.
    """
    import os

    root = engine_jit_root()
    if root is None:
        return
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(root / "torch_extensions"))


try:
    _separate_engine_jit_cache_per_environment()
except ImportError:  # no engine installed
    pass
