"""MiniWorld."""

# Keep in sync with pyproject.toml. See docs/pipeline.md for what a version means:
# 1.0.x = no architecture change, 1.1.x = architecture change.
__version__ = "1.0.0"


def engine_jit_root():
    """This environment's cache of JIT-built engine kernels (a ``Path``), or ``None`` without an engine.

    The engine caches them under ``$XDG_CACHE_HOME/miniworld_engine_jit`` whatever interpreter built them, so two environments
    (another Python, another torch) on one account load each other's ``.so`` and fail with "Python version mismatch". The cache is
    keyed by the interpreter prefix here unless the run set ``engine_jit_root`` itself.
    """
    import hashlib
    import os
    import sys
    from pathlib import Path

    try:
        from miniworld_engine import settings
    except ImportError:  # no engine installed: nothing builds kernels
        return None
    if settings.current().engine_jit_root:
        return Path(settings.current().engine_jit_root)
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    tag = hashlib.sha1(sys.prefix.encode()).hexdigest()[:10]  # noqa: S324 - a directory name, not security
    py = f"py{sys.version_info.major}{sys.version_info.minor}"
    return base / "miniworld_engine_jit" / f"{py}-{tag}"


def _separate_engine_jit_cache_per_environment() -> None:
    """Point the engine and torch's extension builder at this environment's own cache.

    Part of the engine's extensions load without a ``build_directory`` and therefore build under torch's
    ``TORCH_EXTENSIONS_DIR`` (default ``~/.cache/torch_extensions/py310_cu129``), which every environment with the same Python and
    CUDA shares: two checkouts on different engine pins rebuilt each other's extensions on every switch. Unless the run set it,
    it goes next to the engine's own cache. torch reads the variable at every build, so setting it here, at the first
    ``import miniworld``, is early enough for every extension the engine loads later.
    """
    import os

    from miniworld_engine import settings

    root = engine_jit_root()
    if root is None:
        return
    if not settings.current().engine_jit_root:
        settings.configure(engine_jit_root=str(root))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(root / "torch_extensions"))


try:
    _separate_engine_jit_cache_per_environment()
except ImportError:  # no engine installed
    pass
