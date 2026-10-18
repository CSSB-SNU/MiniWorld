"""MiniWorld."""

# Keep in sync with pyproject.toml. See docs/pipeline.md for what a version means:
# 1.0.x = no architecture change, 1.1.x = architecture change.
__version__ = "1.0.0"


def _separate_engine_jit_cache_per_environment() -> None:
    """Give each Python environment its own cache of JIT-built engine kernels.

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
        return
    if settings.current().engine_jit_root:
        return
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    tag = hashlib.sha1(sys.prefix.encode()).hexdigest()[:10]  # noqa: S324 - a directory name, not security
    py = f"py{sys.version_info.major}{sys.version_info.minor}"
    settings.configure(engine_jit_root=str(base / "miniworld_engine_jit" / f"{py}-{tag}"))


_separate_engine_jit_cache_per_environment()
