"""Explicit process-local dQ experiment. Use around eager training or graph capture.

The model keeps its existing training forward and other backward kernels.
This single-threaded benchmark context restores the installed extension on exit;
captured CUDA graphs retain the native launches chosen during capture.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        folder = _ROOT.parent / 'dq/reuse_qdo'
        build = json.loads((folder / 'build-ready.json').read_text())
        for name, digest in build.items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
        name = (folder / 'module-name.txt').read_text().strip()
        spec = importlib.util.spec_from_file_location(name, folder / 'build/triattn_dq.so')
        _EXTENSION = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_EXTENSION)
    return _EXTENSION


@contextmanager
def use():
    from miniworld_engine.kernels.triangle_attention.cuda import dq_backward
    baseline = dq_backward.extension()
    candidate = extension()
    dq_backward._EXT = candidate
    try:
        yield candidate
    finally:
        dq_backward._EXT = baseline
