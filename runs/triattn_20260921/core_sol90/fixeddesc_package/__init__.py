"""Bit-preserving broadcast-mask M1 core for qualified square H100 attention."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import threading
import torch

_ROOT = Path(__file__).resolve().parent
_MODULE = None
_LOCK = threading.RLock()


def _load():
    global _MODULE
    with _LOCK:
        if _MODULE is None:
            manifest = json.loads((_ROOT / "manifest.json").read_text())
            for name, expected in manifest["sha256"].items():
                actual = hashlib.sha256((_ROOT / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Broadcast core artifact mismatch: {name}; rebuild the extension")
            spec = importlib.util.spec_from_file_location("triattn_broadcast", _ROOT / "triattn_broadcast.so")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODULE = module
        return _MODULE


def select_core(q, mask, flags=0, trace=None):
    mode = os.environ.get("FPF_TRIATT_CORE_BROADCAST", "auto")
    if mode not in ("auto", "on", "off"):
        raise ValueError("FPF_TRIATT_CORE_BROADCAST must be auto, on or off")
    if mode == "off" or flags != 0 or trace is not None:
        return None
    if (q.device.type != "cuda" or q.dtype != torch.bfloat16 or q.dim() != 5
            or q.shape[0] != 1 or q.shape[1] != q.shape[3]
            or q.shape[2] != 4 or q.shape[3] not in (768, 1024) or q.shape[4] != 32
            or str(torch.__version__) != "2.10.0+cu128"
            or sys.implementation.cache_tag != "cpython-310"
            or torch.cuda.get_device_capability(q.device) != (9, 0)):
        return None
    if mask is not None and (mask.dtype != torch.bool or mask.dim() != 5
            or tuple(mask.shape) != (1, q.shape[1], 1, 1, q.shape[3])
            or mask.stride(1) != 0 or mask.device != q.device):
        return None
    return _load()
