"""Qualified H100 CUDA/TMA surrounds for square C128/H4/D32 attention.

The prologue preserves bf16 projection rounding but has a slightly different
fp32 LayerNorm reduction order. The epilogue preserves every bf16 rounding point.
Other shapes, layouts, stacks and LayerNorm modes retain their existing routes.
"""
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
_TABLES = {}
_LENGTHS = (384, 768, 1024)


def enabled(cfg, device):
    backend = os.environ.get("FPF_TRIATT_BACKEND", (cfg or {}).get("backend", "triton"))
    if backend != "cuda_tma" or device.type != "cuda":
        return False
    return (str(torch.__version__) == "2.10.0+cu128"
            and sys.implementation.cache_tag == "cpython-310"
            and torch.cuda.get_device_capability(device) == (9, 0))


def _load():
    global _MODULE
    with _LOCK:
        if _MODULE is None:
            manifest = json.loads((_ROOT / "manifest.json").read_text())
            for name, expected in manifest["sha256"].items():
                actual = hashlib.sha256((_ROOT / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"CUDA/TMA artifact mismatch: {name}; rebuild the surround extension")
            spec = importlib.util.spec_from_file_location("triattn_native_surround", _ROOT / "triattn_native_surround.so")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODULE = module
        return _MODULE


def _sigmoid_table(device):
    # A universal function table, independent of weights and input data. BF16 has
    # only 65536 bit patterns. Compute with the same fp32 sigmoid rounding as the
    # reference, then use paired native bf16 multiplication in the CUDA kernel.
    with _LOCK:
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index not in _TABLES:
            values = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
            table = torch.sigmoid(values.float()).to(torch.bfloat16)
            # One-time initialization must be visible to subsequent CUDA streams.
            torch.cuda.synchronize(device)
            _TABLES[index] = table
        return _TABLES[index]


def try_prologue(z, cache, *, ending, ln_mode, write_x, fma_flags, cfg):
    if not enabled(cfg, z.device):
        return None
    if (ln_mode != "fused" or write_x or not fma_flags[2]
            or (cache["C"], cache["H"], cache["D"]) != (128, 4, 32)
            or z.dtype != torch.bfloat16 or not z.is_contiguous()
            or z.dim() != 3 or z.shape[0] != z.shape[1] or z.shape[0] not in _LENGTHS
            or cfg.get("qkv_pair", False)):
        return None
    length = z.shape[0]
    q, k, v = [torch.empty((length, 4, length, 32), device=z.device, dtype=z.dtype) for _ in range(3)]
    gate = torch.empty((length, length, 128), device=z.device, dtype=z.dtype)
    bias = torch.empty((4, length, length), device=z.device, dtype=torch.float32)
    _load().prologue(z, cache["wqkvg"], cache["wb"], cache["lnw"], cache["lnb"],
                     q, k, v, gate, bias, cache["eps"], ending)
    return q, k, v, gate, bias


def try_epilogue(o, g, weight, z, *, ending, residual, out, o_layout, cfg):
    if not residual or z is None or o_layout != "ihjd" or not enabled(cfg, o.device):
        return False, None
    target = z if out is None else out
    length = z.shape[-2]
    if (length not in _LENGTHS or tuple(z.shape[-3:]) != (length, length, 128)
            or z.numel() != length * length * 128 or weight.shape != (128, 128)
            or any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (g, weight, z, target))
            or o.dtype != torch.bfloat16 or o.dim() < 4
            or tuple(o.shape[-4:]) != (length, 4, length, 32)
            or o.numel() != length * length * 128 or o.stride(-1) != 1
            or g.numel() != length * length * 128 or target.shape != z.shape):
        return False, None
    _load().epilogue(o, g, weight, z, target, _sigmoid_table(z.device), ending)
    return True, None if out is None else out
