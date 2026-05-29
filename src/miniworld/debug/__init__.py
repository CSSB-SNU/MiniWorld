"""Debug instrumentation for MiniWorld inference.

Not imported by default — only pulled in by scripts/diffuse_debug.py.
"""
from miniworld.debug.diffusion_capture import DiffusionCapture

__all__ = ["DiffusionCapture"]
