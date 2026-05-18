"""Distogram-only training variant of MiniWorld (no diffusion module)."""

from .client import Client
from .model import Model

__all__ = [
    "Client",
    "Model",
]
