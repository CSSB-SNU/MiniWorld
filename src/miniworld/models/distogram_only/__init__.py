"""Distogram-only training variant of MiniWorld (no diffusion module)."""

from .client import Client
from .model import Model
from .model_mini import MiniModel

__all__ = [
    "Client",
    "MiniModel",
    "Model",
]
