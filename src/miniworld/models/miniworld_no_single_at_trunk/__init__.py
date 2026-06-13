"""MiniWorld AF3-like EDM variant whose trunk carries NO token_single track.

The Pairformer runs pair-only and the diffusion conditioning is built from
``token_single_input`` alone. See ``model.py`` and
``docs/edm_token_single_rank_collapse.md`` for the rationale.
"""

from .client import Client
from .model import Model

__all__ = [
    "Client",
    "Model",
]
