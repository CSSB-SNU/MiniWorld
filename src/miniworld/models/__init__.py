"""Models for MiniWorld."""

from .default_client import Client as DefaultClient
from .embedding_client import Client as EmbeddingClient

__all__ = [
    "DefaultClient",
    "EmbeddingClient",
]
