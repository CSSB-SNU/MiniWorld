import os
from beartype import beartype
from jaxtyping import jaxtyped

from .client import BaseClient
from .batch import BaseBatch


_SHOULD_TYPECHECK = os.environ.get("SHOULD_TYPECHECK", "false").lower() == "true"


def typecheck(cls_or_func):
    """Decorator to typecheck a class or function."""
    if _SHOULD_TYPECHECK:
        return jaxtyped(typechecker=beartype)(cls_or_func)
    return cls_or_func


__all__ = [
    "BaseClient",
    "BaseBatch",
    "typecheck",
]
