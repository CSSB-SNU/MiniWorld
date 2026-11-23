"""Backend names shared by the blocks: ``ImplementationType`` (the value MiniWorld configs spell) and its translation to the engine's
backend strings.

Moved here from ``foldforge.layers.implementation`` (FoldForge dropped it when it ported to miniworld-engine 2.5), and retargeted for
miniworld-engine 3.x, whose modules take ``backend="pytorch" | "cuda" | "triton" | "cueq" | "anthropic"`` (a ``BackendModule``) instead of the
removed ``modules.exceptions.ImplementationType``. The config value is unchanged: ``MINIWORLD_ENGINE`` (``"miniworld_engine"``) selects the
engine's ``"cuda"`` backend. Unlike the engine 2.x ``miniworld`` auto-routing, the 3.x ``"cuda"`` backend does not fall back to Triton or PyTorch
by itself: a call it cannot serve raises ``BackendUnsupported``.
"""

from enum import Enum


class ImplementationType(str, Enum):
    """Implementation types for certain modules."""

    PYTORCH = "pytorch"
    MINIWORLD_ENGINE = "miniworld_engine"
    CUEQUIVARIANCE = "cuequivariance"
    ANTHROPIC = "anthropic"
    # Source compatibility for old consumers; iteration uses the canonical member.
    MINIWORLD_KERNELS = MINIWORLD_ENGINE

    @classmethod
    def _missing_(cls, value):
        if value == "miniworld_kernels":
            return cls.MINIWORLD_ENGINE
        return None


class InvalidImplementationError(ValueError):
    """Raised when an invalid implementation type is specified."""

    def __init__(self, implementation: str) -> None:
        valid = ", ".join(f"'{impl.value}'" for impl in ImplementationType)
        msg = f"Invalid implementation: '{implementation}'. Valid options: {valid}."
        super().__init__(msg)


_ENGINE_BACKEND = {
    ImplementationType.PYTORCH: "pytorch",
    ImplementationType.MINIWORLD_ENGINE: "cuda",
    ImplementationType.CUEQUIVARIANCE: "cueq",
    ImplementationType.ANTHROPIC: "anthropic",
}


def to_engine_impl(implementation: ImplementationType | str) -> str:
    """The engine 3.x ``backend`` string of a MiniWorld ``ImplementationType`` (``MINIWORLD_ENGINE`` -> ``"cuda"``)."""
    return _ENGINE_BACKEND[ImplementationType(implementation)]
