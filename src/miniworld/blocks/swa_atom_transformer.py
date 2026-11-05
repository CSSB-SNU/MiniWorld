"""ESMFold2 ``SWAAtomTransformer`` (Algorithm 8): the engine's ``modules.swa_atom_transformer``.

DiT adaLN-Zero blocks with sliding-window 3D-RoPE attention, a SwiGLU FFN and no atom-pair tensor. The blocks, the FFN, the
fused path (``fused_triton``) and the kernel routing are the engine module's (the same parameters and names: strict loads of
MiniWorld checkpoints are unchanged). What stays here is a constructor adapter: ``Config.implementation`` takes the
architecture ``ImplementationType`` (``miniworld_engine``), which the engine module receives as its own enum (``miniworld``).

The magnitude-preserving options (``magnitude_preserving`` / ``mp_full`` / ``mp_residual``) and ``use_rotation`` are no longer
implemented here (no live config sets them): the Config keeps the fields, and the engine refuses them when switched on. The
fork with them is in git history (``migrate/dedup``).
"""

from __future__ import annotations

from foldforge.layers.implementation import ImplementationType, to_engine_impl
from miniworld_engine.modules.exceptions import (
    ImplementationType as EngineImplementation,
)
from miniworld_engine.modules.swa_atom_transformer import module as _engine

SwiGLUFFN = _engine.SwiGLUFFN
SWAAtomBlock = _engine.SWAAtomBlock

__all__ = ["SWAAtomBlock", "SWAAtomTransformer", "SwiGLUFFN"]


def _engine_implementation(
    implementation: ImplementationType | EngineImplementation | str | None,
) -> EngineImplementation | None:
    if implementation is None or isinstance(implementation, EngineImplementation):
        return implementation
    return to_engine_impl(implementation)


class SWAAtomTransformer(_engine.SWAAtomTransformer):
    """The engine's ``SWAAtomTransformer``, configured with the architecture ``ImplementationType``."""

    class Config(_engine.SWAAtomTransformer.Config):
        """The engine Config; ``implementation`` is the architecture enum (``RoPESWAAF3Transformer`` shares this Config)."""

        implementation: ImplementationType | None = None

    def __init__(self, config: _engine.SWAAtomTransformer.Config) -> None:
        engine_config = _engine.SWAAtomTransformer.Config(
            **{
                **dict(config),
                "implementation": _engine_implementation(config.implementation),
            }
        )
        super().__init__(engine_config)
