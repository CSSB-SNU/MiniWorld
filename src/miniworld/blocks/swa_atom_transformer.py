"""ESMFold2 ``SWAAtomTransformer`` (Algorithm 8): the engine's ``modules.atom_dit.SWAAtomTransformer`` (a stack of ``AtomDiTMiniWorld`` blocks).

DiT adaLN-Zero blocks with sliding-window 3D-RoPE attention, a SwiGLU FFN and no atom-pair tensor. The blocks, the FFN and the kernel
routing are the engine module's. What stays here is a constructor adapter: ``Config.implementation`` takes the MiniWorld
``ImplementationType`` (``miniworld_engine``), which the engine 3.x module receives as its ``backend`` string (``"cuda"``).

The magnitude-preserving options (``magnitude_preserving`` / ``mp_full`` / ``mp_residual``) and ``use_rotation`` are no longer
implemented here (no live config sets them): the Config keeps the fields, and the engine refuses them when switched on.
"""

from __future__ import annotations

from miniworld.layers.implementation import ImplementationType, to_engine_impl
from miniworld_engine.modules import atom_dit as _engine

SwiGLUFFN = _engine.SwiGLUFFN
# engine 2.x ``SWAAtomBlock`` is the engine 3.x ``AtomDiTMiniWorld`` (same block, same parameter names)
SWAAtomBlock = _engine.AtomDiTMiniWorld

__all__ = ["SWAAtomBlock", "SWAAtomTransformer", "SwiGLUFFN"]


class SWAAtomTransformer(_engine.SWAAtomTransformer):
    """The engine's ``SWAAtomTransformer``, configured with the MiniWorld ``ImplementationType``."""

    class Config(_engine.SWAAtomTransformer.Config):
        """The engine Config; ``implementation`` is the MiniWorld enum (``RoPESWAAF3Transformer`` shares this Config)."""

        implementation: ImplementationType | None = None

    def __init__(self, config: _engine.SWAAtomTransformer.Config) -> None:
        implementation = getattr(config, "implementation", None)
        engine_config = _engine.SWAAtomTransformer.Config(
            **{k: v for k, v in dict(config).items() if k != "implementation"}
        )
        super().__init__(
            engine_config,
            backend=to_engine_impl(implementation or ImplementationType.PYTORCH),
        )
