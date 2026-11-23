"""Choose the engine policy before model construction, once per training process.

miniworld-engine 3.x has no process-wide ``settings`` object: the backend is a per-module ``backend=`` string (MiniWorld's
``implementation`` config value, see ``miniworld.layers.implementation``). The helpers below keep their names and call sites; what
cannot be expressed any more raises (or warns) instead of silently doing something else.
"""

import logging
from typing import Literal

EngineBackend = Literal["auto", "triton"]


def configure_engine_backend(backend: EngineBackend) -> None:
    """Engine backend policy before constructing a training model: ``"auto"`` is the only one left (a no-op)."""
    if backend not in ("auto", "triton"):
        message = f"Unknown engine backend: {backend}"
        raise ValueError(message)
    if backend == "triton":
        message = (
            "train.engine_backend='triton' (engine 2.x settings.force_portable_kernels) does not exist in miniworld-engine 3.x: "
            "the backend is chosen per module (implementation=...), there is no process-wide Triton switch."
        )
        raise NotImplementedError(message)
    logging.getLogger(__name__).info("Engine backend=auto (per-module backend= from the model config's implementation)")


def configure_fused_msa_train(enabled: bool) -> None:
    """Engine 2.x ``settings.opm_train`` / ``pwa_train``: gone in 3.x (the OPM / PWA backend is the module's ``backend=``)."""
    logging.getLogger(__name__).warning(
        "fused_msa_train=%s: engine 3.x has no global fused-MSA switch; the OuterProductMean / MSAPairWeightedAveraging modules "
        "run the backend they were built with (MSAModule builds them without a backend, i.e. the PyTorch reference).",
        enabled,
    )


def align_engine_optimizer_state(optimizer) -> int:
    """Adapt restored Adam moments to Engine 2 parameter strides, once on resume."""
    try:
        from miniworld_engine.utils.optimizer import align_optimizer_state_layout_
    except ModuleNotFoundError as exc:
        if exc.name in {
            "miniworld_engine.utils",
            "miniworld_engine.utils.optimizer",
        }:
            return 0  # Older engines do not change the parameter layout.
        raise
    changed = align_optimizer_state_layout_(optimizer)
    if changed:
        logging.getLogger(__name__).info(
            "Aligned %d engine optimizer state tensors", changed
        )
    return changed


def apply_run_options(options) -> None:
    """``train.engine_settings`` / ``train.adaln_fwd``: engine 2.x ``settings`` and ``runtime`` are gone in 3.x; setting either raises."""
    if options.engine_settings:
        message = f"train.engine_settings={options.engine_settings!r}: miniworld-engine 3.x has no Settings object (engine 2.x only)."
        raise NotImplementedError(message)
    if options.adaln_fwd:
        message = f"train.adaln_fwd={options.adaln_fwd!r}: miniworld-engine 3.x has no runtime.set_adaln_forward_mode (engine 2.x only)."
        raise NotImplementedError(message)
