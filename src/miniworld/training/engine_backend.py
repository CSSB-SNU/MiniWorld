"""Choose the engine policy before model construction, once per training process."""

import logging
from typing import Literal

EngineBackend = Literal["auto", "triton"]


def configure_engine_backend(backend: EngineBackend) -> None:
    """Set the engine backend policy before constructing a training model."""
    from miniworld_engine import settings

    if backend not in ("auto", "triton"):
        message = f"Unknown engine backend: {backend}"
        raise ValueError(message)
    if not hasattr(settings.current(), "engine_backend"):
        message = "Engine backend option requires the current engine-setup patches."
        raise RuntimeError(message)
    settings.configure(engine_backend=backend)
    logging.getLogger(__name__).info(
        "Engine backend=%s (process-wide; FlashAttention and library GEMMs retain their own backends)",
        backend,
    )


def configure_fused_msa_train(enabled: bool) -> None:
    """Explicitly select fused MSA training before constructing the model.

    Setting both flags preserves the config on older opt-in engines and Engine 2,
    which enables supported native paths by default. Unsupported inputs still
    use each module's general implementation.
    """
    from miniworld_engine import settings

    # Engine 2 defaults to native MSA: None (automatic) is not "disabled", so the choice is always explicit.
    settings.configure(opm_train=enabled, pwa_train=enabled)
    logging.getLogger(__name__).info(
        "Fused MSA training paths (PWA/OPM kernels): %s", "on" if enabled else "off"
    )


def align_engine_optimizer_state(optimizer) -> int:
    """Adapt restored Adam moments to Engine 2 parameter strides, once on resume."""
    try:
        from miniworld_engine.integrations.optimizer import align_optimizer_state_layout_
    except ModuleNotFoundError as exc:
        if exc.name in {
            "miniworld_engine.integrations",
            "miniworld_engine.integrations.optimizer",
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
    """Apply ``train.engine_settings`` and ``train.adaln_fwd``; call it before the model is built."""
    if options.engine_settings:
        from miniworld_engine import settings

        settings.configure(**options.engine_settings)
    if options.adaln_fwd:
        from miniworld_engine.kernels.adaln.triton import training

        training.set_forward_mode(options.adaln_fwd)
