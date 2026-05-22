"""Inference-only path for MiniWorld.

This package is completely independent of the training stack
(`miniworld.models.miniworld.Client` / `ModelWrapper`). It reuses the
same trained ``nn.Module`` weights (so checkpoints load directly) but
hoists everything that does not depend on ``x_t`` or the diffusion
timestep out of the per-step loop.

Public API:

>>> predictor = Predictor.from_checkpoint(ckpt_path, model_cfg, diffuser_cfg)
>>> predictor = predictor.to(device).eval()
>>> cache = predictor.prepare(batch)              # trunk + static cache
>>> out = predictor.sample(cache, n_samples=N, timesteps=100, ...)

See ``predictor.py`` for the contract and ``cache.py`` /
``diffusion.py`` for which trunk-derived tensors are reused across
steps.
"""

from __future__ import annotations

from miniworld.inference.cache import InferenceCache, StepSchedule
from miniworld.inference.predictor import Predictor, PredictorOutput

__all__ = [
    "InferenceCache",
    "Predictor",
    "PredictorOutput",
    "StepSchedule",
]
