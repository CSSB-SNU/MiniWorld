"""Phase3: EDM diffusion module on top of a FROZEN pair-only mini-SWA trunk.

The trunk is the phase2 ``MiniSWAModel`` (distogram-only, pair-only). Phase3
attaches a :class:`~miniworld.modules.diffusion_module.DiffusionModule`
(ESMFold2 3D-RoPE atom DiT + AF3 token DiT) and trains ONLY the diffusion
module (+ a small single-conditioning projection) with the EDM diffusion loss.
The trunk weights are loaded from the epoch-900 checkpoint and frozen.
"""

from .client import Client
from .model import Model, ModelWrapper, Phase3Model

__all__ = [
    "Client",
    "Model",
    "ModelWrapper",
    "Phase3Model",
]
