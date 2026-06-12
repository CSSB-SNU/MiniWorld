"""Configurations for MiniWorld."""

from .data import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TemplateConfig,
    TokenEmbeddingConfig,
    TokenizerConfig,
)
try:
    from .diffusion import (
        EDMDiffuserConfig,
        XPredDecoupledDiffuserConfig,
        XPredEuclideanDiffuserConfig,
    )
except ModuleNotFoundError:
    EDMDiffuserConfig = None  # type: ignore[assignment]
    XPredDecoupledDiffuserConfig = None  # type: ignore[assignment]
    XPredEuclideanDiffuserConfig = None  # type: ignore[assignment]

try:
    from .models import AtomSWAConfig, SharedConfig
except ModuleNotFoundError:
    SharedConfig = None  # type: ignore[assignment]
    AtomSWAConfig = None  # type: ignore[assignment]

__all__ = [
    "BioMolDBConfig",
    "CropConfig",
    "MSAConfig",
    "SamplerConfig",
    "TemplateConfig",
    "TokenEmbeddingConfig",
    "TokenizerConfig",
]

if SharedConfig is not None:
    __all__.append("SharedConfig")

if AtomSWAConfig is not None:
    __all__.append("AtomSWAConfig")

if EDMDiffuserConfig is not None:
    __all__.extend(
        [
            "EDMDiffuserConfig",
            "XPredDecoupledDiffuserConfig",
            "XPredEuclideanDiffuserConfig",
        ],
    )
