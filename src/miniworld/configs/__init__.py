"""Configurations for MiniWorld."""

from .data import (
    BioMolDBConfig,
    CropConfig,
    MSAConfig,
    SamplerConfig,
    TemplateConfig,
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
    from .models import SharedConfig
except ModuleNotFoundError:
    SharedConfig = None  # type: ignore[assignment]

__all__ = [
    "BioMolDBConfig",
    "CropConfig",
    "MSAConfig",
    "SamplerConfig",
    "TemplateConfig",
    "TokenizerConfig",
]

if SharedConfig is not None:
    __all__ += ["SharedConfig"]

if EDMDiffuserConfig is not None:
    __all__ += [
        "EDMDiffuserConfig",
        "XPredDecoupledDiffuserConfig",
        "XPredEuclideanDiffuserConfig",
    ]
