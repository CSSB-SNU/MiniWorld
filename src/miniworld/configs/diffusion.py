from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field

from miniworld.diffusion import (
    DecoupledXPredScheduler,
    EDMScheduler,
    XPredDecoupledSolver,
)


class EDMDiffuserConfig(BaseModel):
    """Configuration for the diffuser."""

    type: Literal["EDM"] = "EDM"
    seed: int = 0
    scheduler: EDMScheduler.EDMSchedulerConfig
    method: Literal["AF3", "EDM"] = "AF3"


# ---------------------------------------------------------------------------
# X-prediction configs (Back to Basics, Li & He 2025)
# ---------------------------------------------------------------------------


class XPredEuclideanDiffuserConfig(BaseModel):
    """Configuration for VE x-prediction Euclidean diffuser."""

    type: Literal["DecoupledEDM"] = "DecoupledEDM"
    seed: int = 0
    translation_noise: float = 1.0
    scheduler: EDMScheduler.EDMSchedulerConfig


class XPredDecoupledDiffuserConfig(BaseModel):
    """Configuration for VE x-prediction decoupled diffuser (independent scheduler)."""

    seed: int = 0
    translation_noise: float = 1.0
    max_loss_weight: float = 100.0
    scheduler: DecoupledXPredScheduler.DecoupledXPredSchedulerConfig
    solver: XPredDecoupledSolver.Config = Field(
        default_factory=XPredDecoupledSolver.Config,
    )
