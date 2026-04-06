from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from miniworld.diffusion import DecoupledEDMScheduler, EDMScheduler


class EDMDiffuserConfig(BaseModel):
    """Configuration for the diffuser."""

    seed: int = 0
    scheduler: EDMScheduler.EDMSchedulerConfig
    method: Literal["AF3", "EDM"] = "AF3"


class DecoupledEDMDiffuserConfig(BaseModel):
    """Configuration for the decoupled diffuser."""

    seed: int = 0
    method: Literal["AF3", "EDM"] = "AF3"
    translation_noise: float = 0.0
    scheduler: DecoupledEDMScheduler.DecoupledEDMSchedulerConfig
