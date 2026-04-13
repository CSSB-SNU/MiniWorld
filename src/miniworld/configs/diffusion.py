from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator

from miniworld.diffusion import DecoupledEDMScheduler, EDMScheduler


class EDMDiffuserConfig(BaseModel):
    """Configuration for the diffuser."""

    type: Literal["EDM"] = "EDM"
    seed: int = 0
    scheduler: EDMScheduler.EDMSchedulerConfig
    method: Literal["AF3", "EDM"] = "AF3"


class DecoupledEDMDiffuserConfig(BaseModel):
    """Configuration for the decoupled diffuser."""

    type: Literal["DecoupledEDM"] = "DecoupledEDM"
    seed: int = 0
    translation_noise: float = 0.0
    scheduler: DecoupledEDMScheduler.DecoupledEDMSchedulerConfig


DiffuserConfig = Annotated[
    Union[EDMDiffuserConfig, DecoupledEDMDiffuserConfig],
    Discriminator("type"),
]
