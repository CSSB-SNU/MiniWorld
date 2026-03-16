from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from miniworld.diffusion import EDMScheduler


class EDMDiffuserConfig(BaseModel):
    """Configuration for the diffuser."""

    seed: int = 0
    scheduler: EDMScheduler.EDMSchedulerConfig
    method: Literal["AF3", "EDM"] = "AF3"
