"""Run-level switches of the training scripts (formerly ``MW_*`` environment variables)."""

from typing import Any

from pydantic import BaseModel, Field


class RunOptions(BaseModel):
    """Options every ``TrainConfig`` carries; set them as ``train.<name>=...`` overrides."""

    mem_profile: bool = False
    """Hook the diffusion sub-modules and log each one's retained-activation delta as it runs."""
    capture_cache: bool = False
    """Autotune-cache build mode: record the winning config per op with the compile-timeout guard, flush, exit."""
    compile_timeout: bool = False
    """Install only the fork+SIGKILL compile guard so one uncached monster config cannot stall the run."""
    warmup: bool = False
    """Force the synthetic-bucket warmup of phase 3 (skipped by default)."""
    synth_data: bool = False
    """Real training path, static synthetic batch at the config's largest bucket instead of the dataloader."""
    graph_amp: bool = False
    """Capture the recycle graphs under a plain bf16 autocast instead of the model's own autocast."""
    resume_run_subdir: str | None = None
    """Continue the run in this sub-directory (the run / log identity is preserved); needs ``--ckpt``."""
    adaln_fwd: str = ""
    """Forward mode of the Triton AdaLN training kernel (``cublas`` routes its GEMM to cuBLAS); empty keeps the engine's choice."""
    engine_settings: dict[str, Any] = Field(default_factory=dict)
    """Fields of ``miniworld_engine.settings.Settings`` applied before the model is built."""
