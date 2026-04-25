from __future__ import annotations

from pydantic import BaseModel
from team_gm.modules import ImplementationType


class SharedConfig(BaseModel):
    """Shared configuration for StructureFlow model."""

    d_single: int = 384
    d_single_atom: int = 128
    d_single_token: int = 768
    d_single_token_input: int = 441  # 441 = 384 + 32 + 24 + 1

    d_pair: int = 128
    d_pair_template: int = 128
    d_pair_atom: int = 16
    d_contact: int = 3

    d_time: int = 256

    r_max: int = 32
    s_max: int = 2

    dgram_bins_template: int = 39
    relpos_bins: int = 32
    noise_freq: int = 256
    num_res_class: int = 32

    n_distogram_bins: int = 64

    implementation: ImplementationType = ImplementationType.PYTORCH
    use_checkpoint: bool = False
