from __future__ import annotations

from pydantic import BaseModel
from team_gm.modules import ImplementationType


class AtomSWAConfig(BaseModel):
    """Opt-in ESMFold2-style atom attention: sliding-window + 3D RoPE + QK-norm.

    When ``enabled`` is False (default) the diffusion module uses the existing
    pair-bias atom attention, so existing configs are unaffected.
    """

    enabled: bool = False
    # Sliding window (full width); half-window = swa_window_size // 2.
    swa_window_size: int = 128
    # Local-attention backend: "flex" (native block-sparse, default),
    # "sdpa" (dense banded mask), or "flash" (flash_attn varlen, if built).
    backend: str = "flex"
    expansion_ratio: int = 2
    # 3D RoPE config. With atom head_dim = d_single_atom // n_head = 32,
    # the defaults (3*2 spatial + 10 uid = 16 = head_dim/2) fit exactly.
    n_spatial_rope_pairs_per_axis: int = 2
    n_uid_rope_pairs: int = 10
    spatial_rope_base_frequency: float = 20.0
    uid_rope_base_frequency: float = 10000.0


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
