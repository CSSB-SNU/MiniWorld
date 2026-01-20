from typing import Literal

from pydantic import BaseModel


class CommonConfig(BaseModel):
    """Common configuration for the model architecture."""

    d_token_single: int = 384
    d_token_single_input: int = 441  # 441 = 384 + 32 + 24 + 1
    d_token_single_diffusion: int = 768
    d_token_pair: int = 128
    d_time: int = 256
    d_msa: int = 64
    d_atom_single: int = 128
    d_atom_pair: int = 16
    num_res_class: int = 32
    num_distogram_bins: int = 64
    r_max: int = 32
    s_max: int = 2
    implementation: Literal["pytorch", "triton"] = "pytorch"
    use_checkpoint: bool = False
    to_bias_init: Literal["zero", "default"] = "default"
    # 20250824 PreNorm is not compatible with flashattention
    norm: Literal["qk", "hybrid"] = "qk"


class DiffusionConfig(BaseModel):
    """Configuration for the diffusion model."""

    frequency_embedding_size: int = 256
    n_block_atom: int = 3
    n_block_token: int = 12
    n_head_atom: int = 4
    n_head_token: int = 24
    n_transition_expand: int = 2
    n_transition_block: int = 2
    implementation: Literal["pytorch", "triton"] = "pytorch"
    use_beta: bool = True

    pair_moe_experts: int = 1  # 1 -> no MoE
    pair_moe_topk: int = 1
    token_single_moe_experts: int = 1
    token_single_moe_topk: int = 1
    atom_single_moe_experts: int = 1
    atom_single_moe_topk: int = 1
