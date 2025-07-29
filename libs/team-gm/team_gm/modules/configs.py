import torch
from typing import Literal

from pydantic import BaseModel


class CommonConfig(BaseModel):
    d_token_single: int = 384
    d_token_single_input: int = 441  # 441 = 384 + 32 + 24 + 1
    d_token_single_diffusion: int = 768
    d_token_pair: int = 128
    d_time: int = 256
    d_msa: int = 64
    d_atom_single: int = 128
    d_atom_pair: int = 16
    num_res_class: int = 32
    r_max: int = 32
    s_max: int = 2
    implementation: Literal["pytorch", "triton"] = "pytorch"
    use_checkpoint: bool = False
    to_bias_init: Literal["zero", "default"] = "default"
    norm: Literal["pre", "hybrid"] = ("pre",)


class DiffusionConfig(BaseModel):
    frequency_embedding_size: int = 256
    n_block_atom: int = 3
    n_block_token: int = 12
    n_head_atom: int = 4
    n_head_token: int = 24
    n_transition_expand: int = 2
    n_transition_block: int = 2
    implementation: Literal["pytorch", "triton"] = "pytorch"
    use_beta: bool = True
