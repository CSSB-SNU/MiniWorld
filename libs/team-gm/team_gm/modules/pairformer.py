import torch

from torch import nn
from typing import Literal
from pydantic import BaseModel
from jaxtyping import Float, Bool

from team_gm import typecheck
from .primitives import Transition, Dropout
from .attentions import (
    OuterProduct,
    AttentionPairBias,
    TriangleAttention,
    TriangleMultiplication,
)
from .diffusion_module import CommonConfig


class Pairformer(nn.Module):
    """The Pairformer stack consisting of multiple Pairformer blocks.

    This module stacks multiple PairformerBlock to process single and pair presentations.
    """

    class Config(BaseModel):
        d_single: int = 384
        d_pair: int = 128
        d_hidden_tri_multi: int = 128
        d_hidden_tri_attention: int = 32
        n_head_tri_attention: int = 4
        n_head_attention: int = 16
        p_drop: float = 0.25
        use_single: bool = True
        use_single_cond: bool = False
        use_self_attention: bool = True
        implementation: Literal["pytorch", "triton"] = "pytorch"
        n_block: int = 4

    def __init__(self, common_config: CommonConfig, config: Config):
        super().__init__()

        self.pairformer_blocks = nn.ModuleList(
            [PairformerBlock(common_config, config) for _ in range(config.n_block)]
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B L L d_pair"], Float[torch.Tensor, "B L d_single"]]:
        for block in self.pairformer_blocks:
            pair, single = block(pair, single, mask)
        return pair, single


class PairformerBlock(nn.Module):
    def __init__(self, common_config: CommonConfig, config: Pairformer.Config):
        super().__init__()
        self.config = config

        if config.use_single_cond and not config.use_single:
            raise ValueError("use_single_cond requires use_single to be True")

        # Triangle multiplication layers
        self.tri_multi_outgoing = TriangleMultiplication(
            d_pair=config.d_pair,
            d_hidden=config.d_hidden_tri_multi,
            outgoing=True,
            implementation=config.implementation,
        )
        self.tri_multi_incoming = TriangleMultiplication(
            d_pair=config.d_pair,
            d_hidden=config.d_hidden_tri_multi,
            outgoing=False,
            implementation=config.implementation,
        )

        # Triangle attention layers
        self.tri_atten_starting = TriangleAttention(
            d_pair=config.d_pair,
            d_hidden=config.d_hidden_tri_attention,
            n_head=config.n_head_tri_attention,
            starting=True,
            use_self_attention=config.use_self_attention,
            implementation=config.implementation,
            to_bias_init=common_config.to_bias_init,
            norm=common_config.norm,
        )
        self.tri_atten_ending = TriangleAttention(
            d_pair=config.d_pair,
            d_hidden=config.d_hidden_tri_attention,
            n_head=config.n_head_tri_attention,
            starting=False,
            use_self_attention=config.use_self_attention,
            implementation=config.implementation,
            to_bias_init=common_config.to_bias_init,
            norm=common_config.norm,
        )
        self.drop_row = Dropout(broadcast_dim=1, p_drop=config.p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=config.p_drop)
        self.transition_pair = Transition(
            config.d_pair, implementation=config.implementation
        )

        # Single update layers
        if config.use_single:
            self.pair_to_single = AttentionPairBias(
                d_single=config.d_single,
                d_pair=config.d_pair,
                n_head=config.n_head_attention,
                implementation="pytorch",
                to_bias_init=common_config.to_bias_init,
                norm=common_config.norm,
            )  # NOTE: Triton version is not useful for small input, so we use PyTorch
            self.transition_single = Transition(
                config.d_single,
                implementation="pytorch",
            )  # NOTE: Triton Transition is only for pair representation

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B L L d_pair"], Float[torch.Tensor, "B L d_single"]]:
        if self.config.use_single_cond:
            if single is None:
                raise ValueError("single representation required")
            pair = pair + self.single_to_pair(single)

        pair = pair + self.drop_row(self.tri_multi_outgoing(pair, mask))
        pair = pair + self.drop_row(self.tri_multi_incoming(pair, mask))
        pair = pair + self.drop_row(self.tri_atten_starting(pair, mask))
        pair = pair + self.drop_col(self.tri_atten_ending(pair, mask))
        pair = pair + self.transition_pair(pair)

        if self.config.use_single:
            if single is None:
                raise ValueError("single representation required")
            single = single + self.pair_to_single(single, pair, mask)
            single = single + self.transition_single(single)

        return pair, single
