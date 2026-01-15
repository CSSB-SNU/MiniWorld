from typing import Literal

import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel, model_validator
from team_gm import typecheck
from team_gm.modules.primitives import Dropout, Transition
from torch import nn
from typing_extensions import Self

from miniworld.modules.attentions import (
    AttentionPairBias,
    OuterProduct,
    TriangleAttention,
    TriangleMultiplication,
)


class PairformerBlock(nn.Module):
    """A single Pairformer block."""

    class Config(BaseModel):
        """Configuration for PairformerBlock module."""

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

        @model_validator(mode="after")
        def validate_config(self) -> Self:
            """Validate configuration."""
            if self.use_single_cond and not self.use_single:
                msg = "use_single_cond requires use_single to be True"
                raise ValueError(msg)
            return self

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        if config.use_single_cond:
            self.single_to_pair = OuterProduct(config.d_single, config.d_pair, implementation=config.implementation)

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
        self.tri_atten_starting = TriangleAttention(
            d_pair=config.d_pair,
            d_hidden=config.d_hidden_tri_attention,
            n_head=config.n_head_tri_attention,
            starting=True,
            use_self_attention=config.use_self_attention,
            implementation=config.implementation,
        )
        self.tri_atten_ending = TriangleAttention(
            d_pair=config.d_pair,
            d_hidden=config.d_hidden_tri_attention,
            n_head=config.n_head_tri_attention,
            starting=False,
            use_self_attention=config.use_self_attention,
            implementation=config.implementation,
        )
        self.drop_row = Dropout(broadcast_dim=1, p_drop=config.p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=config.p_drop)
        self.transition_pair = Transition(
            config.d_pair,
            implementation=config.implementation,
        )

        if config.use_single:
            self.pair_to_single = AttentionPairBias(
                d_single=config.d_single,
                d_pair=config.d_pair,
                n_head=config.n_head_attention,
            )
            self.transition_single = Transition(
                config.d_single,
                implementation="pytorch",
            )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B L L d_pair"],
        Float[torch.Tensor, "B L d_single"] | None,
    ]:
        """Forward pass."""
        if self.config.use_single_cond:
            if single is None:
                msg = "single representation required"
                raise ValueError(msg)
            pair = pair + self.single_to_pair(single)

        pair = pair + self.drop_row(self.tri_multi_outgoing(pair, mask))
        pair = pair + self.drop_row(self.tri_multi_incoming(pair, mask))
        pair = pair + self.drop_row(self.tri_atten_starting(pair, mask))
        pair = pair + self.drop_col(self.tri_atten_ending(pair, mask))
        pair = pair + self.transition_pair(pair)

        if self.config.use_single:
            if single is None:
                msg = "single representation required"
                raise ValueError(msg)
            single = single + self.pair_to_single(single, pair, mask)
            single = single + self.transition_single(single)

        return pair, single



class PairformerBlockGroup(nn.Module):
    """A group of Pairformer blocks with shared configurations."""

    class Config(PairformerBlock.Config):
        """Configuration for PairformerBlockGroup module."""

        n_block: int = 4

    def __init__(self, config: Config, use_checkpoint: bool) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.pairformer_blocks = nn.ModuleList(
            [PairformerBlock(config) for _ in range(config.n_block)],
        )

    def _forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B L L d_pair"],
        Float[torch.Tensor, "B L d_single"] | None,
    ]:
        for block in self.pairformer_blocks:
            pair, single = block(pair, single, mask)
        return pair, single

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B L L d_pair"],
        Float[torch.Tensor, "B L d_single"] | None,
    ]:
        """Forward pass."""
        if self.use_checkpoint:
            pair, single = torch.utils.checkpoint.checkpoint(
                self._forward,
                pair,
                single,
                mask,
            )
        else:
            for block in self.pairformer_blocks:
                pair, single = block(pair, single, mask)
        return pair, single

class Pairformer(nn.Module):
    """The Pairformer stack consisting of multiple Pairformer blocks."""

    class Config(PairformerBlock.Config):
        """Configuration for Pairformer module."""

        n_block: int = 4
        n_block_per_group: int = 4
        use_checkpoint: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__()

        if config.n_block % config.n_block_per_group != 0:
            msg = "n_block must be divisible by n_block_per_group"
            raise ValueError(msg)
        self.n_block_groups = config.n_block // config.n_block_per_group
        self.pairformer_groups = nn.ModuleList(
            [
                PairformerBlockGroup(
                    PairformerBlockGroup.Config(
                        **config.model_dump(exclude={"n_block"}),
                    ),
                    use_checkpoint=config.use_checkpoint,
                )
                for _ in range(self.n_block_groups)
            ],
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B L L d_pair"],
        Float[torch.Tensor, "B L d_single"] | None,
    ]:
        """Forward pass."""
        for group in self.pairformer_groups:
            pair, single = group(pair, single, mask)
        return pair, single
