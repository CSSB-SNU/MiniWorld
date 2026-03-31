from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from team_gm import typecheck
from team_gm.modules import Dropout, ImplementationType
from team_gm.modules.layers import (
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from torch import nn

if TYPE_CHECKING:
    from jaxtyping import Bool, Float

    from miniworld.configs import SharedConfig


class TemplatePairformerBlock(nn.Module):
    """Single Pairformer block for template processing (pair-only)."""

    def __init__(
        self,
        d_pair: int,
        d_hidden: int,
        n_head_tri_attention: int,
        p_drop: float,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        use_self_attention: bool = True,
    ) -> None:
        super().__init__()

        self.tri_multi_outgoing = TriangleMultiplication(
            d_pair=d_hidden,
            outgoing=True,
            implementation=implementation,
        )
        self.tri_multi_incoming = TriangleMultiplication(
            d_pair=d_hidden,
            outgoing=False,
            implementation=implementation,
        )
        self.tri_atten_starting = TriangleAttention(
            d_pair=d_hidden,
            n_head=n_head_tri_attention,
            starting=True,
            implementation=implementation,
            use_self_attention=use_self_attention,
        )
        self.tri_atten_ending = TriangleAttention(
            d_pair=d_hidden,
            n_head=n_head_tri_attention,
            starting=False,
            implementation=implementation,
            use_self_attention=use_self_attention,
        )
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)
        self.transition_pair = Transition(
            d_pair,
            n=2,
            implementation=implementation
            if implementation != ImplementationType.CUEQUIVARIANCE
            else ImplementationType.TRITON,
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        pair = pair + self.drop_row(self.tri_multi_outgoing(pair, mask))
        pair = pair + self.drop_row(self.tri_multi_incoming(pair, mask))
        pair = pair + self.drop_row(self.tri_atten_starting(pair, mask))
        pair = pair + self.drop_col(self.tri_atten_ending(pair, mask))
        return pair + self.transition_pair(pair)


class TemplatePairformer(nn.Module):
    """Stack of TemplatePairformerBlocks."""

    class Config(BaseModel):
        """Configuration for TemplatePairformer."""

        d_pair_template_input: int = 4
        d_pair: int = 128
        d_hidden: int = 128
        n_head_tri_attention: int = 4
        p_drop: float = 0.25
        use_self_attention: bool = True
        implementation: ImplementationType = ImplementationType.PYTORCH
        n_block: int = 2

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TemplatePairformerBlock(
                    d_pair=config.d_pair,
                    d_hidden=config.d_hidden,
                    n_head_tri_attention=config.n_head_tri_attention,
                    p_drop=config.p_drop,
                    implementation=config.implementation,
                    use_self_attention=config.use_self_attention,
                )
                for _ in range(config.n_block)
            ],
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        for block in self.blocks:
            pair = block(pair, mask)
        return pair


class TemplateEmbedder(nn.Module):
    """Template embedder (Algorithm 16)."""

    def __init__(
        self,
        config: SharedConfig,
        template_pairformer_config: TemplatePairformer.Config,
    ) -> None:
        super().__init__()
        self.config = config

        self.ln_pair = nn.LayerNorm(config.d_pair)
        self.proj_pair = nn.Linear(config.d_pair, config.d_pair_template, bias=False)

        self.proj_template = nn.Linear(
            template_pairformer_config.d_pair_template_input,
            config.d_pair_template,
            bias=False,
        )

        self.template_pairformer = TemplatePairformer(template_pairformer_config)
        self.ln_template = nn.LayerNorm(config.d_pair_template)

        self.proj_out = nn.Linear(config.d_pair_template, config.d_pair, bias=False)

    @typecheck
    def forward(
        self,
        template_feat: Float[torch.Tensor, "B L L d_pair_template_input"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        pair = self.proj_pair(self.ln_pair(pair)) + self.proj_template(template_feat)

        pair = pair + self.template_pairformer(pair, mask=mask)
        pair = self.ln_template(pair)

        return self.proj_out(F.relu(pair))
