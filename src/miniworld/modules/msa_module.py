from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from pydantic import BaseModel
from team_gm import typecheck
from team_gm.modules.attentions import (
    TriangleAttention,
    TriangleMultiplication,
)
from team_gm.modules.primitives import Dropout, Linear, Transition
from torch import nn

from miniworld.data.features.features_biomol import NoisyBatch
from miniworld.modules.attentions import MSAPairWeightedAveraging, OuterProductMean
from miniworld.modules.primitives import MoETransition

from .diffusion_module import CommonConfig


class MSAModuleBlock(torch.nn.Module):
    """MSA update and pair update module block."""

    class Config(BaseModel):
        """Configuration for MSAModuleBlock."""

        d_msa: int = 64
        d_pair: int = 128
        d_hidden_msa: int = 32
        d_hidden_tri_multi: int = 128
        d_hidden_tri_attention: int = 32
        n_head_tri_attention: int = 4
        n_head_attention: int = 16
        p_drop_msa: float = 0.15
        p_drop: float = 0.25
        use_self_attention: bool = True
        implementation: Literal["pytorch", "triton"] = "pytorch"

        msa_moe_experts: int = 1  # 1 -> no MoE
        msa_moe_topk: int = 1
        pair_moe_experts: int = 1
        pair_moe_topk: int = 1

    def __init__(
        self, config: Config, last_block: bool = False,
    ) -> None:
        super().__init__()
        self.d_msa = config.d_msa
        self.d_pair = config.d_pair
        self.d_hidden_msa = config.d_hidden_msa
        self.d_hidden_tri_multi = config.d_hidden_tri_multi
        self.d_hidden_tri_attention = config.d_hidden_tri_attention
        self.n_head_tri_attention = config.n_head_tri_attention
        self.n_head_attention = config.n_head_attention
        self.p_drop = config.p_drop
        self.implementation = config.implementation

        self.last_block = last_block

        self.outer_product_mean = OuterProductMean(
            d_msa=config.d_msa,
            d_pair=config.d_pair,
            d_hidden=config.d_hidden_msa,
            implementation=config.implementation,
        )
        if not last_block:
            self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
                d_msa=config.d_msa,
                d_pair=config.d_pair,
                d_hidden=config.d_hidden_msa,
                implementation=config.implementation,
            )
            if config.msa_moe_experts > 1:
                self.transition_msa = MoETransition(
                    config.d_msa,
                    experts=config.msa_moe_experts,
                    topk=config.msa_moe_topk,
                    implementation=config.implementation,
                )
            else:
                self.transition_msa = Transition(
                    config.d_msa, implementation=config.implementation,
                )

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
        )
        self.tri_atten_ending = TriangleAttention(
            d_pair=config.d_pair,
            d_hidden=config.d_hidden_tri_attention,
            n_head=config.n_head_tri_attention,
            starting=False,
            use_self_attention=config.use_self_attention,
            implementation=config.implementation,
        )
        self.drop_msa = Dropout(broadcast_dim=1, p_drop=config.p_drop_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=config.p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=config.p_drop)
        if config.pair_moe_experts > 1:
            self.transition_pair = MoETransition(
                config.d_pair,
                experts=config.pair_moe_experts,
                topk=config.pair_moe_topk,
                implementation=config.implementation,
            )
        else:
            self.transition_pair = Transition(
                config.d_pair, implementation=config.implementation,
            )

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B N L d_msa"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B N L d_msa"], Float[torch.Tensor, "B L L d_pair"]]:
        """Forward pass."""
        # MSA to pair
        pair = pair + self.outer_product_mean(msa)

        if not self.last_block:
            # Pair to MSA
            msa = msa + self.drop_msa(self.msa_pair_weighted_averaging(msa, pair, mask))
            msa = msa + self.transition_msa(msa)

        # pairformer
        pair = pair + self.drop_row(self.tri_multi_outgoing(pair, mask))
        pair = pair + self.drop_row(self.tri_multi_incoming(pair, mask))
        pair = pair + self.drop_row(self.tri_atten_starting(pair, mask))
        pair = pair + self.drop_col(self.tri_atten_ending(pair, mask))
        pair = pair + self.transition_pair(pair)  # Transition MSA

        return msa, pair


class MSAModule(torch.nn.Module):
    """MSA module with multiple blocks."""

    class Config(MSAModuleBlock.Config):
        """Configuration for MSAModule."""

        n_block: int = 4

    def __init__(self, common_config: CommonConfig, msa_config: Config) -> None:
        super().__init__()
        self.num_res_class = common_config.num_res_class

        self.embed_msa = Linear(self.num_res_class + 2, msa_config.d_msa, bias=False)
        self.single_to_msa = Linear(
            common_config.d_token_single_input, msa_config.d_msa, bias=False,
        )

        # Create multiple blocks
        self.blocks = nn.ModuleList(
            [
                MSAModuleBlock(msa_config)
                for _ in range(msa_config.n_block - 1)
            ],
        )
        self.blocks.append(MSAModuleBlock(msa_config, last_block=True))

    @typecheck
    def forward(
        self,
        batch: NoisyBatch,
        recycle_idx: int,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        with torch.no_grad():
            msa_sequences = batch.msa.aligned_sequences[:, recycle_idx]
            msa_has_deletion = batch.msa.has_deletion[:, recycle_idx]
            msa_deletion_value = batch.msa.deletion_value[:, recycle_idx].float()

            msa_sequences = F.one_hot(
                msa_sequences.long(), num_classes=self.num_res_class,
            )
            msa = torch.cat(
                [
                    msa_sequences,
                    msa_has_deletion.unsqueeze(-1),
                    msa_deletion_value.unsqueeze(-1),
                ],
                dim=-1,
            )  # (B, N, L, num_res_class + 2)
        msa = msa.to(pair.dtype)
        msa = self.embed_msa(msa)  # (B, N, L, d_msa)
        msa = msa + self.single_to_msa(single).unsqueeze(1)  # (B, N, L, d_msa)

        for block in self.blocks:
            msa, pair = block(msa, pair, mask)
        return pair
