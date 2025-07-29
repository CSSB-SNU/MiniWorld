import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal
from pydantic import BaseModel
from jaxtyping import Float, Bool

from team_gm.data.features import NoisyBatch
from team_gm.modules.primitives import Transition, Dropout, Linear

from team_gm.modules.attentions import (
    TriangleAttention,
    TriangleMultiplication,
)

from .attentions import (
    OuterProductMean,
    MSAPairWeightedAveraging,
)
from .diffusion_module import CommonConfig


class MSAModuleBlock(torch.nn.Module):
    class Config(BaseModel):
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

    def __init__(
        self, common_config: CommonConfig, config: Config, last_block: bool = False
    ):
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
            self.transition_msa = Transition(
                config.d_msa, implementation=config.implementation
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
        self.drop_msa = Dropout(broadcast_dim=1, p_drop=config.p_drop_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=config.p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=config.p_drop)
        self.transition_pair = Transition(
            config.d_pair, implementation=config.implementation
        )

    def forward(
        self,
        msa: Float[torch.Tensor, "B N L d_msa"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B N L d_msa"], Float[torch.Tensor, "B L L d_pair"]]:
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
    class Config(MSAModuleBlock.Config):
        n_block: int = 4

    def __init__(self, common_config: CommonConfig, msa_config: Config):
        super().__init__()
        self.num_res_class = common_config.num_res_class

        self.embed_msa = Linear(self.num_res_class + 2, msa_config.d_msa, bias=False)
        self.single_to_msa = Linear(
            common_config.d_token_single_input, msa_config.d_msa, bias=False
        )

        # Create multiple blocks
        self.blocks = nn.ModuleList(
            [
                MSAModuleBlock(common_config, msa_config)
                for _ in range(msa_config.n_block - 1)
            ]
        )
        self.blocks.append(MSAModuleBlock(common_config, msa_config, last_block=True))

    def forward(
        self,
        batch: NoisyBatch,
        recycle_idx: int,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        with torch.no_grad():
            msa_sequences = batch.msa.aligned_sequences[:, recycle_idx] 
            msa_has_deletion = batch.msa.has_deletion[:, recycle_idx]
            msa_deletion_value = batch.msa.deletion_value[:, recycle_idx]

            msa_sequences = F.one_hot(
                msa_sequences.long(), num_classes=self.num_res_class
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

        # # convert dtype
        # pair = pair.to(msa.dtype)

        for block in self.blocks:
            msa, pair = block(msa, pair, mask)
        return pair
