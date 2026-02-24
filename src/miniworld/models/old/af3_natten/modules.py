import torch
from jaxtyping import Bool, Float
from natten import NeighborhoodAttention2D
from pydantic import BaseModel, model_validator
from team_gm import typecheck
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.layers import (
    AttentionPairBias,
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from team_gm.modules.layers.ops import sigmoid_gate
from team_gm.modules.primitives import Dropout, LayerNorm, Linear
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from typing_extensions import Self


class NATLayer(nn.Module):
    """A wrapper for NeighborhoodAttention2D to be used in PairformerBlock."""

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int = 32,
        n_head: int = 4,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        self.d_hidden = d_hidden
        self.n_head = n_head

        self.ln_pair = LayerNorm(d_pair)
        self.to_gate = Linear(d_pair, d_hidden * n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_pair, bias=False, init="zero")
        self.attn = NeighborhoodAttention2D(
            embed_dim=d_pair,
            num_heads=n_head,
            kernel_size=(kernel_size, kernel_size),
            is_causal=False,
        )

    @torch.compiler.disable()
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        pair = self.ln_pair(pair)
        out = self.attn(pair)
        out = sigmoid_gate(self.to_gate(pair), out)
        return self.to_out(out)


class PairformerBlock(nn.Module):
    """A single Pairformer block."""

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 128,
        d_hidden_tri_multi: int = 128,
        d_hidden_tri_attention: int = 32,
        n_head_tri_attention: int = 4,
        n_head_attention: int = 16,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: int = 1,
        p_drop: float = 0.25,
        *,
        use_self_attention: bool = True,
        use_single: bool = True,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()

        self.tri_multi_outgoing = TriangleMultiplication(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_multi,
            outgoing=True,
            implementation=implementation,
        )
        self.tri_multi_incoming = TriangleMultiplication(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_multi,
            outgoing=False,
            implementation=implementation,
        )
        self.tri_atten_starting = TriangleAttention(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_attention,
            n_head=n_head_tri_attention,
            starting=True,
            use_self_attention=use_self_attention,
            implementation=implementation,
        )
        self.tri_atten_ending = TriangleAttention(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_attention,
            n_head=n_head_tri_attention,
            starting=False,
            use_self_attention=use_self_attention,
            implementation=implementation,
        )
        self.natten = NATLayer(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_attention,
            n_head=n_head_tri_attention,
            kernel_size=kernel_size,
        )
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.transition_pair = Transition(d_pair, implementation=implementation)

        if use_single:
            self.pair_to_single = AttentionPairBias(
                d_single=d_single,
                d_pair=d_pair,
                n_head=n_head_attention,
            )
            self.transition_single = Transition(d_single)

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
        pair = pair + self.natten(pair)
        pair = pair + self.drop_row(self.tri_multi_outgoing(pair, mask))
        pair = pair + self.drop_row(self.tri_multi_incoming(pair, mask))
        pair = pair + self.drop_row(self.tri_atten_starting(pair, mask))
        pair = pair + self.drop_col(self.tri_atten_ending(pair, mask))
        pair = pair + self.transition_pair(pair)

        if hasattr(self, "pair_to_single") and single is not None:
            single = single + self.pair_to_single(single, pair, mask)
        if hasattr(self, "transition_single") and single is not None:
            single = single + self.transition_single(single)
        return pair, single


class Pairformer(nn.Module):
    """The Pairformer stack consisting of multiple Pairformer blocks."""

    class Config(BaseModel):
        """Configuration for Pairformer module."""

        d_single: int = 384
        d_pair: int = 128
        d_hidden_tri_multi: int = 128
        d_hidden_tri_attention: int = 32
        n_head_tri_attention: int = 4
        n_head_attention: int = 16
        p_drop: float = 0.25
        use_self_attention: bool = True
        use_single: bool = True
        implementation: ImplementationType = ImplementationType.PYTORCH
        n_block: int = 4
        n_checkpoint_segments: int | None = None

        @model_validator(mode="after")
        def check_checkpoint_segments(self) -> Self:
            """Check n_checkpoint_segments is valid."""
            if self.n_checkpoint_segments is None:
                return self
            if (
                self.n_checkpoint_segments > self.n_block
                or self.n_checkpoint_segments < 1
            ):
                msg = (
                    "n_checkpoint_segments must be between 1 and n_block. "
                    f"Got n_checkpoint_segments={self.n_checkpoint_segments} "
                    f"and n_block={self.n_block}."
                )
                raise ValueError(msg)
            return self

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.pairformer_blocks = nn.ModuleList(
            [
                PairformerBlock(
                    d_single=config.d_single,
                    d_pair=config.d_pair,
                    d_hidden_tri_multi=config.d_hidden_tri_multi,
                    d_hidden_tri_attention=config.d_hidden_tri_attention,
                    n_head_tri_attention=config.n_head_tri_attention,
                    n_head_attention=config.n_head_attention,
                    p_drop=config.p_drop,
                    use_self_attention=config.use_self_attention,
                    use_single=config.use_single,
                    implementation=config.implementation,
                )
                for _ in range(config.n_block)
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
        if self.config.n_checkpoint_segments is None:
            for block in self.pairformer_blocks:
                pair, single = block(pair, single, mask)
        else:

            def run_module(module):  # noqa: ANN001, ANN202
                def forward(inputs):  # noqa: ANN001, ANN202
                    p, s = inputs
                    return module(p, s, mask)

                return forward

            pair, single = checkpoint_sequential(
                [run_module(b) for b in self.pairformer_blocks],
                segments=self.config.n_checkpoint_segments,
                use_reentrant=False,
                input=(pair, single),
            )

        return pair, single


class MSAModuleBlock(torch.nn.Module):
    """MSA update and pair update module block."""

    def __init__(
        self,
        d_msa: int = 64,
        d_pair: int = 128,
        d_hidden_msa: int = 32,
        d_hidden_tri_multi: int = 128,
        d_hidden_tri_attention: int = 32,
        n_head_tri_attention: int = 4,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: int = 1,
        p_drop_msa: float = 0.15,
        p_drop: float = 0.25,
        *,
        use_self_attention: bool = True,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        last_block: bool = False,
    ) -> None:
        super().__init__()

        self.last_block = last_block

        self.outer_product_mean = OuterProductMean(
            d_msa=d_msa,
            d_pair=d_pair,
            d_hidden=d_hidden_msa,
        )
        if not last_block:
            self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
                d_msa=d_msa,
                d_pair=d_pair,
                d_hidden=d_hidden_msa,
            )
            self.transition_msa = Transition(
                d_msa,
                implementation=implementation,
            )

        self.tri_multi_outgoing = TriangleMultiplication(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_multi,
            outgoing=True,
            implementation=implementation,
        )
        self.tri_multi_incoming = TriangleMultiplication(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_multi,
            outgoing=False,
            implementation=implementation,
        )

        # Triangle attention layers
        self.tri_atten_starting = TriangleAttention(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_attention,
            n_head=n_head_tri_attention,
            starting=True,
            use_self_attention=use_self_attention,
            implementation=implementation,
        )
        self.tri_atten_ending = TriangleAttention(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_attention,
            n_head=n_head_tri_attention,
            starting=False,
            use_self_attention=use_self_attention,
            implementation=implementation,
        )
        self.natten = NATLayer(
            d_pair=d_pair,
            d_hidden=d_hidden_tri_attention,
            n_head=n_head_tri_attention,
            kernel_size=kernel_size,
        )
        self.drop_msa = Dropout(broadcast_dim=1, p_drop=p_drop_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)
        self.transition_pair = Transition(
            d_pair,
            implementation=implementation,
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
        pair = pair + self.natten(pair)
        pair = pair + self.drop_row(self.tri_multi_outgoing(pair, mask))
        pair = pair + self.drop_row(self.tri_multi_incoming(pair, mask))
        pair = pair + self.drop_row(self.tri_atten_starting(pair, mask))
        pair = pair + self.drop_col(self.tri_atten_ending(pair, mask))
        pair = pair + self.transition_pair(pair)  # Transition MSA

        return msa, pair


class MSAModule(torch.nn.Module):
    """MSA module with multiple blocks."""

    class Config(BaseModel):
        """Configuration for MSAModuleBlock."""

        d_msa: int = 64
        d_pair: int = 128
        d_single_token_input: int = 441
        d_hidden_msa: int = 32
        d_hidden_tri_multi: int = 128
        d_hidden_tri_attention: int = 32
        n_head_tri_attention: int = 4
        n_head_attention: int = 16
        p_drop_msa: float = 0.15
        p_drop: float = 0.25
        use_self_attention: bool = True
        implementation: ImplementationType = ImplementationType.PYTORCH
        num_res_class: int = 32

        n_block: int = 4
        n_checkpoint_segments: int | None = None

        @model_validator(mode="after")
        def check_checkpoint_segments(self) -> Self:
            """Check n_checkpoint_segments is valid."""
            if self.n_checkpoint_segments is None:
                return self
            if (
                self.n_checkpoint_segments > self.n_block
                or self.n_checkpoint_segments < 1
            ):
                msg = (
                    "n_checkpoint_segments must be between 1 and n_block. "
                    f"Got n_checkpoint_segments={self.n_checkpoint_segments} "
                    f"and n_block={self.n_block}."
                )
                raise ValueError(msg)
            return self

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_res_class = config.num_res_class

        self.embed_msa = Linear(self.num_res_class + 2, config.d_msa, bias=False)
        self.single_to_msa = Linear(
            config.d_single_token_input,
            config.d_msa,
            bias=False,
        )

        # Create multiple blocks
        self.blocks = nn.ModuleList(
            [
                MSAModuleBlock(
                    d_msa=config.d_msa,
                    d_pair=config.d_pair,
                    d_hidden_msa=config.d_hidden_msa,
                    d_hidden_tri_multi=config.d_hidden_tri_multi,
                    d_hidden_tri_attention=config.d_hidden_tri_attention,
                    n_head_tri_attention=config.n_head_tri_attention,
                    p_drop_msa=config.p_drop_msa,
                    p_drop=config.p_drop,
                    use_self_attention=config.use_self_attention,
                    implementation=config.implementation,
                )
                for _ in range(config.n_block - 1)
            ],
        )
        self.blocks.append(
            MSAModuleBlock(
                d_msa=config.d_msa,
                d_pair=config.d_pair,
                d_hidden_msa=config.d_hidden_msa,
                d_hidden_tri_multi=config.d_hidden_tri_multi,
                d_hidden_tri_attention=config.d_hidden_tri_attention,
                n_head_tri_attention=config.n_head_tri_attention,
                p_drop_msa=config.p_drop_msa,
                p_drop=config.p_drop,
                use_self_attention=config.use_self_attention,
                implementation=config.implementation,
                last_block=True,
            ),
        )

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B N L d_msa"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        single: Float[torch.Tensor, "B L d_single"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        msa = msa.to(pair.dtype)
        msa = self.embed_msa(msa)  # (B, N, L, d_msa)
        msa = msa + self.single_to_msa(single).unsqueeze(1)  # (B, N, L, d_msa)

        for block in self.blocks:
            msa, pair = block(msa, pair, mask)
        return pair
