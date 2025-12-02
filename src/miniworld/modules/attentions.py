from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float
from team_gm import typecheck
from team_gm.modules.primitives import (
    Linear,
    Parameter,
)
from torch import nn

from miniworld.data.features.features_biomol import NoisyBatch
from miniworld.modules.kernels import (
    triton_atom_augmented_attention,
    triton_token_augmented_attention,
)
from miniworld.modules.primitives import (
    AdaptiveLayerNorm,
    LayerNorm,
    LinearRMSNorm,
    SigmoidGateFunction,
)


class OuterProductMean(nn.Module):
    """Outer product msa represetation to pair representation.

    Parameters
    ----------
    d_msa: int
        Dimension of msa representation.
    d_pair: int
        Dimension of pair representation.
    d_hidden: int
        Dimension of hidden layer.

    """

    def __init__(
        self,
        d_msa: int = 64,
        d_hidden: int = 32,
        d_pair: int = 128,
        implementation: Literal["pytorch", "triton"] = "pytorch",
    ) -> None:
        super().__init__()

        self.ln_msa = LayerNorm(d_msa, implementation=implementation)
        self.to_left = Linear(d_msa, d_hidden, bias=False)
        self.to_right = Linear(d_msa, d_hidden, bias=False)
        self.to_out = Linear(d_hidden * d_hidden, d_pair, bias=False, init="zero")

    def forward(
        self,
        msa: Float[torch.Tensor, "B N L d_msa"],
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        N = msa.shape[1]

        # Compute left and right transformations
        msa = self.ln_msa(msa) / (N**0.5)
        left = self.to_left(msa)
        right = self.to_right(msa) / (N**0.5)
        left, right = (rearrange(t, "B N L D -> B L D N") for t in (left, right))
        out = torch.einsum("blin,bkjn->blkij", left, right)

        out_flat = rearrange(out, "B L1 L2 D1 D2 -> B L1 L2 (D1 D2)")

        return self.to_out(out_flat)


class MSAPairWeightedAveraging(nn.Module):
    """Weighted averaging of msa representations to pair representation.

    Parameters
    ----------
    d_msa: int
        Dimension of msa representation.
    d_pair: int
        Dimension of pair representation.
    d_hidden: int
        Dimension of hidden layer.

    """

    def __init__(
        self,
        d_msa: int = 64,
        d_hidden: int = 32,
        d_pair: int = 128,
        n_head: int = 8,
        implementation: Literal["pytorch", "triton"] = "pytorch",
        to_bias_init: Literal["zero", "default"] = "zero",
    ) -> None:
        super().__init__()
        self.n_head = n_head

        self.ln_msa = LayerNorm(d_msa, implementation=implementation)
        self.to_value = Linear(d_msa, d_hidden * n_head, bias=False)
        self.to_bias = Linear(d_pair, n_head, bias=False, init=to_bias_init)
        self.to_gate = Linear(d_msa, n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_msa, bias=False, init="zero")
        self.sigmoid_gate = SigmoidGateFunction(implementation=implementation)

    def forward(
        self,
        msa: Float[torch.Tensor, "B N L d_msa"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B N L d_msa"]:
        """Forward pass."""
        B, N, L, _ = msa.shape

        msa = self.ln_msa(msa)
        value = self.to_value(msa)
        value = rearrange(value, "B N L (H D) -> B N H L D", H=self.n_head)
        bias = self.to_bias(pair)
        bias = rearrange(bias, "B L1 L2 H -> B H L1 L2")
        if mask is not None:
            bias.masked_fill_(~mask[:, None, None, :], float("-inf"))
        attention = F.softmax(bias, dim=-1)
        out = torch.einsum("bhlk,bnhkd->bnhld", attention, value)

        gate = self.to_gate(msa)
        gate = rearrange(gate, "B N L H -> B N H L 1")
        out = self.sigmoid_gate(gate, out)
        out = rearrange(out, "B N H L D -> B N L (H D)")
        return self.to_out(out)


class AugmentedAttention(nn.Module):
    """A single block of the Attention Pair Bias model."""

    def __init__(
        self,
        d_single_rep: int,
        d_single_cond: int,
        d_pair_cond: int,
        n_head: int,
        level: Literal["token", "atom"] = "token",
        use_beta: bool = False,
        implementation: Literal["pytorch", "triton"] = "pytorch",
        to_bias_init: Literal["zero", "default"] = "zero",
    ) -> None:
        super().__init__()
        if level not in ["token", "atom"]:
            msg = f"Invalid level: {level}. Choose 'token' or 'atom'."
            raise ValueError(msg)
        self.d_single_rep = d_single_rep
        self.d_single_cond = d_single_cond
        self.d_pair_cond = d_pair_cond
        self.n_head = n_head
        self.use_beta = use_beta
        if use_beta and level != "atom":
            msg = "use_beta can only be True when level is 'atom'."
            raise ValueError(msg)
        self.implementation = implementation
        self.level = level

        d_hidden = self.d_single_rep // self.n_head

        self.last_conditioning = Linear(
            self.d_single_cond,
            self.d_single_rep,
            init="default",
            bias=True,
        )
        self.last_conditioning.bias.data.fill_(-2.0)

        self.to_query = LinearRMSNorm(self.d_single_rep, self.n_head * d_hidden, bias=True)
        self.to_key = LinearRMSNorm(self.d_single_rep, self.n_head * d_hidden, bias=False)
        self.to_value = Linear(self.d_single_rep, self.n_head * d_hidden, bias=False)

        self.ln_query = AdaptiveLayerNorm(
            d_rep=self.n_head * d_hidden,
            d_cond=self.d_single_cond,
            implementation="pytorch",
        )

        self.ln_key = AdaptiveLayerNorm(
            d_rep=self.n_head * d_hidden,
            d_cond=self.d_single_cond,
            implementation="pytorch",
        )

        self.to_gate = Linear(self.d_single_rep, self.n_head, init="gating", bias=False)
        self.ln_pair = LayerNorm(self.d_pair_cond, implementation=self.implementation)
        self.W_bias = Parameter(self.n_head, self.d_pair_cond, init=to_bias_init)
        self.to_out = Linear(
            self.n_head * d_hidden, self.d_single_rep, init="zero", bias=False,
        )
        self.sigmoid_gate = SigmoidGateFunction(implementation=self.implementation)

    def _kernel_attention_pair_bias(
        self,
        query: Float[torch.Tensor, "A B H L D"],
        key: Float[torch.Tensor, "A B H L D"],
        value: Float[torch.Tensor, "A B H L D"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "A B H L D"]:
        if self.implementation == "pytorch":
            query.mul_(query.shape[-1] ** -0.5)
            attention = torch.einsum("abhld,abhkd->abhlk", query, key)
            bias = F.linear(pair, self.W_bias, bias=None)
            bias = rearrange(
                bias, "B L1 L2 H -> 1 B H L1 L2", H=self.n_head,
            ).contiguous()
            if mask is not None:
                bias.masked_fill_(~mask[None, :, None, None, :], float("-inf"))
            attention = attention + bias
            attention = F.softmax(attention, dim=-1)
            return torch.einsum("abhlk,abhkd->abhld", attention, value)

        if self.implementation == "triton" and self.level == "atom":
            return triton_atom_augmented_attention(
                query, key, value, self.W_bias, pair, mask,
            )

        if self.implementation == "triton" and self.level == "token":
            bias = F.linear(pair, self.W_bias, bias=None)
            if mask is not None:
                bias.masked_fill_(~mask[:, None, :, None], float("-inf"))
            return triton_token_augmented_attention(query, key, value, bias)

        msg = (
            f"Invalid implementation: {self.implementation}. "
            "Choose either 'pytorch' or 'triton'."
        )
        raise ValueError(msg)

    def check_input(
        self,
        single_rep: torch.Tensor,
        single_cond: torch.Tensor,
        pair: torch.Tensor | None,
        mask: torch.Tensor | None,
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Check and adjust input shapes."""
        if pair.ndim != 4:
            msg = f"{pair.shape=} must be (B, L, L, d_pair)"
            raise ValueError(msg)
        B, L1, L2, _ = pair.shape
        if L1 != L2:
            msg = f"{L1=} must be equal to {L2=}"
            raise ValueError(msg)
        if mask is not None and mask.shape != (B, L1):
                msg = f"{mask.shape=} must be (B, L)"
                raise ValueError(msg)

        if single_rep.ndim == 3:  # (B, L, d_single)
            single_rep = single_rep.unsqueeze(0)  # (1, B, L, d_single)
            single_cond = single_cond.unsqueeze(0)  # (1, B, L, d_single)

        return single_rep, single_cond, pair

    @typecheck
    def _self_attention(
        self,
        single_rep: Float[torch.Tensor, "B L d_single"],
        single_cond: Float[torch.Tensor, "B L d_single"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L d_single"]:
        original_shape = single_rep.shape
        single_rep, single_cond, pair = self.check_input(
            single_rep, single_cond, pair, mask,
        )
        query_norm, key_norm = (
            self.ln_query(single_rep, single_cond),
            self.ln_key(single_rep, single_cond),
        )
        query, key, value = (
            self.to_query(query_norm),
            self.to_key(key_norm),
            self.to_value(key_norm),
        )

        # to reduce memory usage, head must come before augmentation
        query, key, value = (
            rearrange(t, "A B L (H D) -> A B H L D", H=self.n_head).contiguous()
            for t in (query, key, value)
        )

        pair = self.ln_pair(pair)
        out = self._kernel_attention_pair_bias(query, key, value, pair, mask)
        gate = self.to_gate(query_norm)
        gate = rearrange(gate, "A B L H -> A B H L 1")
        out = self.sigmoid_gate(gate, out)
        out = rearrange(out, "A B H L D -> A B L (H D)")
        out = self.to_out(out)

        last_conditioning = self.last_conditioning(single_cond)
        out = self.sigmoid_gate(last_conditioning, out)
        return out.view(original_shape)  # Restore original shape

    @typecheck
    def forward(
        self,
        noisy_batch: NoisyBatch,  # noqa: ARG002
        single_rep: Float[torch.Tensor, "B L d_single"],
        single_cond: Float[torch.Tensor, "B L d_single"],
        pair: Float[torch.Tensor, "B L L d_pair"] | None = None,
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L d_single"]:
        """Forward pass."""
        if not self.use_beta:
            return self._self_attention(
                single_rep, single_cond, pair, mask,
            )
        msg = "Beta attention is not implemented yet."
        raise NotImplementedError(msg)
