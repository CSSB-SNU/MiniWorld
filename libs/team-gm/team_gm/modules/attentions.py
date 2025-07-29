import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from einops import rearrange
from jaxtyping import Float, Bool

from team_gm import typecheck
from . import kernels
from team_gm.data.atom_layout import convert
from team_gm.data.features import NoisyBatch
from .primitives import Linear, LayerNorm, LinearLayerNorm, SigmoidGateFunction, Parameter, AdaptiveLayerNorm


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
    ):
        super().__init__()

        self.ln_msa = LayerNorm(d_msa, implementation=implementation)
        # self.ln_msa = LayerNorm(d_msa, implementation="pytorch")
        self.to_left = Linear(d_msa, d_hidden, False)
        self.to_right = Linear(d_msa, d_hidden, False)
        self.to_out = Linear(d_hidden * d_hidden, d_pair, False, init="final")

    def forward(
        self,
        msa: Float[torch.Tensor, "B N L d_msa"],
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        N = msa.shape[1]

        # Compute left and right transformations
        msa = self.ln_msa(msa) / (N**0.5)
        left = self.to_left(msa)
        right = self.to_right(msa) / (N**0.5)
        left, right = (rearrange(t, "B N L D -> B L D N") for t in (left, right))
        out = torch.einsum("blin,bkjn->blkij", left, right)

        out_flat = rearrange(out, "B L1 L2 D1 D2 -> B L1 L2 (D1 D2)")

        pair = self.to_out(out_flat)
        return pair


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
    ):
        super().__init__()
        self.n_head = n_head

        self.ln_msa = LayerNorm(d_msa, implementation=implementation)
        self.to_value = Linear(d_msa, d_hidden * n_head, False)
        self.to_bias = Linear(d_pair, n_head, False, init=to_bias_init)
        self.to_gate = Linear(d_msa, n_head, False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_msa, False, init="final")
        self.sigmoid_gate = SigmoidGateFunction(implementation=implementation)

    def forward(
        self,
        msa: Float[torch.Tensor, "B N L d_msa"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B N L d_msa"]:
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
        out = self.to_out(out)
        return out


class AttentionPairBias(nn.Module):
    """Attention with pair bias (Algorithm 24).

    Parameters
    ----------
    d_single: int
        Dimension of single representation.
    d_pair: int
        Dimension of pair representation.
    n_head: int
        Number of attention heads.
    """

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 128,
        n_head: int = 8,
        implementation: Literal["pytorch", "triton"] = "pytorch",
        to_bias_init: Literal["zero", "default"] = "zero",
        norm: Literal["pre", "hybrid"] = "pre",
    ):
        super().__init__()
        self.n_head = n_head
        self.implementation = implementation
        self.norm = norm
        if implementation not in ("pytorch", "triton"):
            raise ValueError(
                f"Invalid implementation: {implementation=}."
                "Choose either 'pytorch' or 'triton'."
            )

        assert d_single % n_head == 0, f"{d_single=} must be divisible by {n_head=}"
        d_hidden = d_single // n_head

        # following HybridNorm
        if norm == "pre":
            self.ln_single = LayerNorm(d_single, implementation=implementation)
            self.to_query = Linear(d_single, d_hidden * n_head, bias=True)
            self.to_key = Linear(d_single, d_hidden * n_head, bias=False)
            self.to_value = Linear(d_single, d_hidden * n_head, bias=False)
        elif norm == "hybrid":
            self.to_query = LinearLayerNorm(
                d_single, d_hidden * n_head, implementation=implementation, bias=True
            )
            self.to_key = LinearLayerNorm(
                d_single, d_hidden * n_head, implementation=implementation, bias=False
            )
            self.to_value = LinearLayerNorm(
                d_single, d_hidden * n_head, implementation=implementation, bias=False
            )
            self.final_norm = LayerNorm(d_single, implementation=implementation)

        self.ln_pair = LayerNorm(d_pair, implementation=implementation)
        self.to_bias = Linear(d_pair, n_head, False, init=to_bias_init)
        self.to_gate = Linear(d_single, n_head, False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_single, False, init="final")
        self.sigmoid_gate = SigmoidGateFunction(implementation=implementation)

    def _kernel_attention_pair_bias(
        self,
        query: Float[torch.Tensor, "B H L D"],
        key: Float[torch.Tensor, "B H L D"],
        value: Float[torch.Tensor, "B H L D"],
        bias: Float[torch.Tensor, "B H L L"],
    ) -> torch.Tensor:
        if self.implementation == "pytorch":
            query.mul_(query.shape[-1] ** -0.5)
            attention = torch.einsum("bhld,bhkd->bhlk", query, key)
            attention = attention + bias
            attention = F.softmax(attention, dim=-1)
            out = torch.einsum("bhlk,bhkd->bhld", attention, value)
            return out

        elif self.implementation == "triton":
            return kernels.triton_attention_pair_bias(query, key, value, bias)

        else:
            raise ValueError(
                f"Invalid implementation: {self.implementation}. "
                "Choose either 'pytorch' or 'triton'."
            )

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "B L d_single"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L d_single"]:
        # assert single.ndim == 3, f"{single.shape=} must be (B, L, d_single)"
        # assert pair.ndim == 4, f"{pair.shape=} must be (B, L, L, d_pair)"
        B, L1, L2, _ = pair.shape
        # assert L1 == L2, f"{L1=} must be equal to {L2=}"
        # assert mask is None or mask.shape == (B, L1), f"{mask.shape=} must be {B, L1}"
        # assert single.shape[:2] == (B, L1), (
        #     f"{single.shape[:2]=} must be equal to {B, L1}"
        # )

        if self.norm == "pre":
            single = self.ln_single(single)

        query = self.to_query(single)
        key = self.to_key(single)
        value = self.to_value(single)
        query = rearrange(query, "B L (H D) -> B H L D", H=self.n_head)
        key = rearrange(key, "B L (H D) -> B H L D", H=self.n_head)
        value = rearrange(value, "B L (H D) -> B H L D", H=self.n_head)

        pair = self.ln_pair(pair)
        bias = self.to_bias(pair)
        bias = rearrange(bias, "B L L2 H -> B H L L2")
        if mask is not None:
            bias.masked_fill_(~mask[:, None, None, :], float("-inf"))
        out = self._kernel_attention_pair_bias(query, key, value, bias)

        if self.norm == "pre":
            gate = self.to_gate(single)
        elif self.norm == "hybrid":
            gate = self.to_gate(query)
        gate = rearrange(gate, "B L H -> B H L 1")
        out = self.sigmoid_gate(gate, out)
        out = rearrange(out, "B H L D -> B L (H D)")
        out = self.to_out(out)
        if self.norm == "hybrid":
            out = self.final_norm(out)
        return out


class TriangleMultiplication(nn.Module):
    """Unified implementation of triangular multiplicative update.

    Parameters
    ----------
    d_pair: int
        Dimension of pair representation.
    d_hidden: int
        Dimension of hidden layer.
    outgoing: bool
        Whether to use outgoing edges.
    implementation: str
        Implementation type, either "pytorch" or "triton".
    """

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int = 128,
        outgoing: bool = True,
        implementation: Literal["pytorch", "triton"] = "pytorch",
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.outgoing = outgoing
        self.implementation = implementation
        if implementation not in ("pytorch", "triton"):
            raise ValueError(
                f"Invalid implementation: {implementation=}."
                "Choose either 'pytorch' or 'triton'."
            )

        self.ln_pair = LayerNorm(d_pair, implementation=implementation)
        self.left_weight = Parameter((d_pair, d_hidden), init="default")
        self.left_gate_weight = Parameter((d_pair, d_hidden), init="gating")
        self.right_weight = Parameter((d_pair, d_hidden), init="default")
        self.right_gate_weight = Parameter((d_pair, d_hidden), init="gating")

        self.ln_weight = Parameter(d_hidden, init="one")
        self.ln_bias = Parameter(d_hidden, init="zero")
        self.gate_weight = Parameter((d_hidden, d_pair), init="gating")
        self.out_weight = Parameter((d_hidden, d_pair), init="final")

    def _kernel_tm1(self, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.implementation == "pytorch":
            left = F.linear(pair, self.left_weight.T)
            right = F.linear(pair, self.right_weight.T)
            left_gate = F.linear(pair, self.left_gate_weight.T)
            right_gate = F.linear(pair, self.right_gate_weight.T)
            left = F.sigmoid(left_gate) * left
            right = F.sigmoid(right_gate) * right
            return left, right

        elif self.implementation == "triton":
            return kernels.triton_tm1(
                pair,
                self.left_weight,
                self.left_gate_weight,
                self.right_weight,
                self.right_gate_weight,
            )

        raise ValueError(
            f"Invalid implementation: {self.implementation=}. "
            "Choose either 'pytorch' or 'triton'."
        )

    def _kernel_tm2(self, pair: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self.implementation == "pytorch":
            out = F.layer_norm(out, (self.d_hidden,), self.ln_weight, self.ln_bias)
            out = F.linear(out, self.out_weight.T)
            gate = F.linear(pair, self.gate_weight.T)
            out = F.sigmoid(gate) * out
            return out

        elif self.implementation == "triton":
            return kernels.triton_tm2(
                pair,
                out,
                self.ln_weight,
                self.ln_bias,
                self.gate_weight,
                self.out_weight,
            )

        raise ValueError(
            f"Invalid implementation: {self.implementation=}. "
            "Choose either 'pytorch' or 'triton'."
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        pair = self.ln_pair(pair)
        left, right = self._kernel_tm1(pair)

        if mask is not None:
            mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            left = left.masked_fill(~mask_2d[..., None], 0)
            right = right.masked_fill(~mask_2d[..., None], 0)

        if self.outgoing:
            out = torch.einsum("bikd,bjkd->bijd", left, right)
        else:
            out = torch.einsum("bkid,bkjd->bijd", left, right)

        return self._kernel_tm2(pair, out)


class TriangleAttention(nn.Module):
    """Unified implementation of triangular gated self-attention.

    Parameters
    ----------
    d_pair: int
        Dimension of pair representation.
    d_hidden: int
        Dimension of hidden layer.
    n_head: int
        Number of attention heads.
    starting: bool
        Whether the attention is around the "starting" node.
    use_self_attention: bool
        Whether to use self-attention.
    implementation: str
        Implementation type, either "pytorch" or "triton".
    """

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int = 32,
        n_head: int = 4,
        starting: bool = True,
        use_self_attention: bool = True,
        implementation: Literal["pytorch", "triton"] = "pytorch",
        to_bias_init: Literal["zero", "default"] = "zero",
        norm: Literal["pre", "hybrid"] = "pre",
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.starting = starting
        self.use_self_attention = use_self_attention
        self.implementation = implementation
        if implementation not in ("pytorch", "triton"):
            raise ValueError(
                f"Invalid implementation: {implementation=}."
                "Choose either 'pytorch' or 'triton'."
            )

        if norm == "pre":
            self.ln_pair = LayerNorm(d_pair, implementation=implementation)
            if use_self_attention:
                self.to_query = Linear(d_pair, d_hidden * n_head, bias=False)
                self.to_key = Linear(d_pair, d_hidden * n_head, bias=False)
            self.to_value = Linear(d_pair, d_hidden * n_head, bias=False)
        elif norm == "hybrid":
            if use_self_attention:
                self.to_query = LinearLayerNorm(
                    d_pair, d_hidden * n_head, implementation=implementation, bias=False
                )
                self.to_key = LinearLayerNorm(
                    d_pair, d_hidden * n_head, implementation=implementation, bias=False
                )
            self.to_value = LinearLayerNorm(
                d_pair, d_hidden * n_head, implementation=implementation, bias=False
            )
            self.final_norm = LayerNorm(d_pair, implementation=implementation)

        self.to_bias = Linear(d_pair, n_head, False, init=to_bias_init)
        self.to_gate = Linear(d_pair, d_hidden * n_head, False, init="gating")
        self.out_weight = Parameter((d_hidden * n_head, d_pair), init="final")

    def _kernel_triangle_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        if self.implementation == "pytorch":
            query.mul_(self.d_hidden**-0.5)
            attention = torch.einsum("bhijd,bhikd->bhijk", query, key)
            attention = attention + bias[:, :, None, :, :]
            attention = F.softmax(attention, dim=-1)
            out = torch.einsum("bhijk,bhikd->bhijd", attention, value)
            return out

        elif self.implementation == "triton":
            return kernels.triton_triangle_attention_pair_bias(
                query,
                key,
                value,
                bias,
            )

        raise ValueError(
            f"Invalid implementation: {self.implementation=}. "
            "Choose either 'pytorch' or 'triton'."
        )

    def _kernel_post_attention(
        self,
        gate: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if self.implementation == "pytorch":
            out = torch.sigmoid(gate) * out
            out = F.linear(out, self.out_weight.T)
            return out

        elif self.implementation == "triton":
            return kernels.triton_post_bias_attention(gate, out, self.out_weight)

        raise ValueError(
            f"Invalid implementation: {self.implementation=}. "
            "Choose either 'pytorch' or 'triton'."
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:

        assert pair.ndim == 4, f"{pair.shape=} must be (B, L, L, d_pair)"
        B, L1, L2, _ = pair.shape
        assert L1 == L2, f"{L1=} must be equal to {L2=}"
        assert mask is None or mask.shape == (B, L1), f"{mask.shape=} must be {B, L1}"

        if self.norm == "pre":
            pair = self.ln_pair(pair)
        if not self.starting:
            pair = rearrange(pair, "B I J D -> B J I D").contiguous()
        value = self.to_value(pair)
        bias = self.to_bias(pair)

        value = rearrange(value, "B L L2 (H D) -> B H L L2 D", H=self.n_head)
        bias = rearrange(bias, "B L L2 H -> B H L L2")

        if mask is not None:
            bias.masked_fill_(~mask[:, None, None, :], float("-inf"))

        if self.use_self_attention:
            query = self.to_query(pair)
            key = self.to_key(pair)

            query = rearrange(query, "B L L2 (H D) -> B H L L2 D", H=self.n_head)
            key = rearrange(key, "B L L2 (H D) -> B H L L2 D", H=self.n_head)
            out = self._kernel_triangle_attention(query, key, value, bias)
        else:
            attention = F.softmax(bias, dim=-1)
            out = torch.einsum("bhjk,bhikd->bhijd", attention, value)

        if self.norm == "pre":
            gate = self.to_gate(pair)
        elif self.norm == "hybrid":
            gate = self.to_gate(query)
        out = rearrange(out, "B H L L2 D -> B L L2 (H D)")
        out = self._kernel_post_attention(gate, out)

        if not self.starting:
            out = rearrange(out, "B J I D -> B I J D")
        out = out.contiguous()
        if self.norm == "hybrid":
            out = self.final_norm(out)
        return out



class AugmentedAttention(nn.Module):
    """
    A single block of the Attention Pair Bias model.
    """

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
        norm: Literal["pre", "hybrid"] = "pre",
    ):
        super().__init__()
        assert level in ["token", "atom"], (
            f"Invalid level: {level}. Choose 'token' or 'atom'."
        )
        self.d_single_rep = d_single_rep
        self.d_single_cond = d_single_cond
        self.d_pair_cond = d_pair_cond
        self.n_head = n_head
        self.use_beta = use_beta
        assert level == "atom" or not use_beta, (
            "use_beta can only be True when level is 'atom'."
        )
        self.implementation = implementation
        self.level = level
        self.norm = norm

        d_hidden = self.d_single_rep // self.n_head

        self.last_conditioning = Linear(
            self.d_single_cond,
            self.d_single_rep,
            init="default",
            bias=True,
        )
        # biasinit = -2.0
        self.last_conditioning.bias.data.fill_(-2.0)

        if norm == "hybrid":
            self.ln_value = AdaptiveLayerNorm(
                d_rep=self.n_head * d_hidden,
                d_cond=self.d_single_cond,
                implementation="pytorch",
            )
            self.final_norm = LayerNorm(
                # self.d_single_rep, implementation=self.implementation
                self.d_single_rep,
                implementation="pytorch",
            )
        self.to_query = Linear(
            self.d_single_rep,
            self.n_head * d_hidden,
            bias=True,
        )
        self.to_key = Linear(
            self.d_single_rep,
            self.n_head * d_hidden,
            bias=False,
        )
        self.to_value = Linear(
            self.d_single_rep,
            self.n_head * d_hidden,
            bias=False,
        )
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
        self.W_bias = Parameter(size=(self.n_head, self.d_pair_cond), init=to_bias_init)
        self.to_out = Linear(
            self.n_head * d_hidden, self.d_single_rep, init="final", bias=False
        )
        self.sigmoid_gate = SigmoidGateFunction(implementation=self.implementation)

    def _kernel_attention_pair_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,  # (B, L)
    ) -> torch.Tensor:
        # if self.implementation == "pytorch" or self.level == "token":
        if self.implementation == "pytorch":
            query.mul_(query.shape[-1] ** -0.5)
            attention = torch.einsum("abhld,abhkd->abhlk", query, key)
            bias = F.linear(pair, self.W_bias, bias=None)
            bias = rearrange(
                bias, "B L1 L2 H -> 1 B H L1 L2", H=self.n_head
            ).contiguous()
            if mask is not None:
                bias.masked_fill_(~mask[None, :, None, None, :], float("-inf"))
            attention = attention + bias
            attention = F.softmax(attention, dim=-1)
            out = torch.einsum("abhlk,abhkd->abhld", attention, value)
            return out

        elif self.implementation == "triton" and self.level == "atom":
            return kernels.triton_atom_augmented_attention(
                query, key, value, self.W_bias, pair, mask
            )

        elif self.implementation == "triton" and self.level == "token":
            bias = F.linear(pair, self.W_bias, bias=None)
            if mask is not None:
                bias.masked_fill_(~mask[:, None, :, None], float("-inf"))
            return kernels.triton_token_augmented_attention(query, key, value, bias)

        else:
            raise ValueError(
                f"Invalid implementation: {self.implementation}. "
                "Choose either 'pytorch' or 'triton'."
            )

    def check_input(
        self,
        single_rep: torch.Tensor,
        single_cond: torch.Tensor,
        pair: torch.Tensor | None,
        mask: torch.Tensor | None,
    ):
        assert pair.ndim == 4, f"{pair.shape=} must be (B, L, L, d_pair)"
        B, L1, L2, _ = pair.shape
        assert L1 == L2, f"{L1=} must be equal to {L2=}"
        if mask is not None:
            assert mask.shape == (B, L1), f"{mask.shape=} must be {B, L1}"

        if single_rep.ndim == 3:  # (B, L, d_single)
            single_rep = single_rep.unsqueeze(0)  # (1, B, L, d_single)
            single_cond = single_cond.unsqueeze(0)  # (1, B, L, d_single)

        return single_rep, single_cond, pair

    def _self_attention(
        self,
        noisy_batch: NoisyBatch,
        single_rep: torch.Tensor,  # (B, L, d_single)
        single_cond: torch.Tensor,  # (B, L, d_single)
        pair: torch.Tensor,  # (B, L, L, d_pair)
        mask: torch.Tensor | None = None,  # (B, L)
    ) -> torch.Tensor:  # (B, L, d_single)
        original_shape = single_rep.shape
        single_rep, single_cond, pair = self.check_input(
            single_rep, single_cond, pair, mask
        )
        if self.norm == "pre":
            query_norm, key_norm = (
                self.ln_query(single_rep, single_cond),
                self.ln_key(single_rep, single_cond),
            )
            query, key, value = (
                self.to_query(query_norm),
                self.to_key(key_norm),
                self.to_value(key_norm),
            )
        elif self.norm == "hybrid":
            query, key, value = (
                self.to_query(single_rep),
                self.to_key(single_rep),
                self.to_value(single_rep),
            )
            query, key, value = (
                self.ln_query(query, single_cond),
                self.ln_key(key, single_cond),
                self.ln_value(value, single_cond),
            )
            query_norm = query

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
        out = out.view(original_shape)  # Restore original shape
        if self.norm == "hybrid":
            out = self.final_norm(out)
        return out

    def _cross_attention(
        self,
        noisy_batch: NoisyBatch,
        single_rep: torch.Tensor,  # (B, L, d_single)
        single_cond: torch.Tensor,  # (B, L, d_single)
        pair: torch.Tensor,  # (B, L, L, d_pair)
        mask: torch.Tensor | None = None,  # (B, L)
    ) -> torch.Tensor:  # (B, L, d_single)
        query_rep, query_cond = [
            convert(
                noisy_batch.cross_att.token_atoms_to_queries, x, layout_axes=(-3, -2)
            )
            for x in (single_rep, single_cond)
        ]
        key_rep, key_cond = [
            convert(noisy_batch.cross_att.queries_to_keys, x, layout_axes=(-3, -2))
            for x in (query_rep, query_cond)
        ]

        query_mask = convert(
            noisy_batch.cross_att.token_atoms_to_queries, mask, layout_axes=(-2, -1)
        )
        key_mask = convert(
            noisy_batch.cross_att.queries_to_keys, mask, layout_axes=(-2, -1)
        )

        # bias: ... x heads (1) x query x key
        bias = (
            1e9
            * (query_mask - 1.0)[..., None, :, None]
            * (key_mask - 1.0)[..., None, None, :]
        )

        x_q = self.ln_query(query_rep, query_cond)
        x_k = self.ln_key(key_rep, key_cond)

        q = self.to_query(x_q)
        k = self.to_key(x_k)
        v = self.to_value(x_k)

        # In AF3, einsum process operate with float 32 tensor

        # logits = torch.einsum('...qhc, ...khc -> ...hqk')
        q.mul_(q.shape[-1] ** -0.5)
        q, k, v = [
            rearrange(x, "a b l i (h d) -> a b l h i d", h=self.n_head)
            for x in (q, k, v)
        ]
        logits = torch.einsum("ablhqd,ablhkd->ablhqk", q, k)
        logits = logits + bias
        if pair is not None:
            logits = logits + pair

        attention = F.softmax(logits, dim=-1)
        out = torch.einsum("ablhqk,ablhkd->ablhd", attention, v)
        gate = self.to_gate(x_q)
        gate = rearrange(gate, "a b l h -> a b l h 1")
        out = self.sigmoid_gate(gate, out)
        out = rearrange(out, "a b l h d -> a b l (h d)")
        out = self.to_out(out)
        last_conditioning = self.last_conditioning(query_cond)
        out = self.sigmoid_gate(last_conditioning, out)

        out = convert(
            noisy_batch.cross_att.queries_to_token_atoms,
            out,
            layout_axes=(-3, -2),
        )

        # Potential shape mismatch

        return out

    def forward(
        self,
        noisy_batch: NoisyBatch,
        single_rep: Float[torch.Tensor, "B L d_single"],  # (B, L, d_single)
        single_cond: Float[torch.Tensor, "B L d_single"],  # (B, L, d_single)
        pair: Float[torch.Tensor, "B L L d_pair"] | None = None,  # (B, L, L, d_pair)
        mask: Bool[torch.Tensor, "B L"] | None = None,  # (B, L)
    ) -> Float[torch.Tensor, "B L d_single"]:  # (B, L, d_single)
        if not self.use_beta:
            return self._self_attention(noisy_batch, single_rep, single_cond, pair, mask)
        else:
            return self._cross_attention(
                noisy_batch, single_rep, single_cond, pair, mask
            )
