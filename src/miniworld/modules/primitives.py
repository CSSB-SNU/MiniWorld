from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from team_gm import typecheck
from team_gm.modules.exceptions import InvalidImplementationError
from team_gm.modules.primitives import (
    Linear,
    Parameter,
)
from torch import nn

from miniworld.modules.moe_utils import group_by_expert, loss_free_route, scatter_expert

from . import kernels


class SigmoidGateFunction(nn.Module):
    """Sigmoid gating function."""

    def __init__(
        self,
        implementation: Literal["pytorch", "triton"] = "pytorch",
    ) -> None:
        super().__init__()
        self.implementation = implementation

    @typecheck
    def forward(
        self,
        gate: Float[torch.Tensor, "* 1"],
        rep: Float[torch.Tensor, "* d_hidden"],
    ) -> Float[torch.Tensor, "* d_hidden"]:
        """Forward pass."""
        if self.implementation == "pytorch":
            return torch.sigmoid(gate) * rep
        if self.implementation == "triton":
            return kernels.triton_sigmoid_gate(gate, rep)
        msg = f"Invalid implementation: {self.implementation}. Choose either 'pytorch' or 'triton'."
        raise ValueError(msg)



class SwishFunction(nn.Module):
    """Swish activation function."""

    def __init__(
        self,
        implementation: Literal["pytorch", "triton"] = "pytorch",
    ) -> None:
        super().__init__()
        self.implementation = implementation

    @typecheck
    def forward(
        self,
        x: Float[torch.Tensor, "*"],
        y: Float[torch.Tensor, "*"],
    ) -> Float[torch.Tensor, "*"]:
        """Forward pass."""
        if self.implementation == "pytorch":
            return x * torch.sigmoid(x) * y
        if self.implementation == "triton":
            return kernels.triton_swish(x, y)
        msg = f"Invalid implementation: {self.implementation}. Choose either 'pytorch' or 'triton'."
        raise ValueError(msg)


class LayerNorm(nn.Module):
    """Custom triton kernel for LayerNorm.

    Parameters
    ----------
    normalized_shape:  int | tuple[int]
        Int or tuple with one element (only 1D normalization is supported)
    eps: float
        Small value to avoid division by zero.
    elementwise_affine: bool
        if True, learnable scale and bias parameters are used.
    implementation: Literal["pytorch", "triton"]
        Implementation to use. Choose either 'pytorch' or 'triton'.
    dtype: torch.dtype
        The desired data type of the layer. Defaults to `torch.float32`.

    """

    def __init__(
        self,
        normalized_shape: int | tuple[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        implementation: Literal["pytorch", "triton"] = "pytorch",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # Ensure normalized_shape is a tuple.
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        if len(normalized_shape) != 1:
            msg = "LayerNorm only supports 1D normalized shape"
            raise ValueError(msg)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.implementation = implementation

        if self.elementwise_affine:
            # Weight and bias should be 1D tensors of size normalized_shape[0].
            self.weight = Parameter(normalized_shape[0], dtype=dtype, init="one")
            if bias:
                self.bias = Parameter(normalized_shape[0], dtype=dtype, init="zero")
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @typecheck
    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        """Forward pass."""
        if self.implementation == "pytorch":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )
        if self.implementation == "triton":
            return kernels.triton_layernorm(
                x,
                self.weight,
                self.bias,
                self.eps,
            )
        msg = f"Invalid implementation: {self.implementation}. Choose either 'pytorch' or 'triton'."
        raise ValueError(msg)


class LinearLayerNorm(nn.Module):
    """Linear layer followed by LayerNorm.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool
        If set to `False`, the layer will not learn an additive bias.
    init: str
        The initializer to use. See `Linear` class for options.
    eps: float
        Small value to avoid division by zero.
    elementwise_affine: bool
        If True, learnable scale and bias parameters are used.
    implementation: Literal["pytorch", "triton"]
        Implementation to use. Choose either 'pytorch' or 'triton'.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        init: str = "default",
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        implementation: Literal["pytorch", "triton"] = "pytorch",
    ) -> None:
        super().__init__()
        self.linear = Linear(
            in_features,
            out_features,
            bias=bias,
            init=init,
        )
        self.layer_norm = LayerNorm(
            normalized_shape=(out_features,),
            eps=eps,
            elementwise_affine=elementwise_affine,
            implementation=implementation,
        )

    @typecheck
    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        """Forward pass."""
        return self.layer_norm(self.linear(x))


class LinearRMSNorm(nn.Module):
    """Linear layer followed by RMSNorm.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool
        If set to `False`, the layer will not learn an additive bias.
    init: str
        The initializer to use. See `Linear` class for options.
    eps: float
        Small value to avoid division by zero.
    elementwise_affine: bool
        If True, learnable scale and bias parameters are used.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        init: str = "default",
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.linear = Linear(
            in_features,
            out_features,
            bias=bias,
            init=init,
        )
        self.rms_norm = torch.nn.RMSNorm(
            normalized_shape=(out_features,),
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    @typecheck
    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        """Forward pass."""
        # I don't know why but torch RMSNorm always output float16
        # so we have to manually convert it
        input_dtype = x.dtype
        return self.rms_norm(self.linear(x)).to(input_dtype)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization."""

    def __init__(
        self,
        d_rep: int,
        d_cond: int,
        implementation: Literal["pytorch", "triton"] = "pytorch",
    ) -> None:
        super().__init__()
        self.layer_norm_rep = LayerNorm(
            d_rep, elementwise_affine=False, implementation=implementation,
        )
        self.layer_norm_cond = LayerNorm(
            d_cond,
            elementwise_affine=True,
            bias=False,
            implementation=implementation,
        )

        self.cond_scale = Linear(d_cond, d_rep, init="gating")
        self.cond_bias = Linear(d_cond, d_rep, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        rep: Float[torch.Tensor, "* d_rep"],  # (..., d_rep)
        cond: Float[torch.Tensor, "* d_cond"],  # (..., d_cond)
    ) -> Float[torch.Tensor, "* d_rep"]:
        """Forward pass."""
        rep = self.layer_norm_rep(rep)
        cond = self.layer_norm_cond(cond)

        scale = self.cond_scale(cond)
        bias = self.cond_bias(cond)

        return F.sigmoid(scale) * rep + bias


class Transition(nn.Module):
    """Transition layer (Algorithm 11).

    Parameters
    ----------
    d_hidden : int
        Dimension of the input and output features.
    n : int
        Expansion factor.
    implementation : Literal["pytorch", "triton"]
        Implementation to use.
    eps : float
        Small value to avoid division by zero.

    """

    def __init__(
        self,
        d_hidden: int = 128,
        n: int = 4,
        implementation: Literal["pytorch", "triton"] = "pytorch",
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.n = n
        self.implementation = implementation

        self.ln_in = nn.LayerNorm(d_hidden)
        self.expand_a_weight = Parameter(n * d_hidden, d_hidden, init="glorot")
        self.expand_b_weight = Parameter(n * d_hidden, d_hidden, init="glorot")
        self.squeeze_weight = Parameter(d_hidden, d_hidden * n, init="zero")
        self.swish = SwishFunction(implementation=implementation)

    @typecheck
    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        """Forward pass."""
        x = self.ln_in(x)

        if self.implementation == "pytorch":
            a = F.linear(x, self.expand_a_weight)
            b = F.linear(x, self.expand_b_weight)
            x = self.swish(a, b)
            return F.linear(x, self.squeeze_weight)

        if self.implementation == "triton":
            return kernels.triton_transition(  # pyright: ignore[reportReturnType]
                x,
                self.expand_a_weight,
                self.expand_b_weight,
                self.squeeze_weight,
                self.n,
            )

        raise InvalidImplementationError(self.implementation)



class ConditionedTransition(nn.Module):
    """Transition layer (Algorithm 11).

    Parameters
    ----------
    d_hidden: int
        Dimension of the input and output features.
    n: int
        Expansion factor.

    """

    def __init__(
        self,
        d_rep: int = 128,
        d_cond: int = 128,
        n: int = 2,
        implementation: Literal["pytorch", "triton"] = "pytorch",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.use_checkpoint = use_checkpoint
        self.implementation = implementation

        self.ada_ln = AdaptiveLayerNorm(
            d_rep=d_rep,
            d_cond=d_cond,  # No conditioning for this layer
            implementation="pytorch",
        )

        self.expand_a_weight = Parameter(
            n * d_rep, d_rep,
            init="relu",
        )
        self.expand_b_weight = Parameter(n * d_rep, d_rep, init="relu")
        self.squeeze_weight = Parameter(d_rep, d_rep * n, init="default")
        self.last_conditioning = Linear(d_cond, d_rep, bias=True, init="zero")
        self.last_conditioning.bias.data.fill_(-2.0)
        self.swish = SwishFunction(implementation=implementation)
        self.sigmoid_gate = SigmoidGateFunction(implementation=implementation)

    def _forward(
        self,
        atom_single_rep: torch.Tensor,  # (..., d_hidden)
        atom_single_cond: torch.Tensor | None = None,  # (..., d_hidden)
    ) -> torch.Tensor:  # (..., d_hidden)
        atom_single_rep = self.ada_ln(atom_single_rep, atom_single_cond)
        a = F.linear(atom_single_rep, self.expand_a_weight, bias=None)
        b = F.linear(atom_single_rep, self.expand_b_weight, bias=None)
        a = self.swish(a, b)
        atom_single_rep = F.linear(a, self.squeeze_weight, bias=None)
        last_conditioning = self.last_conditioning(atom_single_cond)
        return self.sigmoid_gate(last_conditioning, atom_single_rep)


    @typecheck
    def forward(
        self,
        atom_single_rep: Float[torch.Tensor, "* d_rep"],
        atom_single_cond: Float[torch.Tensor, "* d_cond"] | None = None,
    ) -> Float[torch.Tensor, "* d_rep"]:
        """Forward pass."""
        if atom_single_cond is None:
            msg = "atom_single_cond must be provided"
            raise ValueError(msg)

        if self.use_checkpoint:
            atom_single_rep = torch.utils.checkpoint.checkpoint(
                self._forward, atom_single_rep, atom_single_cond, use_reentrant=False,
            )
        else:
            atom_single_rep = self._forward(atom_single_rep, atom_single_cond)
        return atom_single_rep


class _UpdateExpertFrequencyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,  # noqa: ANN001
        x: torch.Tensor,
        expert_frequency: torch.Tensor,
        topk_indices: torch.Tensor,
        momentum: float,
    ) -> torch.Tensor:
        # Save what we need for the backward side-effect update
        ctx.save_for_backward(topk_indices, expert_frequency)
        ctx.momentum = momentum
        return x  # identity pass-through

    @staticmethod
    def backward(
        ctx,  # noqa: ANN001
        grad_x: torch.Tensor,  # trick : I don't use it
    )-> tuple[torch.Tensor, torch.Tensor, None, None]:
        (topk_indices, expert_frequency) = ctx.saved_tensors
        # Update expert_frequency with EMA, no grad tracking
        pooled = topk_indices.reshape(-1)
        expert_counts = pooled.bincount(minlength=expert_frequency.size(0))
        fwd_expert_frequency = expert_counts.float() / expert_counts.sum()
        grad_f = (expert_frequency-fwd_expert_frequency)
        grad_f = grad_f - grad_f.mean()
        return grad_x, grad_f, None, None


update_expert_frequency = _UpdateExpertFrequencyFn.apply


class MoETransition(nn.Module):
    """Transition layer (Algorithm 11).

    Parameters
    ----------
    d_hidden: int
        Dimension of the input and output features.
    n: int
        Expansion factor.

    """

    def __init__(
        self,
        d_hidden: int = 128,
        n: int = 4,
        experts: int = 8,
        topk: int = 2,
        implementation: Literal["pytorch", "triton"] = "pytorch",
        use_checkpoint: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.implementation = implementation
        self.eps = eps
        self.use_checkpoint = use_checkpoint
        self.n = n
        self.d_hidden = d_hidden
        self.experts = experts
        self.topk = topk
        self.block_k = 128

        # loss free balancing
        self.expert_frequency = Parameter(experts, init="one")
        self.expert_frequency.data = self.expert_frequency.data / experts
        self.expert_freq_momentum = 0.99

        self.ln_weight = Parameter(d_hidden, init="one")
        self.ln_bias = Parameter(d_hidden, init="zero")

        self.router_weight = Parameter(experts, d_hidden, init="default")
        self.expand_a_weight = Parameter(experts, n * d_hidden, d_hidden, init="relu")
        self.expand_b_weight = Parameter(experts, n * d_hidden, d_hidden, init="relu")
        self.squeeze_weight = Parameter(experts, d_hidden, d_hidden * n, init="zero")
        self.swish = SwishFunction(implementation="pytorch")

    @typecheck
    def _forward(
        self, x: Float[torch.Tensor, "*"],
    ) -> tuple[Float[torch.Tensor, "*"], Int[torch.Tensor, "* topk"]]:
        original_shape = x.shape
        x = rearrange(x, "... d -> (...) d").contiguous()
        op_dtype = x.dtype
        if self.implementation == "pytorch":
            # for H100, simple pytorch kernel is faster than triton kernel
            # 20250831 psk
            # this kernel is developed by pytorch
            # https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/

            x = F.layer_norm(x, (self.d_hidden,), self.ln_weight, self.ln_bias, self.eps)
            x = x.to(op_dtype)  # autocast doesn't support for layernorm
            topk_score, topk_indices = loss_free_route(
                self.expert_frequency, self.router_weight, x, self.topk,
            )
            topk_score = topk_score.to(op_dtype)
            sorted_y, sorted_score, idx_map, expert_map, _, _ = group_by_expert(
                x, topk_score, topk_indices, padding=self.block_k,
            )
            a = kernels.cg_grouped_gemm(
                sorted_y, self.expand_a_weight, expert_map, group_size_m=self.block_k,
            )
            b = kernels.cg_grouped_gemm(
                sorted_y, self.expand_b_weight, expert_map, group_size_m=self.block_k,
            )
            y_expanded = self.swish(a, b)
            # to do precision manager for cg_grouped_gemm
            y_squeezed = kernels.cg_grouped_gemm(
                y_expanded, self.squeeze_weight, expert_map, group_size_m=self.block_k,
            )
            y_squeezed = y_squeezed * sorted_score.unsqueeze(-1)

            torch.save(idx_map, "idx_map_torch.pt")

            x = scatter_expert(x.shape, y_squeezed, idx_map)
            x = x.reshape(original_shape)
            return x, topk_indices
        if self.implementation == "triton":
            x, topk_indices = kernels.triton_MoE_transition(
                x,
                self.ln_weight,
                self.ln_bias,
                self.router_weight,
                self.expand_a_weight,
                self.expand_b_weight,
                self.squeeze_weight,
                self.expert_frequency,
                self.n,
                self.topk,
            )
            x = x.reshape(original_shape)
            return x, topk_indices
        msg = f"Invalid implementation: {self.implementation}. Choose either 'pytorch' or 'triton'."
        raise ValueError(msg)

    @typecheck
    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        """Forward pass."""
        if self.use_checkpoint:
            x, topk_indices = torch.utils.checkpoint.checkpoint(
                self._forward, x, use_reentrant=False,
            )
        else:
            x, topk_indices = self._forward(x)
        return update_expert_frequency(
            x, self.expert_frequency, topk_indices, self.expert_freq_momentum,
        )

class ConditionedMoETransition(nn.Module):
    """Conditioned Mixture of Experts Transition layer.

    Parameters
    ----------
    d_hidden: int
        Dimension of the input and output features.
    n: int
        Expansion factor.

    """

    def __init__(
        self,
        d_rep: int = 128,
        d_cond: int = 128,
        n: int = 2,
        experts: int = 8,
        topk: int = 2,
        implementation: Literal["pytorch", "triton"] = "pytorch",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.experts = experts
        self.topk = topk
        self.implementation = implementation
        self.use_checkpoint = use_checkpoint
        self.block_k = 128

        self.ada_ln = AdaptiveLayerNorm(
            d_rep=d_rep,
            d_cond=d_cond,
            implementation="pytorch",
        )

        # loss free balancing
        self.expert_frequency = Parameter(experts, init="one")
        self.expert_frequency.data = self.expert_frequency.data / experts
        self.expert_freq_momentum = 0.99

        self.expand_a_weight = Parameter(experts, n * d_rep, d_rep, init="relu")
        self.expand_b_weight = Parameter(experts, n * d_rep, d_rep, init="relu")
        self.squeeze_weight = Parameter(experts, d_rep, d_rep * n, init="default")

        self.router_weight = Parameter(experts, d_rep, init="default")
        self.last_conditioning = Linear(d_cond, d_rep, bias=True, init="zero")
        self.last_conditioning.bias.data.fill_(-2.0)

        self.swish = SwishFunction(implementation=implementation)
        self.sigmoid_gate = SigmoidGateFunction(implementation=implementation)

    def _forward(
        self,
        atom_single_rep: torch.Tensor,
        atom_single_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = atom_single_rep.shape
        x = rearrange(atom_single_rep, "... d -> (...) d").contiguous()
        c = rearrange(atom_single_cond, "... d -> (...) d").contiguous()
        x_norm = self.ada_ln(x, c)
        last_cond = self.last_conditioning(c)

        if self.implementation in {"pytorch", "triton"}:
            # for H100, simple pytorch kernel is faster than triton kernel
            topk_score, topk_indices = loss_free_route(
                self.expert_frequency, self.router_weight, x_norm, self.topk,
            )
            sorted_y, sorted_score, idx_map, expert_map, _, _ = group_by_expert(
                x_norm, topk_score, topk_indices, padding=self.block_k,
            )
            a = kernels.cg_grouped_gemm(
                sorted_y, self.expand_a_weight, expert_map, group_size_m=self.block_k,
            )
            b = kernels.cg_grouped_gemm(
                sorted_y, self.expand_b_weight, expert_map, group_size_m=self.block_k,
            )
            y_expanded = self.swish(a, b)
            y_expanded = y_expanded.to(self.squeeze_weight.dtype)
            y_squeezed = kernels.cg_grouped_gemm(
                y_expanded, self.squeeze_weight, expert_map, group_size_m=self.block_k,
            )
            y_squeezed = y_squeezed * sorted_score.unsqueeze(-1)
            valid = idx_map >= 0
            out = torch.zeros_like(x_norm).index_add_(
                0, idx_map[valid], y_squeezed[valid],
            )

        elif self.implementation == "triton":
            out, topk_indices = kernels.triton_MoE_transition_wo_ln(
                x_norm,
                self.router_weight,
                self.expand_a_weight,
                self.expand_b_weight,
                self.squeeze_weight,
                self.expert_frequency,
                self.n,
                self.topk,
            )

        else:
            msg = f"Invalid implementation: {self.implementation}."
            raise ValueError(msg)
        out = update_expert_frequency(
            out, self.expert_frequency, topk_indices, self.expert_freq_momentum,
        )
        out = self.sigmoid_gate(last_cond, out)
        return out.reshape(original_shape)

    @typecheck
    def forward(
        self,
        atom_single_rep: Float[torch.Tensor, "* d_rep"],
        atom_single_cond: Float[torch.Tensor, "* d_cond"] | None = None,
    ) -> Float[torch.Tensor, "* d_rep"]:
        """Forward pass."""
        if atom_single_cond is None:
            msg = "atom_single_cond must be provided"
            raise ValueError(msg)

        if self.use_checkpoint:
            atom_single_rep = torch.utils.checkpoint.checkpoint(
                self._forward, atom_single_rep, atom_single_cond, use_reentrant=False,
            )
        else:
            atom_single_rep = self._forward(
                atom_single_rep, atom_single_cond,
            )
        return atom_single_rep
