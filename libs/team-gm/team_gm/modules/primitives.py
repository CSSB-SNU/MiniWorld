import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.stats import truncnorm
from collections.abc import Sequence
from typing import Literal
from enum import Enum
from functools import partial
from jaxtyping import Float, Int

from team_gm import typecheck
from . import kernels


class SigmoidGateFunction(nn.Module):
    def __init__(
        self,
        implementation: Literal["pytorch", "triton"] = "pytorch",
    ):
        super().__init__()
        self.implementation = implementation

    @typecheck
    def forward(
        self,
        gate: Float[torch.Tensor, "* 1"],
        rep: Float[torch.Tensor, "* d_hidden"],
    ) -> Float[torch.Tensor, "* d_hidden"]:
        if self.implementation == "pytorch":
            return torch.sigmoid(gate) * rep
        elif self.implementation == "triton":
            return kernels.triton_sigmoid_gate(gate, rep)
        raise ValueError(
            f"Invalid implementation: {self.implementation}. "
            "Choose either 'pytorch' or 'triton'."
        )

    def extra_repr(self):
        return f"implementation={self.implementation}"


class SwishFunction(nn.Module):
    def __init__(
        self,
        implementation: Literal["pytorch", "triton"] = "pytorch",
    ):
        super().__init__()
        self.implementation = implementation

    @typecheck
    def forward(
        self,
        x: Float[torch.Tensor, "*"],
        y: Float[torch.Tensor, "*"],
    ) -> Float[torch.Tensor, "*"]:
        if self.implementation == "pytorch":
            return x * torch.sigmoid(x) * y
        elif self.implementation == "triton":
            return kernels.triton_swish(x, y)
        raise ValueError(
            f"Invalid implementation: {self.implementation}. "
            "Choose either 'pytorch' or 'triton'."
        )

    def extra_repr(self):
        return f"implementation={self.implementation}"


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")
    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = weights.numel()
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device, dtype=weights.dtype))


class InitType(Enum):
    """Enum for initialization types."""

    DEFAULT = partial(trunc_normal_init_, scale=1.0)
    RELU = partial(trunc_normal_init_, scale=2.0)
    NORMAL = partial(nn.init.kaiming_normal_, nonlinearity="linear")
    default = partial(nn.init.xavier_uniform_)
    GATING = partial(nn.init.zeros_)
    FINAL = partial(nn.init.zeros_)
    ZERO = partial(nn.init.zeros_)
    ONE = partial(nn.init.ones_)

    def apply(self, weights, bias=None):
        self.value(weights)
        if bias is not None:
            if self == InitType.GATING:
                nn.init.ones_(bias)
            else:
                nn.init.zeros_(bias)


class Linear(nn.Linear):
    """A Linear layer with built-in nonstandard initializations.

    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    bias: bool
        If set to `False`, the layer will not learn an additive bias.
    init: str
        The initializer to use. Supported options are:
        - "default": LeCun fan-in truncated normal initialization.
        - "relu": He initialization using a truncated normal distribution.
        - "default": Fan-average default uniform initialization.
        - "gating": Weights=0, Bias=1.
        - "normal": Normal initialization with standard deviation `1/sqrt(fan_in)`.
        - "final": Weights=0, Bias=0.
    dtype: torch.dtype
        The desired data type of the layer. Defaults to `torch.float32`.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.init = init
        InitType[init.upper()].apply(self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, init={self.__dict__.get('init', None)}"
        )


class Parameter(nn.Parameter):
    """Custom Parameter class with built-in initialization.

    When creating a new Parameter, the provided tensor (typically created with
    torch.empty(...)) is overridden by the selected initializer.

    Parameters
    ----------
    size : Sequence[int] or torch.SymInt
        The shape of the parameter.
    requires_grad : bool
        Whether gradients should be computed. Defaults to True.
    init : str
        The initializer to use. Supported options are:
        - "default": LeCun fan-in truncated normal initialization.
        - "relu": He initialization using a truncated normal distribution.
        - "default": Fan-average default uniform initialization.
        - "gating": Weights=0, Bias=1.
        - "normal": Normal initialization with standard deviation `1/sqrt(fan_in)`.
        - "final": Weights=0, Bias=0.
    dtype: torch.dtype
        The desired data type of the parameter. Defaults to `torch.float32`.

    """

    def __new__(
        cls,
        size: Sequence[int | torch.SymInt],
        requires_grad: bool = True,
        init: str = "default",
        dtype: torch.dtype = torch.float32,
    ) -> "Parameter":
        data = torch.empty(size, dtype=dtype)
        instance = super().__new__(cls, data, requires_grad)
        InitType[init.upper()].apply(instance)
        return instance


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
        dtype=torch.float32,
    ):
        super().__init__()
        # Ensure normalized_shape is a tuple.
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        if len(normalized_shape) != 1:
            raise ValueError("LayerNorm only supports 1D normalized shape")
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

    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        if self.implementation == "pytorch":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )
        elif self.implementation == "triton":
            return kernels.triton_layernorm(
                x,
                self.weight,
                self.bias,
                self.eps,
            )
        raise ValueError(
            f"Invalid implementation: {self.implementation}. "
            "Choose either 'pytorch' or 'triton'."
        )

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, "
            f"implementation={self.implementation}"
        )


class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super().__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(
            torch.tensor([1 - p_drop])
        )
        self.broadcast_dim = broadcast_dim
        self.p_drop = p_drop

    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        if not self.training:  # no drophead during evaluation mode
            return x
        with torch.no_grad():
            shape = list(x.shape)
            if self.broadcast_dim is not None:
                shape[self.broadcast_dim] = 1
            mask = self.sampler.sample(shape).to(x.device).view(shape)
            mask = mask / (1.0 - self.p_drop)
            mask = mask.to(x.dtype)

        x = mask * x
        return x

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
    ):
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

    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        return self.layer_norm(self.linear(x))


class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        d_rep: int,
        d_cond: int,
        implementation: Literal["pytorch", "triton"] = "pytorch",
    ):
        super().__init__()
        self.layer_norm_rep = LayerNorm(
            d_rep, elementwise_affine=False, implementation=implementation
        )
        self.layer_norm_cond = LayerNorm(
            d_cond,
            elementwise_affine=True,
            bias=False,
            implementation=implementation,
        )

        self.cond_scale = Linear(d_cond, d_rep, init="gating")
        self.cond_bias = Linear(d_cond, d_rep, bias=False, init="zero")

    def forward(
        self,
        rep: Float[torch.Tensor, "* d_rep"],  # (..., d_rep)
        cond: Float[torch.Tensor, "* d_cond"],  # (..., d_cond)
    ) -> Float[torch.Tensor, "* d_rep"]:
        rep = self.layer_norm_rep(rep)
        cond = self.layer_norm_cond(cond)

        scale = self.cond_scale(cond)
        bias = self.cond_bias(cond)

        rep = F.sigmoid(scale) * rep + bias
        return rep

class Transition(nn.Module):
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
        implementation: Literal["pytorch", "triton"] = "pytorch",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.implementation = implementation
        self.n = n
        self.d_hidden = d_hidden
        self.eps = eps

        self.ln_weight = Parameter(d_hidden, init="one")
        self.ln_bias = Parameter(d_hidden, init="zero")
        self.expand_a_weight = Parameter((n * d_hidden, d_hidden), init="relu")
        self.expand_b_weight = Parameter((n * d_hidden, d_hidden), init="relu")
        self.squeeze_weight = Parameter((d_hidden, d_hidden * n), init="final")
        if self.implementation == "pytorch":
            self.swish = SwishFunction(implementation=implementation)

    @typecheck
    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        if self.implementation == "pytorch":
            x = F.layer_norm(x, (self.d_hidden,), self.ln_weight, self.ln_bias, self.eps)
            a = F.linear(x, self.expand_a_weight)
            b = F.linear(x, self.expand_b_weight)
            x = self.swish(a, b)
            x = F.linear(x, self.squeeze_weight)
            return x
        elif self.implementation == "triton":
            if x.ndim != 4:
                raise ValueError(
                    f"Input tensor must be 4D, not {x.ndim}D. "
                    "(3D tensor like B,L,D is not supported yet for triton kernel)"
                )
            return kernels.triton_transition(
                x,
                self.ln_weight,
                self.ln_bias,
                self.expand_a_weight,
                self.expand_b_weight,
                self.squeeze_weight,
                self.n,
            )
        raise ValueError(
            f"Invalid implementation: {self.implementation}. "
            "Choose either 'pytorch' or 'triton'."
        )

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
    ):
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
            (n * d_rep, d_rep),
            init="relu",
        )
        self.expand_b_weight = Parameter((n * d_rep, d_rep), init="relu")
        self.squeeze_weight = Parameter((d_rep, d_rep * n), init="default")

        self.last_conditioning = Linear(d_cond, d_rep, bias=True, init="zero")
        # biasinit = -2.0
        self.last_conditioning.bias.data.fill_(-2.0)
        self.swish = SwishFunction(implementation=implementation)
        self.sigmoid_gate = SigmoidGateFunction(implementation=implementation)
        # self.swish = SwishFunction(implementation="pytorch")
        # self.sigmoid_gate = SigmoidGateFunction(implementation="pytorch")

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
        atom_single_rep = self.sigmoid_gate(last_conditioning, atom_single_rep)

        return atom_single_rep

    def forward(
        self,
        atom_single_rep: Float[torch.Tensor, "* d_rep"],
        atom_single_cond: Float[torch.Tensor, "* d_cond"] | None = None,
    ) -> Float[torch.Tensor, "* d_rep"]:
        assert atom_single_cond is not None, "atom_single_cond must be provided"

        if self.use_checkpoint:
            atom_single_rep = torch.utils.checkpoint.checkpoint(
                self._forward, atom_single_rep, atom_single_cond, use_reentrant=False
            )
        else:
            atom_single_rep = self._forward(atom_single_rep, atom_single_cond)
        return atom_single_rep
