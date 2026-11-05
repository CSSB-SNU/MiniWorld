"""The architecture blocks' primitives. ``Linear``, ``MPLinear``, ``Dropout``, ``InitType``, the initialisers and ``LayerNorm`` are
``modules.primitives``' (one implementation); ``Linear`` / ``MPLinear`` are subclasses only so ``convert_linears_to_mp`` can tell
the Linears architecture code builds (converted) from the engine modules' (left alone), as it always has. ``LayerNorm`` is the
engine's, built from the architecture ``ImplementationType`` (a constructor adapter only)."""

import torch
import torch.nn as nn
from foldforge.layers.implementation import ImplementationType, to_engine_impl
from miniworld_engine.modules import primitives as _modules_primitives
from miniworld_engine.modules.exceptions import (
    ImplementationType as EngineImplementation,
)
from miniworld_engine.modules.primitives import (  # noqa: F401 -- re-exported
    Dropout,
    InitType,
    _Fp32ParamsMixin,
    _trunc_normal_init,
    magnitude_normalize,
)
from torch import Size
class Linear(_modules_primitives.Linear):
    """``modules.primitives.Linear``, as built by architecture code (see the module docstring)."""


class MPLinear(_modules_primitives.MPLinear, Linear):
    """``modules.primitives.MPLinear``, as built by architecture code (see the module docstring)."""


def convert_linears_to_mp(
    root: nn.Module,
    exclude_substrings: tuple[str, ...] = (),
) -> int:
    """Recursively replace ``Linear`` children with ``MPLinear`` (full-MP).

    For every ``Linear`` whose qualified name does NOT contain any string in
    ``exclude_substrings`` (e.g. ``"final_denoising"``), swap it for an
    ``MPLinear`` of the same shape (unit-Gaussian re-init; EDM2 on-use
    normalization makes the init magnitude irrelevant). ``init="zero"`` /
    ``"gating"`` Linears are turned into MP layers too (they cannot stay
    zero-init under MP). Bias is preserved if present.

    Any module exposing ``self.implementation`` (e.g. ``Transition``) is forced
    onto the PyTorch path, because its miniworld_engine kernel consumes raw ``.weight`` and
    would bypass MPLinear's on-use normalization. Returns the number of layers
    converted. Intended for the diffusion module only (never the frozen trunk).
    """
    converted = 0
    for name, child in list(root.named_modules()):
        if hasattr(child, "implementation"):
            child.implementation = ImplementationType.PYTORCH
    for name, child in list(root.named_modules()):
        if not isinstance(child, Linear) or isinstance(child, MPLinear):
            continue
        if any(s in name for s in exclude_substrings):
            continue
        parent = root
        *path, attr = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        new = MPLinear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            dtype=child.weight.dtype,
            init="normal",
        )
        setattr(parent, attr, new)
        converted += 1
    return converted




_shape_t = int | list[int] | Size


def _layernorm_impl(implementation: ImplementationType) -> EngineImplementation:
    """The engine backend for an architecture ``ImplementationType``: cuEquivariance supplies no standalone LayerNorm kernel, so
    its model backend runs the PyTorch reference (as before); ``miniworld_engine`` is the engine's ``miniworld`` (auto-routing)."""
    implementation = ImplementationType(implementation)
    if implementation == ImplementationType.CUEQUIVARIANCE:
        return EngineImplementation.PYTORCH
    return to_engine_impl(implementation)


class LayerNorm(_modules_primitives.LayerNorm):
    """The engine's ``modules.primitives.LayerNorm`` (fp32-pinned affine, the activation dtype preserved, the engine's own kernel
    routing), built from the architecture ``ImplementationType``. Same parameters (``weight`` / ``bias``)."""

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        *,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__(
            normalized_shape,
            eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
            implementation=_layernorm_impl(implementation),
        )
