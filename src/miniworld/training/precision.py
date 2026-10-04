"""Training precision: which dtype the parameters (the optimizer's master weights) live in and how the model computes.

``bf16-mixed`` (the default; what PyTorch AMP and Lightning's ``bf16-mixed`` do): every parameter is fp32 -- the master the
optimizer updates, with fp32 Adam moments and an fp32 EMA -- and the model's forward runs under bf16 autocast. The models still
cast their activations to bf16 where they always did, so the engine's B200 kernels serve the same calls: they cast the fp32
parameters to bf16 for themselves, outside autograd, and hand back fp32 weight gradients (unrounded; a plain Linear under
autocast rounds its weight gradient to bf16 values, as AMP does). Losses run in fp32, outside autocast.

``native``: the parameters keep the dtypes the models build them in (bf16 trunk, the diffusion module per ``diffusion.dtype``,
fp32 norms) and nothing autocasts -- the setting before fp32 masters, kept to resume and compare those runs.
"""

from __future__ import annotations

import contextlib
from typing import Literal

import torch
from torch import nn

Precision = Literal["bf16-mixed", "native"]

_AMP_FLAG = "_miniworld_bf16_mixed"


def apply_precision(model: nn.Module, precision: Precision) -> nn.Module:
    """Put ``model`` in ``precision`` before the optimizer and Fabric see it: ``bf16-mixed`` makes every floating parameter
    (and buffer) fp32 and marks the model so :func:`model_autocast` runs its forward under bf16 autocast."""
    if precision == "bf16-mixed":
        model.float()
        setattr(model, _AMP_FLAG, True)
    elif precision != "native":
        message = f"Unknown training precision: {precision}"
        raise ValueError(message)
    return model


def is_bf16_mixed(model: nn.Module) -> bool:
    """Whether :func:`apply_precision` put ``model`` in ``bf16-mixed``. ``model`` may be the Fabric / DDP / compile wrapper."""
    return any(m is not None and getattr(m, _AMP_FLAG, False)
               for m in (model, getattr(model, "module", None), getattr(model, "_orig_mod", None)))


def model_autocast(model: nn.Module) -> contextlib.AbstractContextManager:
    """The context ``model``'s forward runs in: bf16 autocast for a ``bf16-mixed`` model, nothing otherwise."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if is_bf16_mixed(model) else contextlib.nullcontext()
