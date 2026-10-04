"""miniworld.training.precision: bf16-mixed makes fp32 master parameters and an autocast forward; native leaves the model."""

import pytest
import torch
from torch import nn

from miniworld.training.precision import apply_precision, is_bf16_mixed, model_autocast


def _model() -> nn.Module:
    m = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    m[0].to(torch.bfloat16)                          # a bf16-pinned part, as the trunk builds it
    m.register_buffer("table", torch.zeros(4, dtype=torch.bfloat16))
    return m


def test_bf16_mixed_makes_fp32_masters_and_autocasts():
    m = apply_precision(_model(), "bf16-mixed")
    assert all(p.dtype == torch.float32 for p in m.parameters())
    assert m.table.dtype == torch.float32
    assert is_bf16_mixed(m)
    ctx = model_autocast(m)
    assert isinstance(ctx, torch.autocast) and ctx.fast_dtype == torch.bfloat16


def test_native_keeps_dtypes_and_no_autocast():
    m = apply_precision(_model(), "native")
    assert m[0].weight.dtype == torch.bfloat16 and m[1].weight.dtype == torch.float32
    assert not is_bf16_mixed(m)
    assert not isinstance(model_autocast(m), torch.autocast)


def test_wrappers_reach_the_flag():
    m = apply_precision(_model(), "bf16-mixed")

    class Wrapper(nn.Module):                        # Fabric / DDP keep the model as .module
        def __init__(self, module):
            super().__init__()
            self.module = module

    assert is_bf16_mixed(Wrapper(m))


def test_unknown_precision_raises():
    with pytest.raises(ValueError, match="Unknown training precision"):
        apply_precision(_model(), "fp16-mixed")
