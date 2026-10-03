"""Shared test setup."""
import pytest
import torch


@pytest.fixture(autouse=True)
def _full_fp32_matmul():
    """Start every test at torch's default float32 matmul precision.

    The training scripts set ``torch.set_float32_matmul_precision("medium")`` at import, and several tests import
    them, so without this every later test ran its fp32 comparisons under TF32 (torch 2.13 honours it on CPU too).
    A test that wants the trainer's precision sets it itself.
    """
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)
