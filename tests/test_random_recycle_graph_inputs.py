"""CUDA graph inputs must update masks/indices and reject incompatible batches."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "random_graph", Path(__file__).parents[1] / "scripts/random_recycle_graph_trainer.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def batch(value):
    return SimpleNamespace(
        **{
            name: SimpleNamespace(
                data=torch.full((2, 3), value),
                mask=torch.tensor([True, False]),
                optional=None,
            )
            for name in m.CONSUMED
        }
    )


def test_copy_updates_all_inputs_without_replacing_storage():
    dst, src = batch(0), batch(3)
    pointers = {name: getattr(dst, name).data.data_ptr() for name in m.CONSUMED}
    src.msa.mask.logical_not_()
    m.copy_static(dst, src)
    for name in m.CONSUMED:
        assert getattr(dst, name).data.data_ptr() == pointers[name]
        torch.testing.assert_close(getattr(dst, name).data, getattr(src, name).data)
    torch.testing.assert_close(dst.msa.mask, src.msa.mask)


@pytest.mark.parametrize("change", ["shape", "dtype", "optional"])
def test_incompatible_input_is_rejected(change):
    dst, src = batch(0), batch(1)
    if change == "shape":
        src.msa.data = torch.ones(3, 3, dtype=torch.long)
    if change == "dtype":
        src.msa.data = src.msa.data.float()
    if change == "optional":
        src.msa.optional = torch.ones(1)
    with pytest.raises(ValueError, match="Graph"):
        m.copy_static(dst, src)


def test_sparse_bond_metadata_only_skipped_with_dense_bond_input():
    dst, src = batch(0), batch(1)
    dst.structure.token_bond = torch.zeros((1, 3, 2), dtype=torch.long)
    src.structure.token_bond = torch.zeros((1, 5, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="dense"):
        m.copy_static(dst, src)
    dst.structure.token_bond_feat = torch.zeros((1, 4, 4), dtype=torch.bool)
    src.structure.token_bond_feat = torch.ones((1, 4, 4), dtype=torch.bool)
    m.copy_static(dst, src)
    torch.testing.assert_close(
        dst.structure.token_bond_feat, src.structure.token_bond_feat
    )


def test_empty_template_updates_presence_and_clears_static_slots():
    dst, src = batch(0), batch(1)
    dst.template = SimpleNamespace(
        mask=torch.ones(1, 4, dtype=torch.bool),
        data=torch.ones(1, 4, 3),
        _graph_present=torch.tensor(True),
    )
    src.template = SimpleNamespace(
        mask=torch.empty(1, 0, dtype=torch.bool), data=torch.empty(1, 0, 3)
    )
    m.copy_static(dst, src)
    assert not dst.template._graph_present
    assert not dst.template.mask.any()
    assert not dst.template.data.any()
    src.template = SimpleNamespace(
        mask=torch.ones(1, 4, dtype=torch.bool), data=torch.full((1, 4, 3), 2.0)
    )
    m.copy_static(dst, src)
    assert dst.template._graph_present
    torch.testing.assert_close(dst.template.data, src.template.data)
