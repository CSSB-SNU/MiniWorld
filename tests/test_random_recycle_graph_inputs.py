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


def test_workspace_configured_before_cuda_init(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    m.configure_graph_cublas()
    assert m.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_workspace_missing_after_cuda_init_is_rejected(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before initializing CUDA"):
        m.configure_graph_cublas()


@pytest.mark.parametrize("setting", [":4096:8", ":16:8"])
def test_explicit_reproducible_workspace_is_preserved(monkeypatch, setting):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", setting)
    m.configure_graph_cublas()
    assert m.os.environ["CUBLAS_WORKSPACE_CONFIG"] == setting


def test_mixed_workspace_configuration_is_rejected(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8:4096:2")
    with pytest.raises(ValueError, match="reproducible"):
        m.configure_graph_cublas()


def diffusion_config(recycles=1, cb=True):
    return SimpleNamespace(
        model=SimpleNamespace(trunk=SimpleNamespace(diffusion=object(), n_recycle_max=recycles)),
        loss=SimpleNamespace(distogram_loss=0.25, distogram_cb_target=cb,
                             distogram_interchain_weight=2.0),
    )


def test_diffusion_graph_uses_model_loss_and_passes_interface_weight(monkeypatch):
    monkeypatch.delenv("MW_GRAPH_AMP", raising=False)
    graph = object.__new__(m.RecycleGraphs)
    graph.batch = batch(0)
    graph.loss_fn, graph.forward_kwargs = m.graph_objective(diffusion_config())
    graph.ga = 4
    weight = torch.tensor(3.0, requires_grad=True)

    def forward(**kwargs):
        assert kwargs["interchain_weight"] == 2.0
        return weight.square(), {"sigma_mean": torch.tensor(7.0)}

    graph.model = forward
    loss = graph._compute_on_stream()
    torch.testing.assert_close(loss, torch.tensor(2.25))
    torch.testing.assert_close(weight.grad, torch.tensor(0.375))
    assert graph.last_metrics["sigma_mean"] == 7


@pytest.mark.parametrize("recycles,cb", [(4, True), (1, False)])
def test_invalid_diffusion_graph_policy_rejected(recycles, cb):
    with pytest.raises(ValueError, match="one trunk pass"):
        m.graph_objective(diffusion_config(recycles, cb))
