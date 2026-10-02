"""The atom-to-token mean as a scatter-add, against the one-hot einsums it replaced (CPU only)."""
from types import SimpleNamespace

import pytest
import torch

from miniworld.modules.input_feature_embedder_esmfold2_style import ESMFold2InputAtomAttentionEncoder


def _one_hot_mean(token_length, atom_mask, atom_to_token_idx_map, to_add):
    """The pre-change formulation: a [B, L_atom, L_token] one-hot mapping and two einsums."""
    mapping = torch.nn.functional.one_hot(atom_to_token_idx_map, num_classes=token_length).to(to_add.dtype)
    token_sum = torch.einsum("bat,bad->btd", mapping, to_add)
    count = torch.einsum("bat,ba->bt", mapping, atom_mask.to(mapping.dtype))
    return token_sum / count.unsqueeze(-1).clamp(min=1.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_scatter_add_mean_equals_the_one_hot_mean(dtype):
    torch.manual_seed(0)
    b, atoms, tokens, d_atom, d_token = 2, 300, 40, 16, 12
    stub = SimpleNamespace(atom_single_rep_to_token_single=torch.nn.Linear(d_atom, d_token).to(dtype))
    token_idx = torch.zeros(b, tokens, dtype=torch.long)
    atom_to_token = torch.randint(0, tokens - 5, (b, atoms))          # the last five tokens own no atom
    atom_mask = torch.rand(b, atoms) > 0.15                           # padding atoms
    rep = torch.randn(b, atoms, d_atom).to(dtype)
    got = ESMFold2InputAtomAttentionEncoder._scatter_atom_to_token(stub, token_idx, atom_mask, atom_to_token, rep)
    to_add = stub.atom_single_rep_to_token_single(rep * atom_mask[..., None])
    want = _one_hot_mean(tokens, atom_mask, atom_to_token, to_add.float()).to(dtype)
    assert got.dtype == dtype
    assert torch.allclose(got.float(), want.float(), atol=1e-5 if dtype == torch.float32 else 2e-2, rtol=1e-4 if dtype == torch.float32 else 2e-2)
    assert torch.equal(got[:, -5:], torch.zeros_like(got[:, -5:]))      # a token without atoms stays zero (the count is clamped)


def test_the_scatter_add_mean_is_differentiable_like_the_einsum():
    torch.manual_seed(1)
    stub = SimpleNamespace(atom_single_rep_to_token_single=torch.nn.Linear(8, 6))
    token_idx = torch.zeros(1, 10, dtype=torch.long)
    atom_to_token = torch.randint(0, 10, (1, 64))
    atom_mask = torch.ones(1, 64, dtype=torch.bool)
    rep = torch.randn(1, 64, 8, requires_grad=True)
    got = ESMFold2InputAtomAttentionEncoder._scatter_atom_to_token(stub, token_idx, atom_mask, atom_to_token, rep)
    grad = torch.randn_like(got)
    (g_new,) = torch.autograd.grad(got, rep, grad)
    want = _one_hot_mean(10, atom_mask, atom_to_token, stub.atom_single_rep_to_token_single(rep))
    (g_old,) = torch.autograd.grad(want, rep, grad)
    assert torch.allclose(g_new, g_old, atol=1e-5)
