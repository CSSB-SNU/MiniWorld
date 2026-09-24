"""Padding must retain legacy values/dtypes; spawned catalogs must retain sharing."""
import pickle
from dataclasses import fields, is_dataclass

import pyarrow as pa
import pytest
import torch

from miniworld.data.dataloader.collate import bucketed_collate
from miniworld.data.dataloader.dataloader import _LazyArrowCatalog
from miniworld.data.features import Batch


def legacy_collate(batches, msa=16, token=8, atom=16, template=4):
    b = Batch.collate_fn(batches)
    up = lambda n, k: ((n + k - 1) // k) * k
    dummy = Batch.empty(n_temp=up(b.template_number, template),
                        msa_depth=up(b.msa_depth, msa),
                        n_tokens=up(b.token_length, token),
                        n_atoms=up(b.atom_length, atom))
    return Batch.collate_fn([b, dummy])[:b.batch_size]


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    elif is_dataclass(a):
        for field in fields(a):
            assert_equal(getattr(a, field.name), getattr(b, field.name))
    else:
        assert a == b


def populated(n, t):
    b = Batch.empty(n_temp=t, msa_depth=n + 2, n_tokens=n, n_atoms=n * 2)
    for group in (b.sequence, b.structure, b.reference, b.scheme, b.msa, b.template, b.chain):
        for field in fields(group):
            value = getattr(group, field.name)
            if isinstance(value, torch.Tensor):
                value.copy_((torch.arange(value.numel()).reshape(value.shape) % 5).to(value.dtype))
    b.msa.aligned_sequences = b.msa.aligned_sequences.to(torch.int8)
    b.reference.element = b.reference.element.to(torch.int64)
    b.structure.token_bond = torch.tensor([[[0, 1], [1, 2]]])
    b.name = [f'sample-{n}']
    b.atom_ids = [[str(i) for i in range(n * 2)]]
    return b


@pytest.mark.parametrize('templates', [0, 1, 4])
@pytest.mark.parametrize('batch_size', [1, 2])
def test_direct_padding_matches_dummy_path(templates, batch_size):
    batches = [populated(5 + i, templates) for i in range(batch_size)]
    expected = legacy_collate(batches)
    actual = bucketed_collate(batches, bucket_msa_multiple=16,
                              bucket_token_multiple=8, bucket_atom_multiple=16,
                              bucket_template_multiple=4)
    assert_equal(actual, expected)
    tensor = actual.msa.aligned_sequences
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    # In particular, keep legacy promotion to long/float against Batch.empty.
    assert actual.msa.aligned_sequences.dtype == torch.long
    assert actual.reference.element.dtype == torch.float32


def write_table(path):
    table = pa.table({'payload': ['x' * 128] * 16384})
    with pa.OSFile(str(path), 'wb') as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    source = pa.memory_map(str(path), 'r')
    return pa.ipc.open_file(source).read_all()


def test_catalog_pickle_reopens_file_instead_of_copying_buffers(tmp_path):
    path = tmp_path / 'catalog.arrow'
    table = write_table(path)
    catalog = _LazyArrowCatalog(table, path=path)
    payload = pickle.dumps(catalog)
    assert len(payload) < 2048
    restored = pickle.loads(payload)
    assert restored._t.equals(table)
    assert restored._path == path


def test_catalog_replacement_is_rejected(tmp_path):
    path = tmp_path / 'catalog.arrow'
    catalog = _LazyArrowCatalog(write_table(path), path=path)
    payload = pickle.dumps(catalog)
    replacement = tmp_path / 'new.arrow'
    write_table(replacement)
    replacement.replace(path)
    with pytest.raises(RuntimeError, match='Catalog changed'):
        pickle.loads(payload)


def test_in_memory_catalog_still_pickles():
    table = pa.table({'payload': ['small']})
    restored = pickle.loads(pickle.dumps(_LazyArrowCatalog(table)))
    assert restored._t.equals(table)
