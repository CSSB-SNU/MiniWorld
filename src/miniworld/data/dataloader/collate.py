"""Bucketed collate for torch.compile cache efficiency."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import torch

from miniworld.data.features import Batch


def _pad_like(value, prototype):
    """Pad directly to an empty batch's schema, without materializing its data."""
    if isinstance(value, torch.Tensor):
        if not isinstance(prototype, torch.Tensor) or value.ndim != prototype.ndim:
            raise ValueError("Incompatible tensor schema while padding a batch")
        shape = (value.shape[0], *(max(a, b) for a, b in zip(value.shape[1:], prototype.shape[1:], strict=True)))
        # The former dummy/cat path also promoted dtypes against Batch.empty.
        dtype = torch.promote_types(value.dtype, prototype.dtype)
        if tuple(value.shape) == shape and value.dtype == dtype:
            return value
        output = value.new_zeros(shape, dtype=dtype)
        output[tuple(slice(0, size) for size in value.shape)].copy_(value)
        return output
    if is_dataclass(value):
        return replace(value, **{
            field.name: _pad_like(getattr(value, field.name), getattr(prototype, field.name))
            for field in fields(value)
        })
    if isinstance(value, list):
        return value[:]
    if value is None and prototype is None:
        return None
    raise ValueError("Incompatible field schema while padding a batch")


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def bucketed_collate(
    batch_list: list[Batch],
    bucket_msa_multiple: int | None = None,
    bucket_template_multiple: int | None = 1,
    bucket_token_multiple: int | None = None,
    bucket_atom_multiple: int | None = None,
) -> Batch:
    """Collate a list of Batches with shape bucketing.

    Collates the batch normally, then pads directly to the bucket boundaries.
    An empty meta-device batch provides shapes/dtypes without allocating dummy
    data or retaining a discarded extra batch in the returned tensor storage.
    """
    batch = Batch.collate_fn(batch_list)

    if bucket_token_multiple is None and bucket_atom_multiple is None:
        return batch

    n_temp = batch.template_number
    msa_depth = batch.msa_depth
    n_tokens = batch.token_length
    n_atoms = batch.atom_length

    bucketed_msa = (
        _ceil_to_multiple(msa_depth, bucket_msa_multiple)
        if bucket_msa_multiple
        else msa_depth
    )
    bucketed_template = (
        _ceil_to_multiple(n_temp, bucket_template_multiple)
        if bucket_template_multiple
        else n_temp
    )
    bucketed_tokens = (
        _ceil_to_multiple(n_tokens, bucket_token_multiple)
        if bucket_token_multiple
        else n_tokens
    )
    bucketed_atoms = (
        _ceil_to_multiple(n_atoms, bucket_atom_multiple)
        if bucket_atom_multiple
        else n_atoms
    )

    if (
        bucket_msa_multiple
        and bucketed_msa == msa_depth
        and bucketed_template == n_temp
        and bucketed_tokens == n_tokens
        and bucketed_atoms == n_atoms
    ):
        return batch

    with torch.device("meta"):
        prototype = Batch.empty(
            n_temp=bucketed_template,
            msa_depth=bucketed_msa,
            n_tokens=bucketed_tokens,
            n_atoms=bucketed_atoms,
        )
    return _pad_like(batch, prototype)
