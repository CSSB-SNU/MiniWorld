"""Bucketed collate for torch.compile cache efficiency."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace

import torch

from miniworld.data.features import Batch


def _pad_like(value, prototype):
    """Pad directly to an empty batch's schema, without materializing its data."""
    if isinstance(value, torch.Tensor):
        if not isinstance(prototype, torch.Tensor) or value.ndim != prototype.ndim:
            raise ValueError("Incompatible tensor schema while padding a batch")
        shape = (
            value.shape[0],
            *(
                max(a, b)
                for a, b in zip(value.shape[1:], prototype.shape[1:], strict=True)
            ),
        )
        # The former dummy/cat path also promoted dtypes against Batch.empty.
        dtype = torch.promote_types(value.dtype, prototype.dtype)
        if tuple(value.shape) == shape and value.dtype == dtype:
            return value
        output = value.new_zeros(shape, dtype=dtype)
        output[tuple(slice(0, size) for size in value.shape)].copy_(value)
        return output
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _pad_like(
                    getattr(value, field.name), getattr(prototype, field.name)
                )
                for field in fields(value)
            },
        )
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
    compact_msa: bool = False,
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
        if compact_msa:  # the prototype only gives target shapes; give it the compact MSA dtypes so padding does not widen them again
            for name, dt in (("aligned_sequences", torch.int8), ("has_deletion", torch.int8), ("deletion_value", torch.float32)):
                setattr(prototype.msa, name, getattr(prototype.msa, name).to(dt))
    return _pad_like(batch, prototype)


# ---------------------------------------------------------------------------
# Packed hand-off: one contiguous byte buffer per batch instead of one shared-memory storage per tensor.
#
# A DataLoader worker hands a batch to the main process by moving every tensor to shared memory; with the default file_descriptor
# strategy the receiver then opens one connection to the worker per storage (a Batch has ~60 tensors), all in the single
# pin-memory thread of the rank. Packing the batch into ONE uint8 buffer makes that one storage per sample; the receiver rebuilds the
# tensors as zero-copy views (same dtype / shape / values), so a batch is bit-identical to the unpacked one.
# ---------------------------------------------------------------------------
_ALIGN = 64


class _Slot(int):
    """Index of a tensor in PackedBatch.spec (a plain int subclass, so the skeleton pickles without shared-memory storages)."""


def _walk(obj, path, fn, leaf=torch.Tensor):
    """Shallow-copy ``obj`` replacing every tensor leaf by ``fn(path, tensor)`` (setattr, so no dataclass re-validation)."""
    if isinstance(obj, leaf):
        return fn(path, obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        new = copy.copy(obj)
        for f in fields(obj):
            object.__setattr__(new, f.name, _walk(getattr(obj, f.name), (*path, f.name), fn, leaf))
        return new
    if isinstance(obj, list):
        return [_walk(v, (*path, i), fn, leaf) for i, v in enumerate(obj)]
    return obj


class PackedBatch:
    """A Batch whose tensors live in one uint8 buffer (see above). ``unpack()`` gives the Batch back (views into ``buf``)."""

    def __init__(self, buf: torch.Tensor, spec: list, skeleton) -> None:
        self.buf = buf
        self.spec = spec  # [(dtype, shape, byte offset)] in tensor visit order
        self.skeleton = skeleton  # the Batch with every tensor replaced by its index in ``spec``

    def pin_memory(self, device=None):  # noqa: ARG002  (called by the DataLoader's pin-memory thread)
        return PackedBatch(self.buf.pin_memory(), self.spec, self.skeleton)

    def unpack(self):
        views = []
        for dtype, shape, off in self.spec:
            n = 1
            for s in shape:
                n *= s
            nbytes = n * torch.empty(0, dtype=dtype).element_size()
            views.append(self.buf[off : off + nbytes].view(dtype).view(shape))
        return _walk(self.skeleton, (), lambda _p, idx: views[int(idx)], _Slot)


def pack_batch(batch) -> PackedBatch:
    """Pack all tensors of ``batch`` into one buffer; ``PackedBatch.unpack()`` returns tensors equal bit for bit."""
    tensors: list[torch.Tensor] = []

    def take(_path, t):
        tensors.append(t.detach().contiguous())
        return _Slot(len(tensors) - 1)

    skeleton = _walk(batch, (), take)
    spec = []
    off = 0
    for t in tensors:
        off = (off + _ALIGN - 1) // _ALIGN * _ALIGN
        spec.append((t.dtype, tuple(t.shape), off))
        off += t.numel() * t.element_size()
    buf = torch.empty(max(off, 1), dtype=torch.uint8)
    for t, (_, _, o) in zip(tensors, spec, strict=True):
        n = t.numel() * t.element_size()
        if n:
            buf[o : o + n].copy_(t.reshape(-1).view(torch.uint8))
    return PackedBatch(buf, spec, skeleton)


def bucketed_collate_packed(batch_list, **kwargs) -> PackedBatch:
    """``bucketed_collate`` + ``pack_batch`` (a collate_fn for DataLoader workers)."""
    return pack_batch(bucketed_collate(batch_list, **kwargs))
