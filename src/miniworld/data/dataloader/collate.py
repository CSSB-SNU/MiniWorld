"""Bucketed collate for torch.compile cache efficiency."""

from __future__ import annotations

from miniworld.data.features import Batch


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

    Collates the batch normally, then pads dimensions to bucket boundaries by
    collating with a dummy empty batch of the bucketed size and discarding it.
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

    dummy = Batch.empty(
        n_temp=bucketed_template,
        msa_depth=bucketed_msa,
        n_tokens=bucketed_tokens,
        n_atoms=bucketed_atoms,
    )
    padded = Batch.collate_fn([batch, dummy])
    return padded[0 : batch.batch_size]
