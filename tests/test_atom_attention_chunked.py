"""Equivalence test for AtomAttentionEncoder's chunked inference path.

``_before_atom_transformer_chunked`` is a memory-saving variant of
``_before_atom_transformer`` meant for large-L_atom inference. It MUST
produce numerically identical outputs to the canonical path — anything
else means inference is feeding the model a different conditioning than
training did.

Run:
    pytest libs/MiniWorld/tests/test_atom_attention_chunked.py -v
or:
    python libs/MiniWorld/tests/test_atom_attention_chunked.py
"""

from __future__ import annotations

import torch
from team_gm.modules import DiffusionTransformer

from miniworld.configs import SharedConfig
from miniworld.modules.diffusion_module import AtomAttentionEncoder


def _make_encoder(
    d_single: int = 16,
    d_single_atom: int = 8,
    d_pair: int = 8,
    d_pair_atom: int = 4,
) -> AtomAttentionEncoder:
    shared_cfg = SharedConfig(
        d_single=d_single,
        d_single_atom=d_single_atom,
        d_pair=d_pair,
        d_pair_atom=d_pair_atom,
    )
    diffusion_cfg = DiffusionTransformer.Config(
        d_single=d_single_atom,
        d_cond=d_single_atom,
        d_pair=d_pair_atom,
        n_head=2,
        n_block=1,
    )
    encoder = AtomAttentionEncoder(shared_cfg, diffusion_cfg)
    encoder.eval()
    # Several internal projections (e.g. token_pair_to_atom_pair last linear,
    # mlp_atom_pair last linear) are zero-initialised on purpose so the
    # residual contributions start at 0. That'd silently mask any broadcast
    # mismatch in the chunked path because _to_add_pair / mlp output would
    # collapse to zero regardless of which gather pattern is used. Force
    # non-zero weights so the equivalence check is meaningful.
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in encoder.parameters():
            if p.numel() == 0:
                continue
            p.normal_(mean=0.0, std=0.3, generator=g)
    return encoder


def _make_inputs(
    *,
    A: int = 1,
    B: int = 1,
    L_atom: int = 11,
    L_token: int = 4,
    d_single: int = 16,
    d_pair: int = 8,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    # Per-batch token assignment, replicated across B; the canonical path
    # passes ``atom_to_token_idx_map`` twice into advanced indexing without
    # reshaping, so its layout exercises whatever PyTorch broadcast rule the
    # production model has been trained against.
    base_map = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3], dtype=torch.long,
    )[:L_atom]
    atom_to_token_idx_map = base_map.unsqueeze(0).repeat(B, 1)
    return {
        "x_t": torch.randn(A, B, L_atom, 3, generator=g),
        "x_mask": torch.ones(A, B, L_atom, dtype=torch.bool),
        "atom_single_init": torch.randn(B, L_atom, 6, generator=g),
        "atom_pair_init": torch.randn(B, L_atom, L_atom, 5, generator=g),
        "atom_to_token_idx_map": atom_to_token_idx_map,
        "token_single_cond": torch.randn(B, L_token, d_single, generator=g),
        "token_pair_cond": torch.randn(B, L_token, L_token, d_pair, generator=g),
    }


@torch.no_grad()
def _run_canonical(encoder: AtomAttentionEncoder, inputs: dict[str, torch.Tensor]):
    return encoder._before_atom_transformer(**inputs)


@torch.no_grad()
def _run_chunked(
    encoder: AtomAttentionEncoder,
    inputs: dict[str, torch.Tensor],
    chunk: int,
):
    saved = AtomAttentionEncoder._ATOM_CHUNK
    AtomAttentionEncoder._ATOM_CHUNK = chunk
    try:
        return encoder._before_atom_transformer_chunked(**inputs)
    finally:
        AtomAttentionEncoder._ATOM_CHUNK = saved


def _assert_match(a, b, name: str, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs()
        msg = (
            f"{name}: chunked vs canonical mismatch — "
            f"max_abs_diff={diff.max().item():.3e} "
            f"mean_abs_diff={diff.mean().item():.3e} "
            f"shape={tuple(a.shape)}"
        )
        raise AssertionError(msg)


def test_canonical_gather_pattern_diagnostic() -> None:
    """Pin down what the canonical ``_to_add_pair[bidx, A, A]`` gather does.

    The canonical path passes ``atom_to_token_idx_map`` (shape [B, L_atom])
    twice into a 4-D advanced index alongside ``batch_2d_idx`` (shape
    [B, L_atom, L_atom]). PyTorch broadcasts all three index tensors
    together; the two ``atom_to_token_idx_map`` references end up tracking
    the SAME broadcast position, so both pick the same token id at each
    output (b, i, j) cell.

    For B=1, that broadcast pattern picks the column atom's token id —
    ``canonical[0, i, j] = _to_add_pair[0, A[0, j], A[0, j]]`` — i.e.
    the j-token's self-pair, independent of the row atom. The chunked
    refactor used distinct ``idx_i`` / ``idx_j`` and so changed the
    gather to the full outer product ``_to_add_pair[0, A[0,i], A[0,j]]``.
    That semantic shift is what was breaking inference.
    """
    B, L_atom, L_token, d = 1, 5, 3, 2
    _to_add_pair = torch.arange(B * L_token * L_token * d, dtype=torch.float32)
    _to_add_pair = _to_add_pair.reshape(B, L_token, L_token, d)
    A = torch.tensor([[0, 0, 1, 1, 2]], dtype=torch.long)  # [B, L_atom]
    batch_2d = torch.arange(B).view(B, 1, 1).expand(-1, L_atom, L_atom)

    canonical = _to_add_pair[batch_2d, A, A]

    expected_j_indexed = torch.stack(
        [_to_add_pair[0, A[0, j], A[0, j]] for j in range(L_atom)],
        dim=0,
    )  # [L_atom, d]
    # Every row of canonical[0] should equal expected_j_indexed.
    for i in range(L_atom):
        if not torch.equal(canonical[0, i], expected_j_indexed):
            msg = (
                f"canonical[0, {i}] = {canonical[0, i].tolist()} != "
                f"expected_j_indexed = {expected_j_indexed.tolist()} — "
                f"broadcast assumption is wrong, fix below will not match."
            )
            raise AssertionError(msg)


def test_chunked_matches_canonical_no_split() -> None:
    """chunk >= L_atom: chunked degenerates to one iteration, still must match."""
    encoder = _make_encoder()
    inputs = _make_inputs(L_atom=11)
    out_a = _run_canonical(encoder, inputs)
    out_b = _run_chunked(encoder, inputs, chunk=64)
    _assert_match(out_a[0], out_b[0], "atom_single_rep")
    _assert_match(out_a[1], out_b[1], "atom_single_cond")
    _assert_match(out_a[2], out_b[2], "atom_pair")


def test_chunked_matches_canonical_with_split() -> None:
    """chunk < L_atom: chunked iterates multiple times — equivalence must hold."""
    encoder = _make_encoder()
    inputs = _make_inputs(L_atom=11)
    out_a = _run_canonical(encoder, inputs)
    out_b = _run_chunked(encoder, inputs, chunk=3)
    _assert_match(out_a[0], out_b[0], "atom_single_rep")
    _assert_match(out_a[1], out_b[1], "atom_single_cond")
    _assert_match(out_a[2], out_b[2], "atom_pair")


def test_chunked_matches_canonical_chunk_one() -> None:
    """chunk=1: every atom row is its own chunk — worst-case for tiling bugs."""
    encoder = _make_encoder()
    inputs = _make_inputs(L_atom=11)
    out_a = _run_canonical(encoder, inputs)
    out_b = _run_chunked(encoder, inputs, chunk=1)
    _assert_match(out_a[0], out_b[0], "atom_single_rep")
    _assert_match(out_a[1], out_b[1], "atom_single_cond")
    _assert_match(out_a[2], out_b[2], "atom_pair")


def test_chunked_matches_canonical_with_augmentation() -> None:
    """A>1: production runs draw multiple augmentations through this module."""
    encoder = _make_encoder()
    inputs = _make_inputs(A=4, L_atom=11)
    out_a = _run_canonical(encoder, inputs)
    out_b = _run_chunked(encoder, inputs, chunk=3)
    _assert_match(out_a[0], out_b[0], "atom_single_rep")
    _assert_match(out_a[1], out_b[1], "atom_single_cond")
    _assert_match(out_a[2], out_b[2], "atom_pair")


def test_chunked_matches_canonical_under_inference_mode() -> None:
    """Run both paths inside ``torch.inference_mode`` like production does.

    Inference tensors skip version counting and view metadata; the chunked
    path's in-place ``pair_slice.add_`` and ``atom_pair[:, s:e].add_(...)``
    are the obvious places where that could matter.
    """
    encoder = _make_encoder()
    inputs = _make_inputs(A=2, L_atom=11)
    with torch.inference_mode():
        out_a = encoder._before_atom_transformer(**inputs)
        saved = AtomAttentionEncoder._ATOM_CHUNK
        AtomAttentionEncoder._ATOM_CHUNK = 3
        try:
            out_b = encoder._before_atom_transformer_chunked(**inputs)
        finally:
            AtomAttentionEncoder._ATOM_CHUNK = saved
    _assert_match(out_a[0], out_b[0], "atom_single_rep")
    _assert_match(out_a[1], out_b[1], "atom_single_cond")
    _assert_match(out_a[2], out_b[2], "atom_pair")


def test_chunked_matches_canonical_large() -> None:
    """L_atom ~ small-target scale; exposes accumulation/order issues."""
    encoder = _make_encoder()
    inputs = _make_inputs(A=2, L_atom=64, L_token=16)
    # L_token=16 means atom_to_token_idx_map values need to fit in [0, 16) —
    # rebuild with a richer assignment than the 4-token default.
    g = torch.Generator().manual_seed(123)
    inputs["atom_to_token_idx_map"] = torch.randint(
        0, 16, (1, 64), generator=g, dtype=torch.long,
    )
    out_a = _run_canonical(encoder, inputs)
    out_b = _run_chunked(encoder, inputs, chunk=7)
    _assert_match(out_a[0], out_b[0], "atom_single_rep")
    _assert_match(out_a[1], out_b[1], "atom_single_cond")
    _assert_match(out_a[2], out_b[2], "atom_pair")


if __name__ == "__main__":
    for fn in (
        test_canonical_gather_pattern_diagnostic,
        test_chunked_matches_canonical_no_split,
        test_chunked_matches_canonical_with_split,
        test_chunked_matches_canonical_chunk_one,
        test_chunked_matches_canonical_with_augmentation,
        test_chunked_matches_canonical_under_inference_mode,
        test_chunked_matches_canonical_large,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
