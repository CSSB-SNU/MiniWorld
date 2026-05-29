"""Equivalence test for InputAtomAttentionEncoder's chunked inference path.

``InputAtomAttentionEncoder._before_atom_transformer_chunked`` mirrors the
``AtomAttentionEncoder`` (diffusion_module) chunked path: it streams the
broadcast add and pair-MLP over L_atom row chunks so the
[B, L_atom, L_atom, d] temporaries never materialise in one shot. It MUST
produce numerically identical outputs to the canonical
``_before_atom_transformer`` — drift would mean inference is feeding the
model a different conditioning than the canonical path.

Run:
    pytest libs/MiniWorld/tests/test_input_embedder_chunked.py -v
or:
    python libs/MiniWorld/tests/test_input_embedder_chunked.py
"""

from __future__ import annotations

import torch
from team_gm.modules import DiffusionTransformer

from miniworld.configs import SharedConfig
from miniworld.modules.input_embedder import InputAtomAttentionEncoder


def _make_encoder(
    d_single: int = 16,
    d_single_atom: int = 8,
    d_pair: int = 8,
    d_pair_atom: int = 4,
) -> InputAtomAttentionEncoder:
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
    encoder = InputAtomAttentionEncoder(shared_cfg, diffusion_cfg)
    encoder.eval()
    # mlp_atom_pair's last linear is zero-init by design; force non-zero
    # weights so equivalence is actually exercised (otherwise the MLP
    # contribution collapses to 0 in both paths and bugs hide).
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in encoder.parameters():
            if p.numel() == 0:
                continue
            p.normal_(mean=0.0, std=0.3, generator=g)
    return encoder


def _make_inputs(
    *,
    B: int = 1,
    L_atom: int = 11,
    d_single_atom_init: int = 6,
    d_pair_atom_init: int = 5,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "atom_single_init": torch.randn(B, L_atom, d_single_atom_init, generator=g),
        "atom_pair_init": torch.randn(
            B, L_atom, L_atom, d_pair_atom_init, generator=g,
        ),
    }


@torch.no_grad()
def _run_canonical(encoder, inputs):
    return encoder._before_atom_transformer(**inputs)


@torch.no_grad()
def _run_chunked(encoder, inputs, chunk):
    saved = InputAtomAttentionEncoder._ATOM_CHUNK
    InputAtomAttentionEncoder._ATOM_CHUNK = chunk
    try:
        return encoder._before_atom_transformer_chunked(**inputs)
    finally:
        InputAtomAttentionEncoder._ATOM_CHUNK = saved


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
    """chunk < L_atom: chunked iterates multiple times."""
    encoder = _make_encoder()
    inputs = _make_inputs(L_atom=11)
    out_a = _run_canonical(encoder, inputs)
    out_b = _run_chunked(encoder, inputs, chunk=3)
    _assert_match(out_a[0], out_b[0], "atom_single_rep")
    _assert_match(out_a[1], out_b[1], "atom_single_cond")
    _assert_match(out_a[2], out_b[2], "atom_pair")


def test_chunked_matches_canonical_chunk_one() -> None:
    """chunk=1: worst-case tiling — every row is its own chunk."""
    encoder = _make_encoder()
    inputs = _make_inputs(L_atom=11)
    out_a = _run_canonical(encoder, inputs)
    out_b = _run_chunked(encoder, inputs, chunk=1)
    _assert_match(out_a[0], out_b[0], "atom_single_rep")
    _assert_match(out_a[1], out_b[1], "atom_single_cond")
    _assert_match(out_a[2], out_b[2], "atom_pair")


def test_chunked_matches_canonical_under_inference_mode() -> None:
    """Run both paths inside torch.inference_mode like production does."""
    encoder = _make_encoder()
    inputs = _make_inputs(L_atom=11)
    with torch.inference_mode():
        out_a = encoder._before_atom_transformer(**inputs)
        saved = InputAtomAttentionEncoder._ATOM_CHUNK
        InputAtomAttentionEncoder._ATOM_CHUNK = 3
        try:
            out_b = encoder._before_atom_transformer_chunked(**inputs)
        finally:
            InputAtomAttentionEncoder._ATOM_CHUNK = saved
    _assert_match(out_a[0], out_b[0], "atom_single_rep")
    _assert_match(out_a[1], out_b[1], "atom_single_cond")
    _assert_match(out_a[2], out_b[2], "atom_pair")


def test_chunked_matches_canonical_large() -> None:
    """Larger L_atom — exercises accumulation order."""
    encoder = _make_encoder()
    inputs = _make_inputs(L_atom=64)
    out_a = _run_canonical(encoder, inputs)
    out_b = _run_chunked(encoder, inputs, chunk=7)
    _assert_match(out_a[0], out_b[0], "atom_single_rep")
    _assert_match(out_a[1], out_b[1], "atom_single_cond")
    _assert_match(out_a[2], out_b[2], "atom_pair")


if __name__ == "__main__":
    for fn in (
        test_chunked_matches_canonical_no_split,
        test_chunked_matches_canonical_with_split,
        test_chunked_matches_canonical_chunk_one,
        test_chunked_matches_canonical_under_inference_mode,
        test_chunked_matches_canonical_large,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
