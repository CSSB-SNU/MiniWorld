import os
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from team_gm import typecheck
from team_gm.modules import DiffusionTransformer, SWAAtomTransformer
from team_gm.modules.blocks.rope_swa_af3_transformer import RoPESWAAF3Transformer
from miniworld_engine.modules import Transition
from miniworld_engine.modules.swa_atom_attention import (
    build_attention_params,
    build_local_structure_neighbor_indices,
)
from team_gm.modules.primitives import (
    LayerNorm,
    Linear,
)
from torch import nn
from torch.utils.checkpoint import checkpoint

from miniworld.configs import SharedConfig
from miniworld.data.features import (
    ReferenceFeatures,
    SchemeFeatures,
    StructureFeatures,
)
from team_gm.modules.layers import RelativePositionEmbedding, fourier_embedding


def _make_atom_transformer(
    resolved: "SWAAtomTransformer.Config",
) -> nn.Module:
    """Pick the SWA atom transformer flavor by ``block_style``."""
    if getattr(resolved, "block_style", "esmfold2") == "af3":
        return RoPESWAAF3Transformer(resolved)
    return SWAAtomTransformer(resolved)


# ---------------------------------------------------------------------------
# Atom-pair instrumentation — opt-in via ``MINIWORLD_DUMP_ATOM_PAIR=<dir>``.
#
# When the env var points at a directory, every call to
# ``_before_atom_transformer{,_chunked}`` writes one NPZ snapshot of the
# atom_pair build-up: the per-stage cumulative state, reduced to a
# (L_token, L_token) L2-norm-then-mean-pool image. Default off → zero
# runtime cost. Visualised by ``casp17/scripts/plot_atom_pair_components.py``.
# ---------------------------------------------------------------------------
_DUMP_CALL_COUNTER: dict[str, int] = {}


def _dump_dir() -> Path | None:
    """Return the dump dir when this call should be captured, else None.

    Honours ``MINIWORLD_DUMP_ATOM_PAIR_MAX`` (default 1) so a single run
    typically writes one NPZ per per code path — the diffusion sampler
    can call the module 50+ times per inference and we usually only want
    the first call's snapshot.
    """
    raw = os.environ.get("MINIWORLD_DUMP_ATOM_PAIR")
    if not raw:
        return None
    out = Path(raw)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _should_dump(tag: str) -> bool:
    if _dump_dir() is None:
        return False
    cap_raw = os.environ.get("MINIWORLD_DUMP_ATOM_PAIR_MAX", "1")
    try:
        cap = int(cap_raw)
    except ValueError:
        cap = 1
    return _DUMP_CALL_COUNTER.get(tag, 0) < cap


def _next_call_idx(tag: str) -> int:
    idx = _DUMP_CALL_COUNTER.get(tag, 0)
    _DUMP_CALL_COUNTER[tag] = idx + 1
    return idx


@torch.no_grad()
def _pool_atom_pair_to_token(
    atom_pair: torch.Tensor,
    atom_to_token_idx_map: torch.Tensor,
    L_token: int,
    chunk: int = 1024,
) -> torch.Tensor:
    """Reduce (B=1, L_atom, L_atom, d) -> (L_token, L_token, d) by atom-mean
    of the *signed* per-channel value within each (token_row, token_col)
    block. ``d`` (= ``d_pair_atom``, the learned channel axis) is
    preserved so callers can render one heatmap per channel and see what
    each component of ``atom_pair`` contributes.

    Done in row chunks to keep peak memory at O(chunk * L_atom * d)
    instead of materialising the full (L_atom, L_atom, d) tensor on
    device.
    """
    assert atom_pair.dim() == 4 and atom_pair.shape[0] == 1, atom_pair.shape
    L_atom = atom_pair.shape[1]
    d = atom_pair.shape[3]
    device = atom_pair.device
    tokens = atom_to_token_idx_map[0].to(torch.long)  # (L_atom,)

    sums = torch.zeros(L_token, L_token, d, device=device, dtype=torch.float32)
    counts = torch.zeros(L_token, L_token, device=device, dtype=torch.float32)
    col_tok = tokens

    for s in range(0, L_atom, chunk):
        e = min(s + chunk, L_atom)
        chunk_vals = atom_pair[0, s:e].to(torch.float32)  # (e-s, L_atom, d)
        row_tok = tokens[s:e]
        flat_idx = row_tok.unsqueeze(1) * L_token + col_tok.unsqueeze(0)  # (e-s, L_atom)
        flat_idx_d = flat_idx.unsqueeze(-1).expand(-1, -1, d)
        sums.view(-1, d).scatter_add_(
            0, flat_idx_d.reshape(-1, d), chunk_vals.reshape(-1, d),
        )
        ones = torch.ones(flat_idx.shape, device=device, dtype=torch.float32)
        counts.view(-1).scatter_add_(0, flat_idx.reshape(-1), ones.reshape(-1))

    return (sums / counts.clamp(min=1.0).unsqueeze(-1)).cpu()


@torch.no_grad()
def _dump_atom_pair_snapshot(
    tag: str,
    stages: dict[str, torch.Tensor],
    atom_to_token_idx_map: torch.Tensor,
    L_token: int,
) -> None:
    """Pool each cumulative atom_pair stage and save one NPZ per call."""
    out_dir = _dump_dir()
    if out_dir is None:
        return
    call_idx = _next_call_idx(tag)
    pooled = {
        name: _pool_atom_pair_to_token(t, atom_to_token_idx_map, L_token).numpy()
        for name, t in stages.items()
    }
    pooled["atom_to_token_idx_map"] = atom_to_token_idx_map[0].cpu().numpy()
    np.savez(out_dir / f"atom_pair_{tag}_call{call_idx:04d}.npz", **pooled)


@typecheck
@torch.no_grad
def init_atom_features(
    reference: ReferenceFeatures,
) -> tuple[
    Float[torch.Tensor, "B L_atom d_single_atom_cond"],
    Float[torch.Tensor, "B L_atom L_atom d_pair_atom"],
]:
    """Get input feature for atom single and pair embedding."""
    atom_single_init = torch.cat(
        [
            reference.pos,
            reference.mask.unsqueeze(-1),
            reference.element.unsqueeze(-1),
            torch.arcsinh(reference.charge).unsqueeze(-1),
        ],
        dim=-1,
    )
    atom_single_init = atom_single_init * reference.mask.unsqueeze(-1)

    d_lm = reference.pos[:, :, None] - reference.pos[:, None, :]
    v_lm = reference.space_uid[:, :, None] == reference.space_uid[:, None, :]

    v_lm = v_lm[..., None].to(d_lm.dtype)
    arctan_d_lm = 1 / (1 + d_lm.norm(dim=-1) ** 2)
    arctan_d_lm = arctan_d_lm.unsqueeze(-1)
    d_lm = torch.cat([d_lm, arctan_d_lm, v_lm], dim=-1)
    atom_pair_init = d_lm * v_lm

    return atom_single_init, atom_pair_init


class AtomAttentionEncoder(nn.Module):
    """Atom attention encoder."""

    def __init__(
        self,
        shared_config: SharedConfig,
        diffusion_config: DiffusionTransformer.Config,
    ) -> None:
        super().__init__()
        self.shared_config = shared_config
        self.diffusion_config = diffusion_config
        d_single_atom = shared_config.d_single_atom
        d_pair_atom = shared_config.d_pair_atom
        self.d_single_token = shared_config.d_single_token
        d_pair = shared_config.d_pair

        self.use_checkpoint = shared_config.use_checkpoint

        self.to_atom_single_cond = Linear(6, shared_config.d_single_atom, bias=False)

        self.to_atom_pair = Linear(5, shared_config.d_pair_atom, bias=False)

        self.token_single_to_atom_single_cond = nn.Sequential(
            LayerNorm(
                shared_config.d_single,
            ),
            Linear(
                shared_config.d_single,
                d_single_atom,
                bias=False,
                init="zero",
            ),
        )
        self.token_pair_to_atom_pair = nn.Sequential(
            LayerNorm(d_pair),
            Linear(d_pair, d_pair_atom, bias=False, init="zero"),
        )
        self.noisy_to_atom_single_rep = Linear(
            3,
            d_single_atom,
            bias=True,
        )  # bias set to true for missing atoms

        self.atom_single_to_pair_left = nn.Sequential(
            nn.ReLU(),
            Linear(d_single_atom, d_pair_atom, bias=False),
        )

        self.atom_single_to_pair_right = nn.Sequential(
            nn.ReLU(),
            Linear(d_single_atom, d_pair_atom, bias=False),
        )

        self.mlp_atom_pair = nn.Sequential(
            Linear(d_pair_atom, d_pair_atom, init="relu", bias=False),
            nn.ReLU(),
            Linear(d_pair_atom, d_pair_atom, init="relu", bias=False),
            nn.ReLU(),
            Linear(d_pair_atom, d_pair_atom, init="zero", bias=False),
        )

        self.atom_transformer = DiffusionTransformer(config=diffusion_config)

        self.atom_single_rep_to_token_single = nn.Sequential(
            Linear(d_single_atom, self.d_single_token, bias=False),
            nn.ReLU(),
        )

    @typecheck
    def _before_atom_transformer(
        self,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        atom_single_init: Float[torch.Tensor, "B L_atom d_single_atom_init"],
        atom_pair_init: Float[torch.Tensor, "B L_atom L_atom d_pair_atom_init"],
        atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"],
        token_single_cond: Float[torch.Tensor, "B L_token d_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
        Float[torch.Tensor, "A B L_atom d_single_atom_cond"],
        Float[torch.Tensor, "B L_atom L_atom d_pair_atom_cond"],
    ]:
        atom_single_cond = self.to_atom_single_cond(atom_single_init)
        atom_pair = self.to_atom_pair(atom_pair_init)
        # Snapshot stage 1 — pure geometric init projection (no token / single
        # / MLP contribution yet). Captured before any "+= ..." so it's the
        # baseline against which subsequent stages compose.
        snap_init = atom_pair.detach() if _should_dump("diffusion_canonical") else None

        device = x_t.device
        num_aug, batch_size, atom_length = x_t.shape[:3]

        _to_add_single = self.token_single_to_atom_single_cond(token_single_cond)
        _to_add_pair = self.token_pair_to_atom_pair(token_pair_cond)

        batch_1d_idx = torch.arange(batch_size, device=device)
        batch_1d_idx = batch_1d_idx.view(batch_size, 1).expand(-1, atom_length)
        atom_single_cond = (
            atom_single_cond + _to_add_single[batch_1d_idx, atom_to_token_idx_map]
        )
        # Pair-cond gather: out[b, i, j] = _to_add_pair[b, A[b, i], A[b, j]].
        # Row/col index tensors must broadcast on orthogonal axes — passing
        # the same [B, L_atom] tensor in both slots collapses to the column
        # token's diagonal, so unsqueeze row to [B, L_atom, 1] and col to
        # [B, 1, L_atom].
        batch_2d_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        atom_pair = (
            atom_pair
            + _to_add_pair[
                batch_2d_idx,
                atom_to_token_idx_map.unsqueeze(-1),
                atom_to_token_idx_map.unsqueeze(-2),
            ]
        )
        snap_token_cond = atom_pair.detach() if snap_init is not None else None
        # augmentation
        atom_single_rep = atom_single_cond.unsqueeze(0)
        to_add = self.noisy_to_atom_single_rep(
            x_t.to(torch.float32),
        )
        to_add = to_add * x_mask.unsqueeze(-1)
        atom_single_rep = atom_single_rep + to_add
        _left = self.atom_single_to_pair_left(atom_single_cond)
        _right = self.atom_single_to_pair_right(atom_single_cond)
        atom_single_cond = atom_single_cond.unsqueeze(0).expand(num_aug, -1, -1, -1)

        atom_pair = atom_pair + _left[..., None, :] + _right[..., None, :, :]
        snap_singles = atom_pair.detach() if snap_init is not None else None
        atom_pair = atom_pair + self.mlp_atom_pair(atom_pair)
        if snap_init is not None:
            L_token = token_pair_cond.shape[1]
            _dump_atom_pair_snapshot(
                tag="diffusion_canonical",
                stages={
                    "1_geom_init": snap_init,
                    "2_after_token_cond": snap_token_cond,
                    "3_after_singles": snap_singles,
                    "4_after_mlp": atom_pair.detach(),
                },
                atom_to_token_idx_map=atom_to_token_idx_map,
                L_token=L_token,
            )
        return atom_single_rep, atom_single_cond, atom_pair

    # Inference-time chunking constant. Splits the first L_atom axis of
    # atom_pair so the token->atom gather and the pair MLP each materialise
    # only [B, _ATOM_CHUNK, L_atom, d] temporaries instead of the full
    # [B, L_atom, L_atom, d] (which OOMs once L_atom is in the 10k+ range).
    _ATOM_CHUNK: int = 1024

    @typecheck
    def _before_atom_transformer_chunked(
        self,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        atom_single_init: Float[torch.Tensor, "B L_atom d_single_atom_init"],
        atom_pair_init: Float[torch.Tensor, "B L_atom L_atom d_pair_atom_init"],
        atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"],
        token_single_cond: Float[torch.Tensor, "B L_token d_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
        Float[torch.Tensor, "A B L_atom d_single_atom_cond"],
        Float[torch.Tensor, "B L_atom L_atom d_pair_atom_cond"],
    ]:
        atom_single_cond = self.to_atom_single_cond(atom_single_init)
        atom_pair = self.to_atom_pair(atom_pair_init)
        dump = _should_dump("diffusion_chunked")
        L_token = token_pair_cond.shape[1]
        # Snapshot pooled stages eagerly — atom_pair is mutated in-place
        # below, so each stage must pool *before* the next ``add_()`` writes.
        snap_pools: dict[str, torch.Tensor] = {}
        if dump:
            snap_pools["1_geom_init"] = _pool_atom_pair_to_token(
                atom_pair, atom_to_token_idx_map, L_token,
            )

        device = x_t.device
        num_aug, batch_size, atom_length = x_t.shape[:3]

        _to_add_single = self.token_single_to_atom_single_cond(token_single_cond)
        _to_add_pair = self.token_pair_to_atom_pair(token_pair_cond)

        batch_1d_idx = torch.arange(batch_size, device=device)
        batch_1d_idx = batch_1d_idx.view(batch_size, 1).expand(-1, atom_length)
        atom_single_cond = (
            atom_single_cond + _to_add_single[batch_1d_idx, atom_to_token_idx_map]
        )

        _left = self.atom_single_to_pair_left(atom_single_cond)
        _right = self.atom_single_to_pair_right(atom_single_cond)

        # Pair-cond gather, per row-chunk to cap the temporary at
        # [B, chunk, L_atom, d_pair_atom] instead of [B, L_atom, L_atom, d].
        # Bit-exact with the canonical _before_atom_transformer pair gather:
        # out[b, i, j] = _to_add_pair[b, A[b, i], A[b, j]] with row/col
        # indices orthogonalized via unsqueeze(-1) / unsqueeze(-2).
        b_arange = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        col_tokens = atom_to_token_idx_map.unsqueeze(-2)  # [B, 1, L_atom]

        chunk = self._ATOM_CHUNK
        # First split steps 2 + 3 across two chunk loops *only when dumping*
        # so we can pool the after-token-cond and after-singles states
        # independently. Default path (no env var) keeps the single-loop
        # schedule for max throughput.
        if dump:
            for s in range(0, atom_length, chunk):
                e = min(s + chunk, atom_length)
                row_tokens = atom_to_token_idx_map[:, s:e].unsqueeze(-1)
                atom_pair[:, s:e].add_(
                    _to_add_pair[b_arange, row_tokens, col_tokens]
                )
            snap_pools["2_after_token_cond"] = _pool_atom_pair_to_token(
                atom_pair, atom_to_token_idx_map, L_token,
            )
            for s in range(0, atom_length, chunk):
                e = min(s + chunk, atom_length)
                atom_pair[:, s:e].add_(_left[:, s:e].unsqueeze(2))
                atom_pair[:, s:e].add_(_right.unsqueeze(1))
            snap_pools["3_after_singles"] = _pool_atom_pair_to_token(
                atom_pair, atom_to_token_idx_map, L_token,
            )
        else:
            for s in range(0, atom_length, chunk):
                e = min(s + chunk, atom_length)
                row_tokens = atom_to_token_idx_map[:, s:e].unsqueeze(-1)
                pair_slice = atom_pair[:, s:e]
                pair_slice.add_(
                    _to_add_pair[b_arange, row_tokens, col_tokens]
                )
                pair_slice.add_(_left[:, s:e].unsqueeze(2))
                pair_slice.add_(_right.unsqueeze(1))

        atom_single_rep = atom_single_cond.unsqueeze(0)
        to_add = self.noisy_to_atom_single_rep(x_t.to(torch.float32))
        to_add = to_add * x_mask.unsqueeze(-1)
        atom_single_rep = atom_single_rep + to_add
        atom_single_cond = atom_single_cond.unsqueeze(0).expand(num_aug, -1, -1, -1)

        for s in range(0, atom_length, chunk):
            e = min(s + chunk, atom_length)
            atom_pair[:, s:e].add_(self.mlp_atom_pair(atom_pair[:, s:e]))

        if dump:
            snap_pools["4_after_mlp"] = _pool_atom_pair_to_token(
                atom_pair, atom_to_token_idx_map, L_token,
            )
            out_dir = _dump_dir()
            assert out_dir is not None
            call_idx = _next_call_idx("diffusion_chunked")
            np.savez(
                out_dir / f"atom_pair_diffusion_chunked_call{call_idx:04d}.npz",
                **{k: v.numpy() for k, v in snap_pools.items()},
                atom_to_token_idx_map=atom_to_token_idx_map[0].cpu().numpy(),
            )

        return atom_single_rep, atom_single_cond, atom_pair

    @typecheck
    def _scatter_atom_to_token(
        self,
        token_idx: Int[torch.Tensor, "B L_token"],
        atom_mask: Bool[torch.Tensor, "B L_atom"],
        atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"],
        atom_single_rep: Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
    ) -> Float[torch.Tensor, "B L_token d_single_token"]:
        """Scatter atom single representation to token single representation."""
        dtype = atom_single_rep.dtype

        atom_single_rep = torch.where(
            atom_mask.unsqueeze(-1),
            atom_single_rep,
            torch.zeros_like(atom_single_rep),
        )

        token_length = int(token_idx.shape[1])

        # one-hot assignment: (B, L_atom, L_token)
        mapping = torch.nn.functional.one_hot(
            atom_to_token_idx_map,
            num_classes=token_length,
        ).to(dtype)
        mask_f = atom_mask.to(dtype)  # (B, L_atom)
        count = torch.einsum("bal,ba->bl", mapping, mask_f)

        # project atoms -> token feature dim: (A, B, L_atom, d_single_token)
        to_add_single_token_rep = self.atom_single_rep_to_token_single(atom_single_rep)

        # apply mask AFTER projection (prevents bias leakage if projection has bias)
        atom_mask = atom_mask.unsqueeze(0).unsqueeze(-1)  # (1, B, L_atom, 1)
        to_add_single_token_rep = (
            to_add_single_token_rep * atom_mask
        )  # (A, B, L_atom, d)

        # einsum over atoms -> token sum: (A, B, L_token, d)
        token_single_rep = torch.einsum(
            "bal,nbac->nblc",
            mapping,
            to_add_single_token_rep,
        ).contiguous()
        # Explanation of labels:
        # A:    (B, L_atom, L_token) -> "bal" (b=batch, a=atom, l=token)
        # to_add (A, B, L_atom, d)   -> "abac" where c=d, reuse a=atom
        # out:  (A, B, L_token, d)   -> "ablc"

        return token_single_rep / count.unsqueeze(0).unsqueeze(-1).clamp(min=1.0)

    @typecheck
    def forward(
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        token_single_cond: Float[torch.Tensor, "B L_token d_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B L_token d_single_token_rep"],
        Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
        Float[torch.Tensor, "A B L_atom d_single_atom_cond"],
        Float[torch.Tensor, "A B L_atom L_atom d_pair_atom"],
    ]:
        """Forward pass."""
        atom_single_init, atom_pair_init = init_atom_features(reference)
        atom_to_token_idx_map = scheme.atom_to_token_idx_map

        # Inference chunking is opt-in via env var. The chunked path is now
        # bit-exact with the canonical _before_atom_transformer — see
        # tests/test_atom_attention_chunked.py — bit-exact with the canonical path,
        # which OOMs on [B, L_atom, L_atom, d] temporaries at ~13k atoms.
        # Default fallback: chunk the atom attention whenever not training (the canonical
        # [B, L_atom, L_atom, d] broadcast OOMs at large L_atom); never chunk in training.
        use_chunked_inference = not self.training
        if use_chunked_inference:
            atom_single_rep, atom_single_cond, atom_pair = (
                self._before_atom_transformer_chunked(
                    x_t,
                    x_mask,
                    atom_single_init,
                    atom_pair_init,
                    atom_to_token_idx_map,
                    token_single_cond,
                    token_pair_cond,
                )
            )
            # Drop the [B, L_atom, L_atom, 5] init tensor (~3.8 GiB at L_atom=13k)
            # now that to_atom_pair has already consumed it.
            del atom_single_init, atom_pair_init
        elif self.use_checkpoint and self.training:
            atom_single_rep, atom_single_cond, atom_pair = checkpoint(
                self._before_atom_transformer,
                x_t,
                x_mask,
                atom_single_init,
                atom_pair_init,
                atom_to_token_idx_map,
                token_single_cond,
                token_pair_cond,
                use_reentrant=False,
            )  # pyright: ignore[reportGeneralTypeIssues]
        else:
            atom_single_rep, atom_single_cond, atom_pair = self._before_atom_transformer(
                x_t,
                x_mask,
                atom_single_init,
                atom_pair_init,
                atom_to_token_idx_map,
                token_single_cond,
                token_pair_cond,
            )
        atom_single_rep = self.atom_transformer(
            atom_single_rep,
            atom_single_cond,
            atom_pair,
            structure.atom_mask,
        )

        if self.use_checkpoint:
            token_single_rep = checkpoint(
                self._scatter_atom_to_token,
                scheme.token_idx,
                structure.atom_mask,
                atom_to_token_idx_map,
                atom_single_rep,
                use_reentrant=False,
            )
        else:
            token_single_rep = self._scatter_atom_to_token(
                scheme.token_idx,  # pyright: ignore[reportCallIssue]
                structure.atom_mask,  # pyright: ignore[reportCallIssue]
                atom_to_token_idx_map,
                atom_single_rep,
            )
        return token_single_rep, atom_single_rep, atom_single_cond, atom_pair  # pyright: ignore[reportReturnType]


class AtomAttentionDecoder(nn.Module):
    """Atom attention decoder."""

    def __init__(
        self,
        shared_config: SharedConfig,
        diffusion_config: DiffusionTransformer.Config,
    ) -> None:
        super().__init__()
        self.shared_config = shared_config
        self.diffusion_config = diffusion_config
        d_single_atom = shared_config.d_single_atom
        d_single_token = shared_config.d_single_token

        self.add_token_info = Linear(d_single_token, d_single_atom, bias=False)

        self.atom_transformer = DiffusionTransformer(config=diffusion_config)

        self.final_denoising = nn.Sequential(
            LayerNorm(d_single_atom),
            Linear(
                d_single_atom,
                3,
                bias=False,
                init="zero",
            ),
        )

    @typecheck
    def forward(
        self,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        token_single_rep: Float[torch.Tensor, "A B L_token d_single_token"],
        atom_single_rep: Float[torch.Tensor, "A B L_atom d_single_atom_rep"],
        atom_single_cond: Float[torch.Tensor, "A B L_atom d_single_atom_cond"],
        atom_pair: Float[torch.Tensor, "A B L_atom L_atom d_pair_atom"],
    ) -> Float[torch.Tensor, "A B L_atom 3"]:
        """Forward pass."""
        num_augment, batch_size, atom_length = atom_single_rep.shape[:3]
        device = atom_single_rep.device
        batch_1d_idx = torch.arange(batch_size, device=device)
        batch_1d_idx = batch_1d_idx.view(1, batch_size, 1).expand(
            num_augment,
            -1,
            atom_length,
        )
        aug_1d_idx = torch.arange(num_augment, device=device)
        aug_1d_idx = aug_1d_idx.view(num_augment, 1, 1).expand(
            -1,
            batch_size,
            atom_length,
        )
        atom_to_token_idx_map = scheme.atom_to_token_idx_map
        atom_to_token_idx_map = atom_to_token_idx_map.unsqueeze(0).expand(
            num_augment,
            -1,
            -1,
        )

        _to_add_single = self.add_token_info(token_single_rep)
        atom_single_rep = (
            atom_single_rep
            + _to_add_single[aug_1d_idx, batch_1d_idx, atom_to_token_idx_map]
        )

        atom_single_rep = self.atom_transformer(
            atom_single_rep,
            atom_single_cond,
            atom_pair,
            mask=structure.atom_mask,
        )
        return self.final_denoising(atom_single_rep)


# ===========================================================================
# ESMFold2-style SWA atom attention (no atom-pair tensor; 3D RoPE).
#
# Drop-in alternatives to AtomAttentionEncoder / AtomAttentionDecoder, selected
# by ``DiffusionModule(swa_atom_config=...)``. The atom-pair representation is
# removed entirely — inter-atom geometry enters through 3D RoPE on q/k inside
# the SWA transformer. The encoder hands its precomputed RoPE + varlen unpadding
# tensors ("attention_params") to the decoder, which reuses them (they depend
# only on the fixed reference conformer, so they are identical in both).
# ===========================================================================


def init_atom_single_features(
    reference: ReferenceFeatures,
) -> Float[torch.Tensor, "B L_atom 6"]:
    """Per-atom single features (pos, mask, element, charge) — the single half of
    :func:`init_atom_features` without the O(L_atom^2) pair tensor.
    """
    atom_single_init = torch.cat(
        [
            reference.pos,
            reference.mask.unsqueeze(-1),
            reference.element.unsqueeze(-1),
            torch.arcsinh(reference.charge).unsqueeze(-1),
        ],
        dim=-1,
    )
    return atom_single_init * reference.mask.unsqueeze(-1)


def _scatter_atom_to_token(
    proj: nn.Module,
    token_idx: Int[torch.Tensor, "B L_token"],
    atom_mask: Bool[torch.Tensor, "B L_atom"],
    atom_to_token_idx_map: Int[torch.Tensor, "B L_atom"],
    atom_single_rep: Float[torch.Tensor, "A B L_atom d_single_atom"],
) -> Float[torch.Tensor, "A B L_token d_single_token"]:
    """Masked atom->token scatter-mean (same op as AtomAttentionEncoder)."""
    dtype = atom_single_rep.dtype
    token_length = int(token_idx.shape[1])
    mapping = torch.nn.functional.one_hot(
        atom_to_token_idx_map, num_classes=token_length
    ).to(dtype)
    mask_f = atom_mask.to(dtype)
    count = torch.einsum("bal,ba->bl", mapping, mask_f)
    to_add = proj(atom_single_rep) * atom_mask.unsqueeze(0).unsqueeze(-1)
    token_single_rep = torch.einsum("bal,nbac->nblc", mapping, to_add).contiguous()
    return token_single_rep / count.unsqueeze(0).unsqueeze(-1).clamp(min=1.0)


class SWAAtomAttentionEncoder(nn.Module):
    """ESMFold2 SWAAtomEncoder (Algorithm 6): atoms -> tokens, no atom pair."""

    def __init__(
        self,
        shared_config: SharedConfig,
        swa_config: SWAAtomTransformer.Config,
    ) -> None:
        super().__init__()
        d_single_atom = shared_config.d_single_atom
        self.d_single_token = shared_config.d_single_token

        self.to_atom_single_cond = Linear(6, d_single_atom, bias=False)
        self.token_single_to_atom_single_cond = nn.Sequential(
            LayerNorm(shared_config.d_single),
            Linear(shared_config.d_single, d_single_atom, bias=False, init="zero"),
        )
        self.noisy_to_atom_single_rep = Linear(3, d_single_atom, bias=True)

        resolved = swa_config.model_copy(
            update={"d_atom": d_single_atom, "d_cond": d_single_atom}
        )
        if (
            getattr(resolved, "local_structure_attn", False)
            and getattr(resolved, "block_style", "esmfold2") != "af3"
        ):
            raise ValueError("local_structure_attn is currently supported only with block_style='af3'.")
        self.swa_config = resolved
        self.atom_transformer = _make_atom_transformer(resolved)

        self.atom_single_rep_to_token_single = nn.Sequential(
            Linear(d_single_atom, self.d_single_token, bias=False),
            nn.ReLU(),
        )

    def forward(
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        token_single_cond: Float[torch.Tensor, "B L_token d_single"],
    ) -> tuple[
        Float[torch.Tensor, "A B L_token d_single_token"],
        Float[torch.Tensor, "A B L_atom d_single_atom"],
        Float[torch.Tensor, "A B L_atom d_single_atom"],
        tuple,
    ]:
        """Returns (token_single_rep, atom_single_rep, atom_single_cond, attn_params)."""
        num_aug, batch_size, atom_length = x_t.shape[:3]
        device = x_t.device
        atom_to_token = scheme.atom_to_token_idx_map

        atom_single_init = init_atom_single_features(reference)
        atom_single_cond = self.to_atom_single_cond(atom_single_init)  # [B, L, d]

        to_add_single = self.token_single_to_atom_single_cond(token_single_cond)
        batch_1d_idx = torch.arange(batch_size, device=device).view(batch_size, 1)
        batch_1d_idx = batch_1d_idx.expand(-1, atom_length)
        atom_single_cond = (
            atom_single_cond + to_add_single[batch_1d_idx, atom_to_token]
        )  # [B, L, d]

        atom_single_rep = atom_single_cond.unsqueeze(0)
        to_add = self.noisy_to_atom_single_rep(x_t.to(torch.float32))
        atom_single_rep = atom_single_rep + to_add * x_mask.unsqueeze(-1)  # [A, B, L, d]
        atom_single_cond = atom_single_cond.unsqueeze(0).expand(num_aug, -1, -1, -1)

        # 3D RoPE + varlen params (step- and augment-invariant).
        cos, sin = self.atom_transformer.build_rope(reference.pos, reference.space_uid)
        valid = (
            structure.atom_mask.unsqueeze(0)
            .expand(num_aug, -1, -1)
            .reshape(num_aug * batch_size, atom_length)
        )
        attn_params = build_attention_params(cos, sin, valid, num_aug)
        if getattr(self.swa_config, "local_structure_attn", False):
            x_t_flat = x_t.reshape(num_aug * batch_size, atom_length, 3)
            neighbor_idx, neighbor_mask = build_local_structure_neighbor_indices(
                x_t_flat,
                valid,
                seq_neighbors=self.swa_config.seq_neighbors,
                structure_neighbors=self.swa_config.structure_neighbors,
                query_chunk_size=self.swa_config.structure_query_chunk_size,
            )
            attn_params = (
                *attn_params,
                neighbor_idx,
                neighbor_mask,
                self.swa_config.sparse_attention_query_chunk_size,
            )

        d = atom_single_rep.shape[-1]
        q = atom_single_rep.reshape(num_aug * batch_size, atom_length, d)
        c = atom_single_cond.reshape(num_aug * batch_size, atom_length, d)
        q = self.atom_transformer(q, c, attn_params)
        atom_single_rep = q.reshape(num_aug, batch_size, atom_length, d)

        token_single_rep = _scatter_atom_to_token(
            self.atom_single_rep_to_token_single,
            scheme.token_idx,
            structure.atom_mask,
            atom_to_token,
            atom_single_rep,
        )
        return token_single_rep, atom_single_rep, atom_single_cond, attn_params


class SWAAtomAttentionDecoder(nn.Module):
    """ESMFold2 SWAAtomDecoder (Algorithm 7): tokens -> atoms, no atom pair."""

    def __init__(
        self,
        shared_config: SharedConfig,
        swa_config: SWAAtomTransformer.Config,
    ) -> None:
        super().__init__()
        d_single_atom = shared_config.d_single_atom
        d_single_token = shared_config.d_single_token

        self.add_token_info = Linear(d_single_token, d_single_atom, bias=False)
        resolved = swa_config.model_copy(
            update={"d_atom": d_single_atom, "d_cond": d_single_atom}
        )
        self.atom_transformer = _make_atom_transformer(resolved)
        self.final_denoising = nn.Sequential(
            LayerNorm(d_single_atom),
            Linear(d_single_atom, 3, bias=False, init="zero"),
        )

    def forward(
        self,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        token_single_rep: Float[torch.Tensor, "A B L_token d_single_token"],
        atom_single_rep: Float[torch.Tensor, "A B L_atom d_single_atom"],
        atom_single_cond: Float[torch.Tensor, "A B L_atom d_single_atom"],
        attn_params: tuple,
    ) -> Float[torch.Tensor, "A B L_atom 3"]:
        """Forward pass; ``attn_params`` is reused from the encoder."""
        num_aug, batch_size, atom_length = atom_single_rep.shape[:3]
        device = atom_single_rep.device

        aug_1d_idx = torch.arange(num_aug, device=device).view(num_aug, 1, 1)
        aug_1d_idx = aug_1d_idx.expand(-1, batch_size, atom_length)
        batch_1d_idx = torch.arange(batch_size, device=device).view(1, batch_size, 1)
        batch_1d_idx = batch_1d_idx.expand(num_aug, -1, atom_length)
        atom_to_token = scheme.atom_to_token_idx_map.unsqueeze(0).expand(
            num_aug, -1, -1
        )

        to_add = self.add_token_info(token_single_rep)
        atom_single_rep = (
            atom_single_rep + to_add[aug_1d_idx, batch_1d_idx, atom_to_token]
        )

        d = atom_single_rep.shape[-1]
        q = atom_single_rep.reshape(num_aug * batch_size, atom_length, d)
        c = atom_single_cond.reshape(num_aug * batch_size, atom_length, d)
        q = self.atom_transformer(q, c, attn_params)
        atom_single_rep = q.reshape(num_aug, batch_size, atom_length, d)
        return self.final_denoising(atom_single_rep)


class DiffusionConditioning(nn.Module):
    """Diffusion conditioning module."""

    class Config(BaseModel):
        """Configuration for DiffusionConditioning."""

        n_expand: int = 2
        n_blocks: int = 2

    def __init__(
        self,
        shared_config: SharedConfig,
        dit_cond_config: Config,
    ) -> None:
        super().__init__()
        d_pair = shared_config.d_pair
        d_time = shared_config.d_time
        d_single = shared_config.d_single
        self.relative_position_embedder = RelativePositionEmbedding(
            d_hidden=d_pair,
            r_max=shared_config.r_max,
            s_max=shared_config.s_max,
        )

        self.linear_token_pair = nn.Sequential(
            LayerNorm(
                2 * d_pair,
            ),
            Linear(2 * d_pair, d_pair, bias=False),
        )
        self.pair_transitions = nn.ModuleList(
            [
                Transition(
                    d_hidden=d_pair,
                    n=dit_cond_config.n_expand,
                )
                for _ in range(dit_cond_config.n_blocks)
            ],
        )
        self.linear_token_single = nn.Sequential(
            LayerNorm(
                shared_config.d_single_token_input + shared_config.d_single,
            ),
            Linear(
                shared_config.d_single_token_input + shared_config.d_single,
                d_single,
                bias=False,
            ),
        )
        self.add_time_embedding = nn.Sequential(
            LayerNorm(
                d_time,
            ),
            Linear(d_time, d_single, bias=False),
        )
        self.single_transitions = nn.ModuleList(
            [
                Transition(
                    d_hidden=d_single,
                    n=dit_cond_config.n_expand,
                )
                for _ in range(dit_cond_config.n_blocks)
            ],
        )
        self.final_layernorm_token_single = LayerNorm(d_single)

    @typecheck
    def forward(
        self,
        scheme: SchemeFeatures,
        t_emb: Float[torch.Tensor, "A B"],
        token_single_input: Float[torch.Tensor, "B L_token d_single_input"],
        token_single_trunk: Float[torch.Tensor, "B L_token d_single"],
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B L_token d_single"],
        Float[torch.Tensor, "B L_token L_token d_pair"],
    ]:
        """Forward pass of the diffusion conditioning module."""
        rel_emb = self.relative_position_embedder(
            asym_id=scheme.token_asym_id,
            token_residue_idx=scheme.token_residue_idx,
            token_idx=scheme.token_idx,
            entity_id=scheme.token_entity_id,
            sym_id=scheme.token_sym_id,
        )
        token_pair = torch.cat([token_pair_trunk, rel_emb], dim=-1)
        token_pair = self.linear_token_pair(token_pair)

        for transition in self.pair_transitions:
            token_pair = transition(token_pair)

        token_single = torch.cat([token_single_input, token_single_trunk], dim=-1)

        token_single = self.linear_token_single(token_single)
        time_embedding = fourier_embedding(t_emb)
        time_embedding = time_embedding.squeeze(-2)
        token_single = token_single + self.add_time_embedding(time_embedding)

        for transition in self.single_transitions:
            token_single = transition(token_single)

        token_single = self.final_layernorm_token_single(token_single)

        return token_single, token_pair


class DiffusionModule(nn.Module):
    """Diffusion module for processing input features."""

    def __init__(
        self,
        shared_config: SharedConfig,
        atom_dit_config: DiffusionTransformer.Config,
        token_dit_config: DiffusionTransformer.Config,
        dit_cond_config: DiffusionConditioning.Config,
        swa_atom_config: SWAAtomTransformer.Config | None = None,
    ) -> None:
        super().__init__()
        self.diffusion_conditioning = DiffusionConditioning(
            shared_config=shared_config,
            dit_cond_config=dit_cond_config,
        )
        # When ``swa_atom_config`` is given, the atom encoder/decoder are the
        # ESMFold2 SWA stack (3D RoPE, sliding window, no atom-pair tensor);
        # otherwise the default AF3 pair-bias atom attention is used.
        self.use_swa_atom = swa_atom_config is not None
        if self.use_swa_atom:
            self.atom_attention_encoder = SWAAtomAttentionEncoder(
                shared_config=shared_config,
                swa_config=swa_atom_config,
            )
            self.atom_attention_decoder = SWAAtomAttentionDecoder(
                shared_config=shared_config,
                swa_config=swa_atom_config,
            )
        else:
            self.atom_attention_encoder = AtomAttentionEncoder(
                shared_config=shared_config,
                diffusion_config=atom_dit_config,
            )
            self.atom_attention_decoder = AtomAttentionDecoder(
                shared_config=shared_config,
                diffusion_config=atom_dit_config,
            )
        self.add_single_token_cond = nn.Sequential(
            LayerNorm(
                shared_config.d_single,
            ),
            Linear(
                shared_config.d_single,
                shared_config.d_single_token,
                bias=False,
                init="zero",
            ),
        )
        self.diffusion_transformer = DiffusionTransformer(config=token_dit_config)
        self.ln_token_single_rep = LayerNorm(
            shared_config.d_single_token,
        )

    def _encode_atoms(
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        enc_token_single: Float[torch.Tensor, "B L_token d_single"],
        token_pair_cond: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> tuple:
        """Run the atom encoder; returns (token_single_rep, atom_single_rep,
        atom_single_cond, carry) where ``carry`` is the atom-pair tensor (AF3) or
        the SWA ``attention_params`` to thread into the decoder.
        """
        if self.use_swa_atom:
            return self.atom_attention_encoder(
                reference, scheme, structure, x_t, x_mask, enc_token_single
            )
        return self.atom_attention_encoder(
            reference, scheme, structure, x_t, x_mask, enc_token_single, token_pair_cond
        )

    def _decode_atoms(
        self,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        token_single_rep: torch.Tensor,
        atom_single_rep: torch.Tensor,
        atom_single_cond: torch.Tensor,
        carry,
    ) -> torch.Tensor:
        """Run the atom decoder. ``carry`` is atom_pair (AF3) or attn_params (SWA)."""
        return self.atom_attention_decoder(
            scheme,
            structure,
            token_single_rep,
            atom_single_rep,
            atom_single_cond,
            carry,
        )

    @typecheck
    def forward(
        self,
        reference: ReferenceFeatures,
        scheme: SchemeFeatures,
        structure: StructureFeatures,
        x_t: Float[torch.Tensor, "A B L_atom 3"],
        x_mask: Bool[torch.Tensor, "A B L_atom"],
        t_emb: Float[torch.Tensor, "A B"],
        token_single_input: Float[torch.Tensor, "B L_token d_single_token_input"],
        token_single_trunk: Float[torch.Tensor, "B L_token d_single"],
        token_pair_trunk: Float[torch.Tensor, "B L_token L_token d_pair"],
    ) -> Float[torch.Tensor, "B L_atom 3"]:
        """Forward pass of the diffusion module."""
        token_single_cond, token_pair_cond = self.diffusion_conditioning(
            scheme,
            t_emb,
            token_single_input,
            token_single_trunk,
            token_pair_trunk,
        )
        token_single_rep, atom_single_rep, atom_single_cond, carry = self._encode_atoms(
            reference,
            scheme,
            structure,
            x_t,
            x_mask,
            token_single_trunk,
            token_pair_cond,
        )
        token_single_rep = token_single_rep + self.add_single_token_cond(
            token_single_cond,
        )
        token_single_rep = self.diffusion_transformer(
            token_single_rep,
            token_single_cond,
            token_pair_cond,
            mask=structure.token_mask.unsqueeze(0).expand(
                token_single_rep.shape[0],
                -1,
                -1,
            ),
        )

        token_single_rep = self.ln_token_single_rep(token_single_rep)
        return self._decode_atoms(
            scheme,
            structure,
            token_single_rep,
            atom_single_rep,
            atom_single_cond,
            carry,
        )
