"""Per-step 2D-matrix capture for the MiniWorld diffusion module.

Three full-atom × full-atom matrices are dumped per diffusion step:

  1. atom_single pairwise L2 distance — computed in the analyze script
     from the saved atom_single feature matrix (full L_atom × d, fp16,
     no downsampling here so the distance can be recomputed exactly).
     Captured at the *input* of each atom-level transformer stack and
     after every block.

  2. atom_pair feature L2 norm — full L_atom × L_atom matrix downsampled
     to a fixed grid (default 1024 × 1024) via block-mean. atom_pair is
     fed unchanged into every block of a stack, so one matrix per stack
     per step.

  3. atom QK attention softmax probability — head-averaged. The chunked
     softmax recomputation streams over query rows and writes into the
     downsampled grid directly, so the full L_q × L_k matrix never lives
     on the GPU.

No chain aggregation, no nano restriction. The raw 2D matrices.

Layout on disk:
  <save_dir>/
    scheme.pt
    layout.json
    step_000/
      encoder/
        single_input.pt           # (L_atom, d_atom_single) fp16
        single_block_00.pt
        single_block_01.pt
        single_block_02.pt
        pair_norm_ds.pt           # (≈ds, ≈ds) fp16, full atom_pair norm
        attn_block_00_ds.pt       # (≈ds, ≈ds) fp16, head-avg softmax
        attn_block_01_ds.pt
        attn_block_02_ds.pt
      decoder/
        (same layout)
    step_001/...

Only the atom-level transformer stacks are instrumented.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch

from miniworld.modules.diffusion_module import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionModule,
)
from team_gm.modules.blocks.diffusion_transformer import (
    DiffusionTransformer,
)
from miniworld_engine.modules import (
    AugmentedAttentionPairBias,
)

_STACK_ENCODER = "encoder"
_STACK_DECODER = "decoder"
_ATOM_STACKS = (_STACK_ENCODER, _STACK_DECODER)

_DEFAULT_QUERY_CHUNK = 256
_DEFAULT_DOWNSAMPLE = 1024


def _to_cpu_fp16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", dtype=torch.float16, copy=False)


def _save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def _block_mean_2d(m: torch.Tensor, target: int) -> torch.Tensor:
    """Block-average a 2D matrix down to (≤target, ≤target).

    Edge atoms beyond the largest exact-multiple block are dropped (at
    most ``block_size-1`` rows/cols per axis). With L_atom ~ 11364 and
    target 1024 this drops at most ~10 atoms on each axis, which is
    invisible at the display resolution we end up plotting.
    """
    L_q, L_k = m.shape
    bq = max(1, L_q // target)
    bk = max(1, L_k // target)
    new_q = L_q // bq
    new_k = L_k // bk
    m = m[: new_q * bq, : new_k * bk]
    return m.view(new_q, bq, new_k, bk).float().mean(dim=(1, 3))


class DiffusionCapture(AbstractContextManager):
    """Monkey-patches the diffusion module to dump full-atom 2D matrices.

    Parameters
    ----------
    diffusion_module:
        ``client.model.diffusion_module`` (unwrap Fabric first).
    save_dir:
        Output directory.
    downsample:
        Target side length for the downsampled pair/attention matrices.
        Defaults to 1024.
    capture_single / capture_pair / capture_attention:
        Per-matrix opt-out.
    query_chunk:
        Query-chunk size for the chunked attention recomputation.

    Notes
    -----
    * Disable ``torch.compile`` before entering.
    * Assumes A=1, B=1.
    """

    def __init__(
        self,
        diffusion_module: DiffusionModule,
        save_dir: Path | str,
        *,
        downsample: int = _DEFAULT_DOWNSAMPLE,
        capture_single: bool = True,
        capture_pair: bool = True,
        capture_attention: bool = True,
        query_chunk: int = _DEFAULT_QUERY_CHUNK,
    ) -> None:
        self.diffusion_module = diffusion_module
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.downsample = int(downsample)
        self.capture_single = capture_single
        self.capture_pair = capture_pair
        self.capture_attention = capture_attention
        self.query_chunk = query_chunk

        self._step: int = -1
        self._scheme_saved: bool = False
        self._current_stack: str | None = None
        self._current_block_idx: int = -1

        self._original_dm_forward = None
        self._original_dt_forwards: list[tuple[DiffusionTransformer, Any]] = []
        self._original_attn_kernels: list[
            tuple[AugmentedAttentionPairBias, Any]
        ] = []

        self._dt_stacks: dict[int, str] = {}
        self._attn_blocks: dict[str, list[AugmentedAttentionPairBias]] = {
            _STACK_ENCODER: [],
            _STACK_DECODER: [],
        }

    def __enter__(self) -> "DiffusionCapture":
        self._discover_stacks()
        self._install_patches()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._uninstall_patches()

    def _discover_stacks(self) -> None:
        dm = self.diffusion_module
        enc = dm.atom_attention_encoder
        dec = dm.atom_attention_decoder
        assert isinstance(enc, AtomAttentionEncoder)
        assert isinstance(dec, AtomAttentionDecoder)

        self._dt_stacks[id(enc.atom_transformer)] = _STACK_ENCODER
        self._dt_stacks[id(dec.atom_transformer)] = _STACK_DECODER

        for stack_name, dt in [
            (_STACK_ENCODER, enc.atom_transformer),
            (_STACK_DECODER, dec.atom_transformer),
        ]:
            for block in dt.blocks:
                attn = block.attention_pair_bias
                assert isinstance(attn, AugmentedAttentionPairBias)
                self._attn_blocks[stack_name].append(attn)

    def _install_patches(self) -> None:
        cap = self

        dm = self.diffusion_module
        self._original_dm_forward = dm.forward

        def patched_dm_forward(*args, **kwargs):
            cap._step += 1
            cap._on_dm_forward_enter(args, kwargs)
            return cap._original_dm_forward(*args, **kwargs)

        dm.forward = patched_dm_forward  # type: ignore[method-assign]

        for dt_id, stack_name in self._dt_stacks.items():
            dt = self._get_dt_by_id(dt_id)
            original_forward = dt.forward
            self._original_dt_forwards.append((dt, original_forward))

            def make_patched(_dt: DiffusionTransformer, _stack: str):
                def patched_dt_forward(single, cond, pair, mask=None):
                    cap._save_pair(_stack, pair)
                    cap._current_stack = _stack
                    cap._save_single(_stack, "input", single)
                    out = single
                    for i, block in enumerate(_dt.blocks):
                        cap._current_block_idx = i
                        out = block(out, cond, pair, mask)
                        cap._save_single(_stack, f"block_{i:02d}", out)
                    cap._current_block_idx = -1
                    cap._current_stack = None
                    return out

                return patched_dt_forward

            dt.forward = make_patched(dt, stack_name)  # type: ignore[method-assign]

        if not self.capture_attention:
            return
        for stack_name, attns in self._attn_blocks.items():
            for attn in attns:
                original_kernel = attn._kernel_attention_pair_bias
                self._original_attn_kernels.append((attn, original_kernel))

                def make_patched_kernel(_attn: AugmentedAttentionPairBias, _orig):
                    def patched_kernel(query, key, value, bias, mask=None):
                        try:
                            cap._save_attention(
                                query.detach().clone(),
                                key.detach(),
                                bias.detach(),
                                mask,
                            )
                        except Exception as e:  # noqa: BLE001
                            cap._log_failure("attn", e)
                        return _orig(query, key, value, bias, mask)

                    return patched_kernel

                attn._kernel_attention_pair_bias = make_patched_kernel(  # type: ignore[method-assign]
                    attn, original_kernel,
                )

    def _uninstall_patches(self) -> None:
        dm = self.diffusion_module
        if self._original_dm_forward is not None:
            dm.forward = self._original_dm_forward  # type: ignore[method-assign]
            self._original_dm_forward = None
        for dt, original in self._original_dt_forwards:
            dt.forward = original  # type: ignore[method-assign]
        self._original_dt_forwards.clear()
        for attn, original in self._original_attn_kernels:
            attn._kernel_attention_pair_bias = original  # type: ignore[method-assign]
        self._original_attn_kernels.clear()

    def _get_dt_by_id(self, target_id: int) -> DiffusionTransformer:
        dm = self.diffusion_module
        for dt in (
            dm.atom_attention_encoder.atom_transformer,
            dm.atom_attention_decoder.atom_transformer,
        ):
            if id(dt) == target_id:
                return dt
        msg = f"DiffusionTransformer id {target_id} not in atom stacks."
        raise KeyError(msg)

    def _step_dir(self) -> Path:
        return self.save_dir / f"step_{self._step:03d}"

    def _stack_dir(self, stack: str) -> Path:
        return self._step_dir() / stack

    def _on_dm_forward_enter(self, args, kwargs) -> None:
        if self._scheme_saved:
            return
        bound = self._bind_dm_args(args, kwargs)
        scheme = bound["scheme"]
        structure = bound["structure"]
        atom_chain = scheme.atom_to_chain_id[0].detach().cpu()
        token_asym = scheme.token_asym_id[0].detach().cpu()
        atom_to_token = scheme.atom_to_token_idx_map[0].detach().cpu()
        atom_mask = structure.atom_mask[0].detach().cpu()
        _save(
            self.save_dir / "scheme.pt",
            {
                "atom_to_chain_id": atom_chain,
                "token_asym_id": token_asym,
                "atom_to_token_idx_map": atom_to_token,
                "atom_mask": atom_mask,
                "n_atom": int(atom_chain.numel()),
            },
        )
        (self.save_dir / "layout.json").write_text(
            json.dumps(
                {
                    "n_atom": int(atom_chain.numel()),
                    "n_token": int(token_asym.numel()),
                    "atom_stacks": list(_ATOM_STACKS),
                    "n_blocks_per_stack": {
                        stack: len(self._attn_blocks[stack])
                        for stack in _ATOM_STACKS
                    },
                    "downsample": int(self.downsample),
                },
                indent=2,
            ),
        )
        self._scheme_saved = True

    def _save_single(
        self,
        stack: str,
        when: str,
        single: torch.Tensor,
    ) -> None:
        if not self.capture_single:
            return
        # single: (A=1, B=1, L_atom, d) — save full (L_atom, d) so analyze
        # can compute the exact pairwise distance matrix.
        _save(
            self._stack_dir(stack) / f"single_{when}.pt",
            _to_cpu_fp16(single[0, 0]),
        )

    def _save_pair(self, stack: str, pair: torch.Tensor) -> None:
        if not self.capture_pair:
            return
        out_path = self._stack_dir(stack) / "pair_norm_ds.pt"
        if out_path.exists():
            return
        # pair: (B=1, L, L, d) -> (L, L, d) -> (L, L)
        # Compute the full feature-norm map (≈258 MB fp16 for H1312), then
        # block-mean to the requested downsampled grid.
        p = pair[0].detach()
        # Compute norm in chunks over rows to keep peak memory modest;
        # writes a fp16 (L, L) buffer on the same device.
        L = p.shape[0]
        norm = torch.empty(L, L, device=p.device, dtype=torch.float16)
        chunk = 1024
        for start in range(0, L, chunk):
            end = min(start + chunk, L)
            sl = p[start:end].float().norm(dim=-1)
            norm[start:end] = sl.to(torch.float16)
            del sl
        ds = _block_mean_2d(norm, self.downsample)
        _save(out_path, _to_cpu_fp16(ds))
        del norm, ds, p

    def _save_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        """Full atom × atom softmax-probability matrix, head-averaged.

        We build the L_q × L_k probability matrix in a chunked query loop
        and write it to a fp16 GPU buffer; then block-mean to the
        downsampled grid for disk.
        """
        if self._current_stack not in _ATOM_STACKS:
            return
        if self._current_block_idx < 0:
            return
        block_idx = self._current_block_idx
        out_path = self._stack_dir(self._current_stack) / (
            f"attn_block_{block_idx:02d}_ds.pt"
        )

        # query, key: (A=1, B=1, L, H, D); bias: (B=1, L_q, L_k, H);
        # mask: (A, B, L) or (B, L).
        q = query[0, 0]
        k = key[0, 0]
        b = bias[0]
        L_q, n_head, d_head = q.shape
        L_k = k.shape[0]
        scale = d_head ** -0.5

        if mask is not None:
            m = mask if mask.ndim == 2 else mask[0]
            m = m[0] if m.ndim == 2 else m
        else:
            m = None

        full = torch.empty(L_q, L_k, device=q.device, dtype=torch.float16)
        chunk = max(1, int(self.query_chunk))
        for start in range(0, L_q, chunk):
            end = min(start + chunk, L_q)
            q_chunk = q[start:end] * scale          # (cq, H, D)
            scores = torch.einsum("qhd,khd->qhk", q_chunk, k)
            b_chunk = b[start:end].permute(0, 2, 1)  # (cq, H, L_k)
            scores = scores + b_chunk
            if m is not None:
                scores = scores.masked_fill(~m.view(1, 1, L_k), float("-inf"))
            p = torch.softmax(scores.float(), dim=-1)  # (cq, H, L_k)
            full[start:end] = p.mean(dim=1).to(torch.float16)
            del scores, p

        ds = _block_mean_2d(full, self.downsample)
        _save(out_path, _to_cpu_fp16(ds))
        del full, ds

    def _log_failure(self, kind: str, e: Exception) -> None:
        msg_path = self.save_dir / "capture_errors.log"
        with msg_path.open("a") as f:
            f.write(
                f"step={self._step} stack={self._current_stack} "
                f"block={self._current_block_idx} kind={kind} "
                f"err={type(e).__name__}: {e}\n",
            )

    @staticmethod
    def _bind_dm_args(args: tuple, kwargs: dict) -> dict:
        names = [
            "reference",
            "scheme",
            "structure",
            "x_t",
            "x_mask",
            "t_emb",
            "token_single_input",
            "token_single_trunk",
            "token_pair_trunk",
        ]
        bound: dict[str, Any] = dict(kwargs)
        for name, value in zip(names, args):
            bound[name] = value
        return bound
