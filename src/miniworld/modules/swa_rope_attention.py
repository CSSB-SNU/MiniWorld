"""Sliding-window atom attention with 3D RoPE and QK-norm (ESMFold2-style).

This is an opt-in alternative to the pair-bias atom attention used by the
diffusion module's ``AtomAttentionEncoder`` / ``AtomAttentionDecoder``.

It transplants the three ESMFold2 atom-attention techniques:

  * **QK-norm** -- non-affine RMSNorm on per-head Q/K (before RoPE).
  * **3D RoPE** -- rotary embedding built from the *reference* 3D atom
    coordinates + a per-atom space UID (``ref_pos`` + ``ref_space_uid``),
    so attention is conditioned on reference geometry instead of pair bias.
  * **bf16 local attention without pair bias** -- sliding-window attention
    (no pair-bias term enters the logits) computed in bf16.

Reference: ``Biohub/transformers`` ``modeling_esmfold2_common.py``
(``SWA3DRoPEAttention`` / ``SWAAtomBlock`` / ``SWAAtomTransformer``).

The local-attention backend is selectable (``flex`` | ``sdpa`` | ``flash``):

  * ``flex``  -- native ``torch.nn.attention.flex_attention`` block-sparse
    sliding window. Default; no external dependency, avoids the O(L^2)
    dense mask.
  * ``sdpa``  -- dense banded ``scaled_dot_product_attention`` mask. Simplest
    and always correct, but materialises an O(L^2) mask (only for small L).
  * ``flash`` -- FlashAttention-4 (``flash_attn.cute.flash_attn_varlen_func``)
    with ``window_size`` and per-molecule packing via cu_seqlens. FA4 is the
    Hopper/Blackwell line (supports sm_100 / B200); install ``flash-attn-4``.
    Falls back to ``flex`` if unavailable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Optional backends, detected lazily so import never fails.
# ---------------------------------------------------------------------------
try:  # native to torch >= 2.5, efficient block-sparse sliding window
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    _FLEX_AVAILABLE = True
    # dynamic=True keeps recompiles bounded as atom counts vary across steps.
    _flex_attention_compiled = torch.compile(flex_attention, dynamic=True)
except Exception:  # pragma: no cover - older torch
    _FLEX_AVAILABLE = False

try:
    # FlashAttention-4 (CuTeDSL) — the Hopper/Blackwell line. Pure-Python
    # wheel (`flash-attn-4`), JIT-compiles CUTLASS kernels, so it is not
    # pinned to the torch ABI and supports sm_100 (B200). Returns (out, lse).
    from flash_attn.cute import flash_attn_varlen_func  # type: ignore

    _FLASH_AVAILABLE = True
except Exception:
    _FLASH_AVAILABLE = False


# ===========================================================================
# 3D RoPE + QK-norm primitives (ported from ESMFold2, kept bit-faithful)
# ===========================================================================


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_3d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE with batch-dependent cos/sin.

    Args:
        x: ``[B, L, H, D]``
        cos: ``[B, L, D/2]``
        sin: ``[B, L, D/2]``

    """
    ro_dim = cos.shape[-1] * 2
    cos = cos.unsqueeze(2).repeat(1, 1, 1, 2)
    sin = sin.unsqueeze(2).repeat(1, 1, 1, 2)
    return torch.cat(
        [x[..., :ro_dim] * cos + _rotate_half(x[..., :ro_dim]) * sin, x[..., ro_dim:]],
        dim=-1,
    )


@torch.compiler.disable
def build_3d_rope(
    ref_pos: Tensor,
    ref_space_uid: Tensor,
    head_dim: int,
    n_spatial_per_axis: int = 2,
    n_uid_pairs: int = 10,
    spatial_base_freq: float = 20.0,
    uid_base_freq: float = 10000.0,
) -> tuple[Tensor, Tensor]:
    """Build cos/sin for 3D spatial RoPE + UID RoPE.

    Args:
        ref_pos: ``[B, N, 3]`` reference atom coordinates.
        ref_space_uid: ``[B, N]`` per-atom space/chain id.
        head_dim: attention head dim ``D``; cos/sin have last dim ``D/2``.

    """
    device = ref_pos.device
    B, N = ref_pos.shape[:2]
    half_dim = head_dim // 2
    n_spatial_total = 3 * n_spatial_per_axis

    spatial_inv_freq = 1.0 / (
        spatial_base_freq
        ** (
            torch.arange(0, n_spatial_per_axis, dtype=torch.float32, device=device)
            / n_spatial_per_axis
        )
    )
    uid_inv_freq = 1.0 / (
        uid_base_freq
        ** (torch.arange(0, n_uid_pairs, dtype=torch.float32, device=device) / n_uid_pairs)
    )

    pos_f32 = ref_pos.float()
    spatial_freqs = torch.einsum("bna,k->bnak", pos_f32, spatial_inv_freq)
    spatial_freqs = spatial_freqs.reshape(B, N, n_spatial_total)

    uid_f32 = ref_space_uid.float()
    uid_freqs = torch.einsum("bn,k->bnk", uid_f32, uid_inv_freq)

    n_active = n_spatial_total + n_uid_pairs
    if n_active > half_dim:
        msg = (
            f"3D RoPE active freqs ({n_active}) exceed head_dim/2 ({half_dim}); "
            f"reduce n_spatial_pairs_per_axis / n_uid_pairs or raise head_dim."
        )
        raise ValueError(msg)
    freqs = torch.cat([spatial_freqs, uid_freqs], dim=-1)

    if n_active < half_dim:
        padding = torch.zeros(
            B, N, half_dim - n_active, device=device, dtype=torch.float32,
        )
        freqs = torch.cat([freqs, padding], dim=-1)

    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    return cos, sin


def qk_norm(x: Tensor) -> Tensor:
    """Non-affine RMSNorm over the head dim (ESMFold2 QK-norm)."""
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)


# ===========================================================================
# SwiGLU FFN (atom transformer blocks)
# ===========================================================================


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN with rounded hidden size for hardware alignment."""

    def __init__(self, d_model: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        hidden_size = ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256
        self.w_up = nn.Linear(d_model, 2 * hidden_size, bias=False)
        self.w_down = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.w_up.weight.dtype)
        x1, x2 = self.w_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(x1) * x2)


# ===========================================================================
# Attention params -- step-invariant tensors shared across atom encoder /
# decoder for one forward (cos/sin + masking metadata).
# ===========================================================================


def build_attention_params(
    ref_pos: Tensor,
    ref_space_uid: Tensor,
    atom_mask: Tensor,
    head_dim: int,
    *,
    n_spatial_per_axis: int,
    n_uid_pairs: int,
    spatial_base_freq: float,
    uid_base_freq: float,
    n_repeat: int = 1,
) -> dict:
    """Precompute cos/sin and masking metadata once per forward.

    Args:
        ref_pos: ``[B, N, 3]`` reference coords.
        ref_space_uid: ``[B, N]`` space id.
        atom_mask: ``[B, N]`` bool validity.
        n_repeat: tile batch by this factor (= num augmentation samples) so
            the result aligns with ``[A*B, N, ...]`` atom tensors.

    """
    cos, sin = build_3d_rope(
        ref_pos=ref_pos,
        ref_space_uid=ref_space_uid,
        head_dim=head_dim,
        n_spatial_per_axis=n_spatial_per_axis,
        n_uid_pairs=n_uid_pairs,
        spatial_base_freq=spatial_base_freq,
        uid_base_freq=uid_base_freq,
    )
    valid = atom_mask.bool()
    doc = ref_space_uid.long()
    if n_repeat > 1:
        cos = cos.repeat(n_repeat, 1, 1)
        sin = sin.repeat(n_repeat, 1, 1)
        valid = valid.repeat(n_repeat, 1)
        doc = doc.repeat(n_repeat, 1)
    return {"cos": cos, "sin": sin, "valid": valid, "doc": doc}


# ===========================================================================
# SWA3DRoPEAttention
# ===========================================================================


class SWA3DRoPEAttention(nn.Module):
    """Sliding-window attention with 3D RoPE and QK-norm. No pair bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        half_window: int = 64,
        backend: str = "flex",
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.half_window = half_window
        self.backend = backend

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

    def _resolve_backend(self) -> str:
        if self.backend == "flash" and not _FLASH_AVAILABLE:
            return "flex" if _FLEX_AVAILABLE else "sdpa"
        if self.backend == "flex" and not _FLEX_AVAILABLE:
            return "sdpa"
        return self.backend

    def forward(self, x: Tensor, attention_params: dict) -> Tensor:
        """``x``: ``[B, N, d_model]`` (caller flattens any augmentation axis)."""
        B, N = x.shape[:2]
        cos, sin = attention_params["cos"], attention_params["sin"]

        x_input = x
        qkv = self.Wqkv(x)
        qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)  # each [B, N, H, D]
        q, k = qk_norm(q), qk_norm(k)

        q = apply_rotary_emb_3d(q, cos, sin)
        k = apply_rotary_emb_3d(k, cos, sin)

        input_dtype = q.dtype
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        backend = self._resolve_backend()
        if backend == "flash":
            out = self._flash_forward(q, k, v, attention_params, B, N)
        elif backend == "flex":
            out = self._flex_forward(q, k, v, attention_params)
        else:
            out = self._sdpa_forward(q, k, v, attention_params, N)

        out = out.to(input_dtype).reshape(B, N, -1)
        out = out * torch.sigmoid(self.gate_proj(x_input))
        return self.out_proj(out)

    # --- backends ----------------------------------------------------------

    def _flex_forward(self, q: Tensor, k: Tensor, v: Tensor, p: dict) -> Tensor:
        # q,k,v: [B, N, H, D] -> [B, H, N, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        B, _, N = q.shape[:3]
        valid, doc, hw = p["valid"], p["doc"], self.half_window

        def mask_mod(b, h, qi, kj):  # noqa: ANN001, ANN202
            within = (qi - kj).abs() <= hw
            same = doc[b, qi] == doc[b, kj]
            ok = valid[b, qi] & valid[b, kj]
            return (within & same & ok) | (qi == kj)

        block_mask = create_block_mask(
            mask_mod, B=B, H=None, Q_LEN=N, KV_LEN=N, device=q.device,
        )
        out = _flex_attention_compiled(q, k, v, block_mask=block_mask, scale=self.scale)
        return out.transpose(1, 2)  # back to [B, N, H, D]

    def _sdpa_forward(self, q: Tensor, k: Tensor, v: Tensor, p: dict, N: int) -> Tensor:
        valid, doc, hw = p["valid"], p["doc"], self.half_window
        idx = torch.arange(N, device=q.device)
        within = (idx[None, :, None] - idx[None, None, :]).abs() <= hw
        same = doc[:, :, None] == doc[:, None, :]
        ok = valid[:, :, None] & valid[:, None, :]
        allowed = (within & same & ok) | torch.eye(
            N, dtype=torch.bool, device=q.device,
        )
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=allowed.unsqueeze(1),
            scale=self.scale,
        ).transpose(1, 2)
        return out * valid[..., None, None]

    def _flash_forward(
        self, q: Tensor, k: Tensor, v: Tensor, p: dict, B: int, N: int,
    ) -> Tensor:
        """FlashAttention-4 varlen with per-molecule packing + sliding window.

        Valid atoms are packed into contiguous segments, one per
        (sample, space_uid), so the window never crosses a molecule
        boundary — matching the flex / sdpa backends. q/k/v arrive as
        [B, N, H, D].
        """
        valid, doc = p["valid"], p["doc"]
        device = q.device
        H, D = self.n_heads, self.head_dim

        # Segment id unique per (row, doc); pack valid atoms grouped by segment,
        # preserving within-segment order via a stable sort (robust to any
        # atom ordering / interleaved space_uids).
        n_doc = int(doc.max().item()) + 1
        row = torch.arange(B, device=device).view(B, 1).expand(B, N)
        seg = (row * n_doc + doc).reshape(-1)  # [B*N]
        flat_valid = valid.reshape(-1)
        sel = flat_valid.nonzero(as_tuple=False).flatten()  # [n_valid]
        order = torch.argsort(seg[sel], stable=True)
        packed_idx = sel[order]  # [n_valid] indices into [B*N], grouped by seg
        seg_sorted = seg[packed_idx]
        _, counts = torch.unique_consecutive(seg_sorted, return_counts=True)
        cu = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
        max_seqlen = int(counts.max().item())

        qf = q.reshape(B * N, H, D)
        kf = k.reshape(B * N, H, D)
        vf = v.reshape(B * N, H, D)
        out_packed, _ = flash_attn_varlen_func(
            qf[packed_idx],
            kf[packed_idx],
            vf[packed_idx],
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scale,
            causal=False,
            window_size=(self.half_window, self.half_window),
        )
        out = q.new_zeros(B * N, H, D)
        out[packed_idx] = out_packed.to(out.dtype)
        return out.reshape(B, N, H, D)


# ===========================================================================
# SWAAtomBlock / SWAAtomTransformer (adaLN-Zero + SWA attn + SwiGLU FFN)
# ===========================================================================


class SWAAtomBlock(nn.Module):
    """adaLN-Zero conditioning + SWA attention + SwiGLU FFN."""

    def __init__(
        self,
        d_atom: int,
        n_heads: int,
        half_window: int = 64,
        expansion_ratio: int = 2,
        backend: str = "flex",
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_atom, elementwise_affine=False)
        self.ffn_norm = nn.RMSNorm(d_atom, elementwise_affine=False)

        adaln_linear = nn.Linear(d_atom, 6 * d_atom, bias=False)
        nn.init.zeros_(adaln_linear.weight)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), adaln_linear)

        self.attn = SWA3DRoPEAttention(
            d_atom, n_heads, half_window=half_window, backend=backend,
        )
        self.ffn = SwiGLUFFN(d_atom, expansion_ratio)

    def forward(self, x: Tensor, c_l: Tensor, attention_params: dict) -> Tensor:
        mod = self.adaln_modulation(c_l)
        if mod.dim() == 2:
            mod = mod.unsqueeze(1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.chunk(6, dim=-1)

        attn_input = self.attn_norm(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(attn_input, attention_params)

        ffn_input = self.ffn_norm(x) * (1 + scale_f) + shift_f
        x = x + gate_f * self.ffn(ffn_input)
        return x


class SWAAtomTransformer(nn.Module):
    """Stack of SWAAtomBlocks. Drop-in for the atom-level DiffusionTransformer.

    Unlike the pair-bias transformer, this consumes no ``pair`` tensor; it is
    conditioned on ``cond`` (adaLN) and reference geometry (3D RoPE carried in
    ``attention_params``).
    """

    def __init__(
        self,
        d_atom: int,
        n_blocks: int,
        n_heads: int,
        *,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        backend: str = "flex",
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SWAAtomBlock(
                    d_atom=d_atom,
                    n_heads=n_heads,
                    half_window=swa_window_size // 2,
                    expansion_ratio=expansion_ratio,
                    backend=backend,
                )
                for _ in range(n_blocks)
            ],
        )

    def forward(self, q_l: Tensor, c_l: Tensor, attention_params: dict) -> Tensor:
        for block in self.blocks:
            q_l = block(q_l, c_l, attention_params)
        return q_l
