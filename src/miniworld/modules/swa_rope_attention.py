"""Sliding-window atom attention with 3D RoPE and QK-norm.

An opt-in alternative to the pair-bias atom attention used by the diffusion
module's ``AtomAttentionEncoder`` / ``AtomAttentionDecoder``. It conditions
attention on the *reference* geometry through rotary position embeddings
instead of an additive pair bias, and restricts each atom to a local window:

* **QK-norm** -- RMSNorm (no affine) on the per-head query/key before the
  rotary embedding, to keep the logits well scaled.
* **3D RoPE** -- rotary embedding whose angles come from the reference atom
  coordinates (one frequency band per spatial axis) plus the per-atom space
  uid, so relative geometry enters attention directly. No pair tensor.
* **Sliding-window attention in bf16** -- each query attends only to keys
  within ``+/- swa_window_size // 2`` and never across a space-uid boundary;
  no pair-bias term is added to the logits.

The local-attention backend is selectable (see ``AtomSWAConfig.backend``):

* ``flex``  -- ``torch.nn.attention.flex_attention`` block-sparse window
  (native, no extra dependency, avoids the dense L x L mask). Default.
* ``sdpa``  -- dense banded ``scaled_dot_product_attention`` (materialises an
  L x L mask; only sensible for small L / as a reference).
* ``flash`` -- FlashAttention-4 (``flash_attn.cute``) varlen with a window;
  packs valid atoms per space-uid. Supports sm_100 (B200). Falls back to
  ``flex`` when the wheel is absent.

The 3D-RoPE construction is adapted from the ESMFold2 atom encoder; the rest
follows the conventions of ``team_gm.modules.layers`` attention modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Int
from team_gm import typecheck
from team_gm.modules.layers.ops import sigmoid_gate
from team_gm.modules.primitives import Linear
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Optional accelerated backends, probed lazily so importing never fails.
# ---------------------------------------------------------------------------
try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    _FLEX_AVAILABLE = True
    # dynamic=True keeps recompiles bounded as the atom count varies per step.
    _flex_attention = torch.compile(flex_attention, dynamic=True)
except Exception:  # noqa: BLE001 - older torch without flex_attention
    _FLEX_AVAILABLE = False

try:
    # FlashAttention-4 (CuTeDSL): a pure-Python wheel that JIT-compiles CUTLASS
    # kernels (not pinned to the torch ABI) and supports Hopper/Blackwell.
    from flash_attn.cute import flash_attn_varlen_func

    _FLASH_AVAILABLE = True
except Exception:  # noqa: BLE001 - flash-attn-4 not installed
    _FLASH_AVAILABLE = False


# ===========================================================================
# 3D rotary position embedding
# ===========================================================================


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of the last dimension: ``[a, b] -> [-b, a]``."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@typecheck
def apply_rotary_3d(
    x: Float[Tensor, "B L H D"],
    cos: Float[Tensor, "B L D_half"],
    sin: Float[Tensor, "B L D_half"],
) -> Float[Tensor, "B L H D"]:
    """Apply the rotary embedding given per-token ``cos``/``sin`` tables."""
    rotary_dim = cos.shape[-1] * 2
    cos = cos.unsqueeze(2).repeat(1, 1, 1, 2)  # [B, L, 1, D]
    sin = sin.unsqueeze(2).repeat(1, 1, 1, 2)
    rotated = x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim]) * sin
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


@torch.compiler.disable
def build_3d_rope(
    ref_pos: Float[Tensor, "B L 3"],
    ref_space_uid: Int[Tensor, "B L"],
    d_hidden: int,
    n_spatial_per_axis: int,
    n_uid_pairs: int,
    spatial_base_freq: float,
    uid_base_freq: float,
) -> tuple[Float[Tensor, "B L D_half"], Float[Tensor, "B L D_half"]]:
    """Build ``cos``/``sin`` tables for the spatial + uid rotary embedding.

    Each of the three spatial axes contributes ``n_spatial_per_axis`` rotary
    pairs and the space uid contributes ``n_uid_pairs``; the remaining
    head-dim pairs (up to ``d_hidden // 2``) are zero-frequency (identity).
    """
    device = ref_pos.device
    batch, length = ref_pos.shape[:2]
    d_half = d_hidden // 2
    n_spatial = 3 * n_spatial_per_axis
    n_active = n_spatial + n_uid_pairs
    if n_active > d_half:
        msg = (
            f"3D RoPE needs {n_active} frequency pairs but d_hidden // 2 = "
            f"{d_half}; lower n_spatial_per_axis / n_uid_pairs or raise d_hidden."
        )
        raise ValueError(msg)

    spatial_inv_freq = 1.0 / (
        spatial_base_freq
        ** (torch.arange(n_spatial_per_axis, device=device).float() / n_spatial_per_axis)
    )
    uid_inv_freq = 1.0 / (
        uid_base_freq
        ** (torch.arange(n_uid_pairs, device=device).float() / n_uid_pairs)
    )

    spatial_angles = torch.einsum("bla,k->blak", ref_pos.float(), spatial_inv_freq)
    spatial_angles = spatial_angles.reshape(batch, length, n_spatial)
    uid_angles = torch.einsum("bl,k->blk", ref_space_uid.float(), uid_inv_freq)

    angles = torch.cat([spatial_angles, uid_angles], dim=-1)
    if n_active < d_half:
        pad = angles.new_zeros(batch, length, d_half - n_active)
        angles = torch.cat([angles, pad], dim=-1)

    return angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)


# ===========================================================================
# SwiGLU feed-forward
# ===========================================================================


class SwiGLU(nn.Module):
    """SwiGLU feed-forward with a hardware-aligned hidden size."""

    def __init__(self, d_model: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        hidden = ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256
        self.to_hidden = Linear(d_model, 2 * hidden, bias=False, init="relu")
        self.to_out = Linear(hidden, d_model, bias=False, init="zero")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = x.to(self.to_hidden.weight.dtype)
        gate, value = self.to_hidden(x).chunk(2, dim=-1)
        return self.to_out(F.silu(gate) * value)


# ===========================================================================
# Sliding-window attention with 3D RoPE
# ===========================================================================


class SlidingWindowAttention(nn.Module):
    """Local self-attention with 3D RoPE and QK-norm, no pair bias.

    Parameters
    ----------
    d_single : int
        Input / output dimension.
    n_head : int
        Number of attention heads.
    half_window : int
        Each query attends to keys within ``+/- half_window``.
    backend : str
        ``"flex"`` | ``"sdpa"`` | ``"flash"``.

    """

    def __init__(
        self,
        d_single: int,
        n_head: int,
        half_window: int,
        backend: str = "flex",
    ) -> None:
        super().__init__()
        if d_single % n_head != 0:
            msg = f"{d_single=} must be divisible by {n_head=}"
            raise ValueError(msg)
        self.n_head = n_head
        self.d_hidden = d_single // n_head
        self.scale = self.d_hidden**-0.5
        self.half_window = half_window
        self.backend = backend

        self.to_query = Linear(d_single, d_single, bias=False, init="glorot")
        self.to_key = Linear(d_single, d_single, bias=False, init="glorot")
        self.to_value = Linear(d_single, d_single, bias=False, init="glorot")
        self.norm_query = nn.RMSNorm(self.d_hidden, elementwise_affine=False)
        self.norm_key = nn.RMSNorm(self.d_hidden, elementwise_affine=False)
        self.to_gate = Linear(d_single, d_single, bias=False, init="gating")
        self.to_out = Linear(d_single, d_single, bias=False, init="zero")

    def _resolve_backend(self) -> str:
        if self.backend == "flash" and not _FLASH_AVAILABLE:
            return "flex" if _FLEX_AVAILABLE else "sdpa"
        if self.backend == "flex" and not _FLEX_AVAILABLE:
            return "sdpa"
        return self.backend

    @typecheck
    def forward(
        self,
        single: Float[Tensor, "B L d_single"],
        attention_params: dict,
    ) -> Float[Tensor, "B L d_single"]:
        """Forward pass; ``single`` has any augmentation axis flattened into B."""
        cos, sin = attention_params["cos"], attention_params["sin"]

        query = rearrange(self.to_query(single), "B L (H D) -> B L H D", H=self.n_head)
        key = rearrange(self.to_key(single), "B L (H D) -> B L H D", H=self.n_head)
        value = rearrange(self.to_value(single), "B L (H D) -> B L H D", H=self.n_head)

        query, key = self.norm_query(query), self.norm_key(key)
        query, key = apply_rotary_3d(query, cos, sin), apply_rotary_3d(key, cos, sin)
        query, key, value = query.bfloat16(), key.bfloat16(), value.bfloat16()

        backend = self._resolve_backend()
        if backend == "flash":
            out = self._attend_flash(query, key, value, attention_params)
        elif backend == "flex":
            out = self._attend_flex(query, key, value, attention_params)
        else:
            out = self._attend_sdpa(query, key, value, attention_params)

        out = rearrange(out.to(single.dtype), "B L H D -> B L (H D)")
        return self.to_out(sigmoid_gate(self.to_gate(single), out))

    # -- backends (each takes / returns [B, L, H, D]) -----------------------

    def _attend_flex(self, query: Tensor, key: Tensor, value: Tensor, p: dict) -> Tensor:
        valid, space_uid, half_window = p["valid"], p["space_uid"], self.half_window
        query = rearrange(query, "B L H D -> B H L D")
        key = rearrange(key, "B L H D -> B H L D")
        value = rearrange(value, "B L H D -> B H L D")
        batch, _, length = query.shape[:3]

        def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:  # noqa: ARG001
            within = (q_idx - kv_idx).abs() <= half_window
            same_space = space_uid[b, q_idx] == space_uid[b, kv_idx]
            both_valid = valid[b, q_idx] & valid[b, kv_idx]
            return (within & same_space & both_valid) | (q_idx == kv_idx)

        block_mask = create_block_mask(
            mask_mod, B=batch, H=None, Q_LEN=length, KV_LEN=length, device=query.device,
        )
        out = _flex_attention(query, key, value, block_mask=block_mask, scale=self.scale)
        return rearrange(out, "B H L D -> B L H D")

    def _attend_sdpa(self, query: Tensor, key: Tensor, value: Tensor, p: dict) -> Tensor:
        valid, space_uid, half_window = p["valid"], p["space_uid"], self.half_window
        length = query.shape[1]
        idx = torch.arange(length, device=query.device)
        within = (idx[:, None] - idx[None, :]).abs() <= half_window
        same_space = space_uid[:, :, None] == space_uid[:, None, :]
        both_valid = valid[:, :, None] & valid[:, None, :]
        allowed = (within[None] & same_space & both_valid) | torch.eye(
            length, dtype=torch.bool, device=query.device,
        )
        out = F.scaled_dot_product_attention(
            rearrange(query, "B L H D -> B H L D"),
            rearrange(key, "B L H D -> B H L D"),
            rearrange(value, "B L H D -> B H L D"),
            attn_mask=allowed.unsqueeze(1),
            scale=self.scale,
        )
        out = rearrange(out, "B H L D -> B L H D")
        return out * valid[..., None, None]

    def _attend_flash(self, query: Tensor, key: Tensor, value: Tensor, p: dict) -> Tensor:
        # FA4 varlen: pack valid atoms into contiguous segments, one per
        # (sample, space_uid), so the window never crosses a space boundary.
        valid, space_uid = p["valid"], p["space_uid"]
        batch, length = valid.shape
        device = query.device
        n_space = int(space_uid.max().item()) + 1
        row = rearrange(torch.arange(batch, device=device), "B -> B 1").expand(batch, length)
        segment = (row * n_space + space_uid).reshape(-1)

        selected = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        order = torch.argsort(segment[selected], stable=True)
        packed = selected[order]  # indices into [B*L], grouped by segment
        _, counts = torch.unique_consecutive(segment[packed], return_counts=True)
        cu_seqlens = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
        max_seqlen = int(counts.max().item())

        flat = lambda t: rearrange(t, "B L H D -> (B L) H D")[packed]
        out_packed, _ = flash_attn_varlen_func(
            flat(query), flat(key), flat(value),
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            softmax_scale=self.scale, causal=False,
            window_size=(self.half_window, self.half_window),
        )
        out = query.new_zeros(batch * length, self.n_head, self.d_hidden)
        out[packed] = out_packed.to(out.dtype)
        return rearrange(out, "(B L) H D -> B L H D", B=batch)


# ===========================================================================
# adaLN-Zero block and transformer stack
# ===========================================================================


class SWAAtomBlock(nn.Module):
    """adaLN-Zero conditioning + sliding-window attention + SwiGLU FFN."""

    def __init__(
        self,
        d_single: int,
        n_head: int,
        half_window: int,
        expansion_ratio: int = 2,
        backend: str = "flex",
    ) -> None:
        super().__init__()
        self.norm_attn = nn.RMSNorm(d_single, elementwise_affine=False)
        self.norm_ffn = nn.RMSNorm(d_single, elementwise_affine=False)
        self.to_modulation = nn.Sequential(
            nn.SiLU(),
            Linear(d_single, 6 * d_single, bias=False, init="zero"),
        )
        self.attention = SlidingWindowAttention(
            d_single, n_head, half_window=half_window, backend=backend,
        )
        self.transition = SwiGLU(d_single, expansion_ratio)

    @typecheck
    def forward(
        self,
        single: Float[Tensor, "B L d_single"],
        cond: Float[Tensor, "B L d_single"],
        attention_params: dict,
    ) -> Float[Tensor, "B L d_single"]:
        """Forward pass."""
        modulation = self.to_modulation(cond)
        if modulation.dim() == 2:
            modulation = modulation.unsqueeze(1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.chunk(6, dim=-1)

        attn_in = self.norm_attn(single) * (1 + scale_a) + shift_a
        single = single + gate_a * self.attention(attn_in, attention_params)

        ffn_in = self.norm_ffn(single) * (1 + scale_f) + shift_f
        return single + gate_f * self.transition(ffn_in)


class SWAAtomTransformer(nn.Module):
    """Stack of :class:`SWAAtomBlock`; drop-in for the atom-level DIT.

    Consumes no pair tensor: conditioned on ``cond`` (adaLN) and on reference
    geometry carried in ``attention_params`` (3D RoPE).
    """

    def __init__(
        self,
        d_atom: int,
        n_blocks: int,
        n_heads: int,
        *,
        swa_window_size: int,
        expansion_ratio: int = 2,
        backend: str = "flex",
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SWAAtomBlock(
                    d_atom, n_heads, half_window=swa_window_size // 2,
                    expansion_ratio=expansion_ratio, backend=backend,
                )
                for _ in range(n_blocks)
            ],
        )

    @typecheck
    def forward(
        self,
        single: Float[Tensor, "B L d_atom"],
        cond: Float[Tensor, "B L d_atom"],
        attention_params: dict,
    ) -> Float[Tensor, "B L d_atom"]:
        """Forward pass."""
        for block in self.blocks:
            single = block(single, cond, attention_params)
        return single


# ===========================================================================
# Shared attention parameters (cos/sin + masking metadata), built once.
# ===========================================================================


def build_attention_params(
    ref_pos: Float[Tensor, "B L 3"],
    ref_space_uid: Int[Tensor, "B L"],
    atom_mask: Bool[Tensor, "B L"],
    d_hidden: int,
    *,
    n_spatial_per_axis: int,
    n_uid_pairs: int,
    spatial_base_freq: float,
    uid_base_freq: float,
    n_repeat: int = 1,
) -> dict:
    """Precompute the rotary tables and masking metadata for one forward.

    ``n_repeat`` tiles the batch (= number of augmentation samples) so the
    result lines up with the ``[A * B, L, ...]`` atom tensors.
    """
    cos, sin = build_3d_rope(
        ref_pos, ref_space_uid, d_hidden,
        n_spatial_per_axis=n_spatial_per_axis, n_uid_pairs=n_uid_pairs,
        spatial_base_freq=spatial_base_freq, uid_base_freq=uid_base_freq,
    )
    valid = atom_mask.bool()
    space_uid = ref_space_uid.long()
    if n_repeat > 1:
        cos = cos.repeat(n_repeat, 1, 1)
        sin = sin.repeat(n_repeat, 1, 1)
        valid = valid.repeat(n_repeat, 1)
        space_uid = space_uid.repeat(n_repeat, 1)
    return {"cos": cos, "sin": sin, "valid": valid, "space_uid": space_uid}
