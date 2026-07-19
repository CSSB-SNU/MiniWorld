"""AF3-style template embedder.

Faithful port of google-deepmind/alphafold3
``src/alphafold3/model/network/template_modules.py`` (TemplateEmbedding /
SingleTemplateEmbedding). Unlike the previous ``TemplateEmbedder`` — which took a
pre-assembled ``template_feat`` and never averaged over templates — this module builds
every AF3 template feature INTERNALLY from the raw ``TemplateFeatures`` (pseudo-beta
distogram, aatype one-hot in both axes, backbone-frame unit vectors, pseudo-beta /
backbone masks), projects each feature (+ the query pair) to ``num_channels``, runs a
pair-only Pairformer per template, LayerNorms, then AVERAGES over the valid templates,
applies ReLU, and projects back to ``d_pair``.

Template signal is restricted to intra-chain token pairs via a multichain mask
(AF3 ``multichain_mask_2d``). All shapes are static (no variable-length scatter), so the
whole module is CUDA-graph capturable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn

from team_gm import typecheck
from team_gm.modules import MiniPairformer
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.primitives import LayerNorm, Linear

from miniworld.data.features import TemplateFeatures


def _dgram_from_positions(
    cb: Float[torch.Tensor, "B L 3"],
    lower: Float[torch.Tensor, "num_bins"],
    upper: Float[torch.Tensor, "num_bins"],
) -> Float[torch.Tensor, "B L L num_bins"]:
    """AF3 dgram_from_positions: squared-distance histogram over pseudo-beta coords.

    ``lower``/``upper`` are precomputed squared-distance bin edges (module buffers), so
    nothing is allocated per forward — creating them here (``torch.linspace`` /
    ``new_tensor``) does a host->device copy that is illegal under CUDA-graph capture.
    """
    diff = cb[:, :, None, :] - cb[:, None, :, :]
    dist2 = diff.pow(2).sum(dim=-1)  # [B, L, L]
    dgram = (dist2[..., None] > lower) & (dist2[..., None] < upper)
    return dgram.to(cb.dtype)


def _backbone_unit_vectors(
    bb: Float[torch.Tensor, "B L 3 3"],
) -> Float[torch.Tensor, "B L L 3"]:
    """Unit vector from residue i's backbone frame to residue j's CA (AF3 style).

    ``bb`` holds the backbone (N, CA, C) coordinates. Builds a per-residue orthonormal
    frame (Gram-Schmidt on C-CA / N-CA), then expresses CA_j in frame_i and normalizes.
    """
    n, ca, c = bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]
    e1 = F.normalize(c - ca, dim=-1, eps=1e-8)
    v2 = n - ca
    u2 = v2 - (e1 * v2).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1, eps=1e-8)
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    rot = torch.stack([e1, e2, e3], dim=-1)  # [B, L, 3, 3], columns = frame axes
    diff = ca[:, None, :, :] - ca[:, :, None, :]  # [B, i, j, 3] = CA_j - CA_i
    # components of diff in frame_i: (R_i^T diff)_x = sum_r R[b,i,r,x] diff[b,i,j,r]
    vec = torch.einsum("birx,bijr->bijx", rot, diff)  # [B, L, L, 3]
    return F.normalize(vec, dim=-1, eps=1e-8)


class AF3TemplateEmbedder(nn.Module):
    """AF3 TemplateEmbedding: build features from raw templates, embed, average."""

    def __init__(
        self,
        d_pair: int,
        *,
        num_channels: int = 64,
        num_res_class: int = 32,
        n_block: int = 2,
        dgram_min: float = 3.25,
        dgram_max: float = 50.75,
        dgram_bins: int = 39,
        dropout_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_res_class = num_res_class
        self.dgram_min = dgram_min
        self.dgram_max = dgram_max
        self.dgram_bins = dgram_bins

        # Precompute the squared-distance bin edges ONCE as buffers so the dgram op
        # allocates nothing per forward (host->device tensor creation breaks CUDA-graph
        # capture). Moved to the right device/dtype with the module.
        _lower = torch.linspace(dgram_min, dgram_max, dgram_bins) ** 2
        _upper = torch.cat([_lower[1:], _lower.new_tensor([float("inf")])])
        self.register_buffer("_dgram_lower", _lower, persistent=False)
        self.register_buffer("_dgram_upper", _upper, persistent=False)

        # Per-feature input projections (AF3 sums independent relu-init projections).
        self.ln_query = LayerNorm(d_pair)
        self.proj_query = Linear(d_pair, num_channels, bias=False, init="relu")
        self.proj_dgram = Linear(dgram_bins, num_channels, bias=False, init="relu")
        self.proj_pb_mask = Linear(1, num_channels, bias=False, init="relu")
        self.proj_aatype_i = Linear(num_res_class, num_channels, bias=False, init="relu")
        self.proj_aatype_j = Linear(num_res_class, num_channels, bias=False, init="relu")
        self.proj_unit_vec = Linear(3, num_channels, bias=False, init="relu")
        self.proj_bb_mask = Linear(1, num_channels, bias=False, init="relu")

        # Pair-only trunk per template. MiniPairformer is trimul + transition (NO
        # triangle attention), routed through miniworld-kernels — same backend as the
        # main trunk, so the template stack carries no cuequivariance dependency.
        self.template_pairformer = MiniPairformer(
            MiniPairformer.Config(
                d_pair=num_channels,
                n_block=n_block,
                p_drop=dropout_prob,
                implementation=ImplementationType.TRITON,  # miniworld-kernels whole-op
            ),
        )
        self.ln_out = LayerNorm(num_channels)
        self.proj_out = Linear(num_channels, d_pair, bias=False, init="relu")

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        template: TemplateFeatures,
        token_asym_id: Int[torch.Tensor, "B L"],
        token_mask: Bool[torch.Tensor, "B L"],
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Embed and average all templates into a pair update.

        Each of the ``N_temp`` templates is embedded through the per-template trunk
        SEPARATELY (Python loop over a fixed ``N_temp``) rather than folded into the
        batch dim: the trunk is a miniworld-kernels MiniPairformer whose bidirectional
        trimul kernel only supports batch size 1, so the template axis cannot be folded
        into B. ``N_temp`` is static, so the unrolled loop stays CUDA-graph capturable.
        """
        b, n_temp = template.mask.shape
        dtype = pair.dtype

        # Query pair + intra-chain mask are shared across templates (per-B).
        query = self.proj_query(self.ln_query(pair))  # [B, L, L, C]
        multichain = (
            token_asym_id[:, :, None] == token_asym_id[:, None, :]
        ).to(dtype)[..., None]  # [B, L, L, 1]

        summed = query.new_zeros((b, query.shape[1], query.shape[2], self.num_channels))
        for t in range(n_temp):
            # Invalid templates (mask=0) and missing residues can carry NaN/Inf coords.
            # Sanitize BEFORE any math: F.normalize(NaN)=NaN would flow into ``act`` and
            # then ``act * mask`` gives NaN*0 = NaN (IEEE), poisoning the whole average.
            # nan_to_num keeps everything finite; per-residue validity is still carried by
            # the pb/bb mask features, and whole-template validity by the ``mask`` multiply.
            cb = torch.nan_to_num(template.cb_xyz[:, t])  # [B, L, 3]
            cb_mask = template.cb_mask[:, t]  # [B, L]
            res_type = template.res_type[:, t].clamp(0, self.num_res_class - 1)
            bb = torch.nan_to_num(template.bb_xyz[:, t])  # [B, L, 3, 3]
            bb_mask = template.bb_mask[:, t]  # [B, L]

            dgram = _dgram_from_positions(
                cb, self._dgram_lower, self._dgram_upper,
            ).to(dtype)
            pb2d = (cb_mask[:, :, None] & cb_mask[:, None, :]).to(dtype)[..., None]
            aatype = F.one_hot(res_type, self.num_res_class).to(dtype)  # [B, L, C_res]
            unit_vec = _backbone_unit_vectors(bb).to(dtype)  # [B, L, L, 3]
            bb2d = (bb_mask[:, :, None] & bb_mask[:, None, :]).to(dtype)[..., None]

            act = (
                query
                + self.proj_dgram(dgram)
                + self.proj_pb_mask(pb2d)
                + self.proj_aatype_i(aatype)[:, None, :, :]  # broadcast over i
                + self.proj_aatype_j(aatype)[:, :, None, :]  # broadcast over j
                + self.proj_unit_vec(unit_vec)
                + self.proj_bb_mask(bb2d)
            )
            act = act * multichain
            act = self.template_pairformer(act, mask=token_mask)  # B=1 -> miniworld
            act = self.ln_out(act)
            # Accumulate only the VALID templates (AF3 average over valid).
            summed = summed + act * template.mask[:, t].to(dtype)[:, None, None, None]

        n_valid = template.mask.to(dtype).sum(dim=1)[:, None, None, None]  # [B,1,1,1]
        avg = summed / (1e-7 + n_valid)
        return self.proj_out(F.relu(avg))
