import torch
from jaxtyping import Bool, Float

from miniworld.data.features import MSAFeatures, SequenceFeatures

import torch.nn.functional as F

@torch.no_grad()
def init_msa_explicit(
    msa: MSAFeatures,
    token_embedding: torch.Tensor,              # (V, D)
    recycle_idx: int,
    profile32_to_fp_index: torch.Tensor,        # (32,), long
) -> tuple[Float[torch.Tensor, "B N L C"], Bool[torch.Tensor, "B N"]]:
    """Initialize MSA features using fingerprint embedding lookup (no one_hot)."""
    msa_mask = msa.msa_mask[:, recycle_idx]
    msa_sequences = msa.aligned_sequences[:, recycle_idx]      # (B, N, L), values in [0..31]
    msa_has_deletion = msa.has_deletion[:, recycle_idx]
    msa_deletion_value = msa.deletion_value[:, recycle_idx].float()

    device = msa.aligned_sequences.device
    emb = token_embedding.to(device).float()
    idx32 = profile32_to_fp_index.to(device)

    # Map canonical class ids (0..31) -> fingerprint table row ids (0..V-1)
    msa_sequences = msa_sequences.long().clamp_(0, idx32.numel() - 1)
    fp_ids = idx32[msa_sequences]                                  # (B, N, L)
    msa_sequences_fp = F.embedding(fp_ids, emb)                    # (B, N, L, D)

    msa_feat = torch.cat(
        [
            msa_sequences_fp,
            msa_has_deletion.unsqueeze(-1),
            msa_deletion_value.unsqueeze(-1),
        ],
        dim=-1,
    )
    msa_feat = msa_feat * msa_mask[:, :, None, None]
    return msa_feat.float(), msa_mask.bool()

@torch.no_grad()
def init_token_single_msa_explicit(
    msa: MSAFeatures,
    sequence: SequenceFeatures,
    token_embedding: torch.Tensor,
    profile32_to_fp_index: torch.Tensor,    # (32,), long
    token_type_is_fp_index: bool = True,
) -> Float[torch.Tensor, "B L_token d_single_token_init"]:
    device = msa.aligned_sequences.device
    dtype = msa.profile.dtype
    emb = token_embedding.to(device).float()
    idx32 = profile32_to_fp_index.to(device).long()

    # 1) token identity in fingerprint space
    token_ids = sequence.token_type.long()
    if not token_type_is_fp_index:
        token_ids = idx32[token_ids.clamp(0, idx32.numel() - 1)]
    token_fp = F.embedding(token_ids, emb).to(dtype=dtype) 

    # 2) msa.profile (B, L, 32) -> fingerprint-space msa_profile (B, L, D_fp)
    fp_rows_for_32 = emb.index_select(0, idx32)             # (32, D_fp)
    msa_profile = torch.einsum(
        "blc,cd->bld",
        msa.profile.float(),
        fp_rows_for_32,
    ).to(dtype=dtype)

    # 3) final token single init
    return torch.cat(
        [
            token_fp,
            msa_profile,
            msa.deletion_mean.unsqueeze(-1).to(dtype=dtype),
        ],
        dim=-1,
    )